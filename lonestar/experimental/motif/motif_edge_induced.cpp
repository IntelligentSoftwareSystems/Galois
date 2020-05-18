#include "galois/Galois.h"
#include "galois/Reduction.h"
#include "galois/Bag.h"
#include "galois/Timer.h"
#include "galois/graphs/LCGraph.h"
#include "galois/ParallelSTL.h"
#include "llvm/Support/CommandLine.h"
#include "galois/runtime/Profile.h"
#include <boost/iterator/transform_iterator.hpp>

const char* name = "Motif Counting";
const char* desc =
    "Counts the edge-induced motifs in a graph using BFS extension";
const char* url = 0;
namespace cll   = llvm::cl;
static cll::opt<std::string> filetype(cll::Positional,
                                      cll::desc("<filetype: txt,adj,mtx,gr>"),
                                      cll::Required);
static cll::opt<std::string>
    filename(cll::Positional, cll::desc("<filename: symmetrized graph>"),
             cll::Required);
static cll::opt<unsigned>
    k("k", cll::desc("max number of vertices in k-motif(default value 3)"),
      cll::init(3));
static cll::opt<unsigned> show("s", cll::desc("print out the details"),
                               cll::init(0));
typedef galois::graphs::LC_CSR_Graph<uint32_t, void>::with_numa_alloc<
    true>::type ::with_no_lockable<true>::type Graph;
typedef Graph::GraphNode GNode;

#define USE_STRUCTURAL
#define EDGE_INDUCED
#define CHUNK_SIZE 256
#include "BfsMining/edge_miner.h"
#include "util.h"

void MotifSolver(EdgeMiner& miner) {
  EmbeddingQueueType in_queue, out_queue; // in&out worklist. double buffering
  miner.init(in_queue);                   // initialize the worklist
  unsigned level = 1;
  galois::gPrint("\n");

  while (level <
         k) { // to get the same output as RStream (which is not complete)
    miner.extend_edge(in_queue, out_queue); // edge extension
    in_queue.swap(out_queue);
    out_queue.clear();

    if (show)
      std::cout << "\n------------------------ Step 2: Aggregation "
                   "------------------------\n";
    QpMapFreq qp_map;           // quick patterns map for counting the frequency
    LocalQpMapFreq qp_localmap; // quick patterns local map for each thread
    galois::do_all(
        galois::iterate(in_queue),
        [&](const EmbeddingType& emb) {
          miner.quick_aggregate_each(
              emb, *(qp_localmap.getLocal())); // quick pattern aggregation
        },
        galois::chunk_size<CHUNK_SIZE>(), galois::steal(),
        galois::no_conflicts(),
        galois::wl<galois::worklists::PerSocketChunkFIFO<CHUNK_SIZE>>(),
        galois::loopname("QuickAggregation"));
    miner.merge_qp_map(qp_localmap, qp_map);

    CgMapFreq cg_map;           // canonical graph map for couting the frequency
    LocalCgMapFreq cg_localmap; // canonical graph local map for each thread
    galois::do_all(
        galois::iterate(qp_map),
        [&](std::pair<QPattern, Frequency> element) {
          miner.canonical_aggregate_each(
              element,
              *(cg_localmap.getLocal())); // canonical pattern aggregation
        },
        galois::chunk_size<CHUNK_SIZE>(), galois::steal(),
        galois::no_conflicts(),
        galois::wl<galois::worklists::PerSocketChunkFIFO<CHUNK_SIZE>>(),
        galois::loopname("CanonicalAggregation"));
    miner.merge_cg_map(cg_localmap, cg_map);
    miner.printout_agg(cg_map);
    galois::gPrint("\n");
    level++;
  }
}

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url);
  Graph graph;
  galois::StatTimer Tinit("GraphReadingTime");
  Tinit.start();
  read_graph(graph, filetype, filename);
  Tinit.stop();
  galois::gPrint("num_vertices ", graph.size(), " num_edges ",
                 graph.sizeEdges(), "\n");

  EdgeMiner miner(&graph);
  galois::StatTimer Tcomp("Compute");
  Tcomp.start();
  MotifSolver(miner);
  Tcomp.stop();
  return 0;
}
