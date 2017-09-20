/** Breadth-first search -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 *
 * @section Description
 *
 * Breadth-first search.
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#include "galois/Galois.h"
#include "galois/Accumulator.h"
#include "galois/Timer.h"
#include "galois/Timer.h"
#include "galois/graphs/LCGraph.h"
#include "galois/graphs/TypeTraits.h"
#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

#include <iostream>

namespace cll = llvm::cl;

static const char* name = "Breadth-first Search";
static const char* desc =
  "Computes the shortest path from a source node to all nodes in a directed "
  "graph using a modified Bellman-Ford algorithm";
static const char* url = "breadth_first_search";

static cll::opt<std::string> filename(cll::Positional, 
                                      cll::desc("<input graph>"), 
                                      cll::Required);

static cll::opt<unsigned int> startNode("startNode",
                                        cll::desc("Node to start search from"),
                                        cll::init(0));
static cll::opt<unsigned int> reportNode("reportNode", 
                                         cll::desc("Node to report distance to"),
                                         cll::init(1));
static cll::opt<int> stepShift("delta",
                               cll::desc("Shift value for the deltastep"),
                               cll::init(10));

typedef galois::graphs::LC_InlineEdge_Graph<std::atomic<unsigned int>,void>::with_no_lockable<true>::type::with_compressed_node_ptr<true>::type::with_numa_alloc<true>::type Graph;

typedef Graph::GraphNode GNode;

static const bool trackWork = false;
static galois::Statistic* BadWork;
static galois::Statistic* WLEmptyWork;

#include "Lonestar/BFS_SSSP.h"


struct BFS {
  Graph& graph;
  BFS(Graph& g) : graph(g) {}
  void operator()(UpdateRequest& req,
                  galois::UserContext<UpdateRequest>& ctx) {
    const galois::MethodFlag flag = galois::MethodFlag::UNPROTECTED;
    Dist sdist = graph.getData(req.n, flag);
    
    if (req.w != sdist) {
      if (trackWork)
        *WLEmptyWork += 1;
      return;
    }
    
    for (auto ii : graph.edges(req.n, flag)) {
      GNode dst = graph.getEdgeDst(ii);
      auto& ddist  = graph.getData(dst, flag);
      Dist newDist = sdist + 1;
      Dist oldDist = ddist;
      while (newDist < oldDist) {
        if (ddist.compare_exchange_weak(oldDist, newDist, std::memory_order_relaxed)) {
          if (trackWork && oldDist != DIST_INFINITY)
            *BadWork += 1;
          ctx.push(UpdateRequest(dst, newDist));
        }
      }
    }
  }
};

int main(int argc, char** argv) {
  galois::StatManager statManager;
  LonestarStart(argc, argv, name, desc, url);

  if (trackWork) {
    BadWork     = new galois::Statistic("BadWork");
    WLEmptyWork = new galois::Statistic("EmptyWork");
  }

  galois::StatTimer T("OverheadTime");
  T.start();
  
  Graph graph;
  GNode source, report;

  galois::graphs::readGraph(graph, filename); 
  std::cout << "Read " << graph.size() << " nodes\n";

  if (startNode >= graph.size() || reportNode >= graph.size()) {
    std::cerr << "failed to set report: " << reportNode
              << " or failed to set source: " << startNode << "\n";
    assert(0);
    abort();
  }

  auto it = graph.begin();
  std::advance(it, startNode);
  source = *it;
  it = graph.begin();
  std::advance(it, reportNode);
  report = *it;

  size_t approxNodeData = graph.size() * 64;
  // size_t approxEdgeData = graph.sizeEdges() * sizeof(typename
  // Graph::edge_data_type) * 2;
  galois::preAlloc(numThreads +
                   approxNodeData / galois::runtime::pagePoolSize());
  galois::reportPageAlloc("MeminfoPre");

  std::cout << "Running Asynch with CAS version\n";
  std::cout << "INFO: Using delta-step of " << (1 << stepShift) << "\n";
  std::cout << "WARNING: Performance varies considerably due to delta parameter.\n";
  std::cout << "WARNING: Do not expect the default to be good for your graph.\n";
  galois::do_all_local(graph, 
                       [&graph] (GNode n) { graph.getData(n) = DIST_INFINITY; });
  graph.getData(source) = 0;
  galois::StatTimer Tmain;
  Tmain.start();

  using namespace galois::worklists;
  typedef dChunkedFIFO<64> dChunk;
  typedef OrderedByIntegerMetric<UpdateRequestIndexer,dChunk> OBIM;
  typedef BulkSynchronous<dChunkedLIFO<256> > BSWL;
  galois::for_each(UpdateRequest{source, 0}, BFS{graph}, galois::wl<BSWL>(), galois::does_not_need_aborts<>());
  Tmain.stop();
  T.stop();

  galois::reportPageAlloc("MeminfoPost");
  galois::runtime::reportNumaAlloc("NumaPost");
  
  std::cout << "Node " << reportNode << " has distance "
            << graph.getData(report) << "\n";

  if (!skipVerify) {
    if (verify<true>(graph, source)) {
      std::cout << "Verification successful.\n";
    } else {
      GALOIS_DIE("Verification failed");
    }
  }

  if (trackWork) {
    delete BadWork;
    delete WLEmptyWork;
  }

  return 0;
}
