#ifndef PAGERANK_DETERMINISTIC_H
#define PAGERANK_DETERMINISTIC_H

#include "Galois/Accumulator.h"
#include "Galois/Timer.h"
#include "Galois/Statistic.h"
#include "Galois/Galois.h"
#include "Galois/DoAllWrap.h"
#include "Galois/Graph/Util.h"
#include "Galois/Graph/Graph.h"
// #include "Galois/Graph/FileGraph.h"

#include "Galois/Runtime/Sampling.h"
#include "Galois/Runtime/PerThreadContainer.h"
#include "Galois/Runtime/DetChromatic.h"
#include "Galois/Runtime/DetKDGexecutor.h"

#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

#include "PageRank.h"

#include <cstdio>

namespace cll = llvm::cl;

static cll::opt<std::string> inputFile (cll::Positional, cll::desc ("<input file>"), cll::Required);

static cll::opt<std::string> transposeFile ("transpose", cll::desc ("<transpose file>"), cll::Required);

static const char* const name = "Page Rank";
static const char* const desc = "Page Rank of a graph of web pages";
static const char* const url = "pagerank";

static const double PAGE_RANK_INIT = 1.0;

struct PData {
  float value;
  unsigned outdegree; 

  PData (void)
    : value (PAGE_RANK_INIT), outdegree (0) {}

  PData (unsigned outdegree)
    : value (PAGE_RANK_INIT), outdegree (outdegree) {}

  double getPageRank () const {
    return value;
  }

};

template <typename IG>
class PageRankBase {

protected:

  typedef typename Galois::Graph::LC_InOut_Graph<IG> Graph;
  typedef typename Graph::GraphNode GNode;
  typedef typename Graph::node_data_type NodeData;

  static const unsigned DEFAULT_CHUNK_SIZE = 1;

  Graph graph;

  void readGraph (void) {
    Galois::Graph::readGraph (graph, inputFile, transposeFile);

    std::printf ("Graph read with %zd nodes and %zd edges\n", 
        graph.size (), graph.sizeEdges ());
  }

  template <bool useOnWL, typename C>
  void applyOperator (GNode src, C& ctx) {
    double sum = 0;

    for (auto jj = graph.in_edge_begin(src, Galois::MethodFlag::UNPROTECTED), ej = graph.in_edge_end(src, Galois::MethodFlag::UNPROTECTED); jj != ej; ++jj) {
      GNode dst = graph.getInEdgeDst(jj);
      auto& ddata = graph.getData(dst, Galois::MethodFlag::UNPROTECTED);
      sum += ddata.value / ddata.outdegree;
    }

    float value = (1.0 - alpha) * sum + alpha;
    auto& sdata = graph.getData(src, Galois::MethodFlag::UNPROTECTED);
    float diff = std::fabs(value - sdata.value);

    
    
    if (diff > tolerance) {
      sdata.value = value;
      for (auto jj = graph.edge_begin(src, Galois::MethodFlag::UNPROTECTED), ej = graph.edge_end(src, Galois::MethodFlag::UNPROTECTED); jj != ej; ++jj) {
        GNode dst = graph.getEdgeDst(jj);

        if (useOnWL) {
          auto& dd = graph.getData (dst, Galois::MethodFlag::UNPROTECTED);
          bool expected = false;
          if (dd.onWL.compare_exchange_strong (expected, true)) {
            ctx.push (dst);
          }

        } else {
          ctx.push(dst);
        }
      }
    } 

    if (useOnWL) {
      sdata.onWL = false;
    }

  }

  void verify (void) {
    if (skipVerify) {
      std::printf ("Verification skipped\n");
      return;
    }

    printTop (graph, 10);

  }


  virtual void runPageRank (void) = 0;

public:

  int run (int argc, char* argv[]) {
    LonestarStart (argc, argv, name, desc, url);
    Galois::StatManager sm;

    readGraph ();

    Galois::preAlloc (Galois::getActiveThreads () + (4*sizeof(NodeData)*graph.size () + 2*graph.sizeEdges ())/Galois::Runtime::MM::hugePageSize);
    Galois::reportPageAlloc("MeminfoPre");

    Galois::StatTimer t;

    t.start ();
    runPageRank ();
    t.stop ();

    Galois::reportPageAlloc("MeminfoPost");

    verify ();

    return 0;
  }

};


#endif // PAGERANK_DETERMINISTIC_H
