#ifndef PAGERANK_DETERMINISTIC_H
#define PAGERANK_DETERMINISTIC_H

#include "Galois/Accumulator.h"
#include "Galois/Timer.h"
#include "Galois/Statistic.h"
#include "Galois/Galois.h"
#include "Galois/DoAllWrap.h"
#include "Galois/PerThreadContainer.h"

#include "Galois/Graphs/Util.h"
#include "Galois/Graphs/Graph.h"
// #include "Galois/Graph/FileGraph.h"

#include "Galois/Runtime/Sampling.h"
#include "Galois/Runtime/DetChromatic.h"
#include "Galois/Runtime/DetPartInputDAG.h"
#include "Galois/Runtime/DetKDGexecutor.h"

#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

#include "PageRankOld.h"

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
  Galois::GAccumulator<size_t> numIter;

  void readGraph (void) {
    Galois::Graph::readGraph (graph, inputFile, transposeFile);

    Galois::do_all_choice (Galois::Runtime::makeLocalRange (graph),
        [&] (GNode n) {
          auto& pdata = graph.getData (n, Galois::MethodFlag::UNPROTECTED);
          pdata.outdegree = std::distance (
            graph.edge_begin (n, Galois::MethodFlag::UNPROTECTED),
            graph.edge_end (n, Galois::MethodFlag::UNPROTECTED));
        },
        "init-pdata", Galois::chunk_size<DEFAULT_CHUNK_SIZE> ());

    std::printf ("Graph read with %zd nodes and %zd edges\n", 
        graph.size (), graph.sizeEdges ());
  }

  void visitNhood (GNode src) { 
    graph.getData (src, Galois::MethodFlag::WRITE);

    for (auto i = graph.in_edge_begin (src, Galois::MethodFlag::READ)
        , end_i = graph.in_edge_end (src, Galois::MethodFlag::READ); i != end_i; ++i) {
    }

    // for (auto i = graph.edge_begin (src, Galois::MethodFlag::WRITE)
    // , end_i = graph.edge_end (src, Galois::MethodFlag::WRITE); i != end_i; ++i) {
    // }
  }

  template <typename C, bool use_onWL=false, bool doLock=false>
  void applyOperator (GNode src, C& ctx) {

    if (doLock) {
      graph.getData (src, Galois::MethodFlag::WRITE);
    }

    if (use_onWL) {
      auto& sdata = graph.getData (src, Galois::MethodFlag::UNPROTECTED);
      sdata.onWL = 0;
    }

    numIter += 1;
    double sum = 0;

    if (doLock) {
      for (auto jj = graph.in_edge_begin(src, Galois::MethodFlag::READ), ej = graph.in_edge_end(src, Galois::MethodFlag::READ); jj != ej; ++jj) {
        GNode dst = graph.getInEdgeDst(jj);
        auto& ddata = graph.getData(dst, Galois::MethodFlag::UNPROTECTED);
        sum += ddata.value / ddata.outdegree;
      }

    } else {

      for (auto jj = graph.in_edge_begin(src, Galois::MethodFlag::UNPROTECTED), ej = graph.in_edge_end(src, Galois::MethodFlag::UNPROTECTED); jj != ej; ++jj) {
        GNode dst = graph.getInEdgeDst(jj);
        auto& ddata = graph.getData(dst, Galois::MethodFlag::UNPROTECTED);
        sum += ddata.value / ddata.outdegree;
      }
    }

    float value = (1.0 - alpha) * sum + alpha;
    auto& sdata = graph.getData(src, Galois::MethodFlag::UNPROTECTED);
    float diff = std::fabs(value - sdata.value);

    
    
    if (diff >= tolerance) {
      sdata.value = value;

      for (auto jj = graph.edge_begin(src, Galois::MethodFlag::UNPROTECTED), ej = graph.edge_end(src, Galois::MethodFlag::UNPROTECTED); jj != ej; ++jj) {
        GNode dst = graph.getEdgeDst(jj);

        if (use_onWL) {

          auto& ddata = graph.getData (dst, Galois::MethodFlag::UNPROTECTED);
          unsigned old = 0;
          if (ddata.onWL.cas (old, 1)) {
            ctx.push (dst);
          }

        } else {
          ctx.push(dst);
        } 
      }
    } 
  }

  bool checkConvergence (void) {
    Galois::GReduceLogicalAND allConverged;

    Galois::do_all_choice (Galois::Runtime::makeLocalRange (graph),
        [&] (GNode src) {
          double sum = 0;

          for (auto jj = graph.in_edge_begin(src, Galois::MethodFlag::UNPROTECTED), ej = graph.in_edge_end(src, Galois::MethodFlag::UNPROTECTED); jj != ej; ++jj) {
            GNode dst = graph.getInEdgeDst(jj);
            auto& ddata = graph.getData(dst, Galois::MethodFlag::UNPROTECTED);
            sum += ddata.value / ddata.outdegree;
          }

          float value = (1.0 - alpha) * sum + alpha;
          auto& sdata = graph.getData(src, Galois::MethodFlag::UNPROTECTED);
          float diff = std::fabs(value - sdata.value);

          
          
          if (diff >= tolerance) {
            allConverged.update (false);
            // std::fprintf (stderr, "ERROR: convergence failed on node %d, error=%f, tolerance=%f\n", src, diff, tolerance);
          }
        }, 
        "check-convergence", Galois::chunk_size<DEFAULT_CHUNK_SIZE> ());

    return allConverged.reduceRO ();
  }


  void verify (void) {
    if (skipVerify) {
      std::printf ("WARNING, Verification skipped\n");
      return;
    }

    if (!checkConvergence ()) {
      std::fprintf (stderr, "ERROR: Convergence check FAILED\n");
    } else {
      std::printf ("OK: Convergence check PASSED\n");
    }

    printTop (graph, 10);

  }


  virtual void runPageRank (void) = 0;

public:

  int run (int argc, char* argv[]) {
    LonestarStart (argc, argv, name, desc, url);
    return run ();
  }

  int run (void) {

    Galois::StatManager sm;

    readGraph ();

    Galois::preAlloc (Galois::getActiveThreads () + (4*sizeof(NodeData)*graph.size () + 2*graph.sizeEdges ())/Galois::Runtime::pagePoolSize());
    Galois::reportPageAlloc("MeminfoPre");

    Galois::StatTimer t;

    t.start ();
    runPageRank ();
    t.stop ();

    std::printf ("Total number of updates/iterations performed by PageRank: %zd\n", numIter.reduceRO ());

    Galois::reportPageAlloc("MeminfoPost");

    verify ();

    return 0;
  }

};


#endif // PAGERANK_DETERMINISTIC_H
