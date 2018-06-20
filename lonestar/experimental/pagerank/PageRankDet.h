/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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
 */

#ifndef PAGERANK_DETERMINISTIC_H
#define PAGERANK_DETERMINISTIC_H

#include "galois/Reduction.h"
#include "galois/Timer.h"
#include "galois/Timer.h"
#include "galois/Galois.h"
#include "galois/DoAllWrap.h"
#include "galois/PerThreadContainer.h"

#include "galois/graphs/Util.h"
#include "galois/graphs/Graph.h"
// #include "galois/Graph/FileGraph.h"

#include "galois/runtime/Profile.h"
#include "galois/runtime/DetChromatic.h"
#include "galois/runtime/DetPartInputDAG.h"
#include "galois/runtime/DetKDGexecutor.h"

#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

#include "PageRankOld.h"

#include <cstdio>

namespace cll = llvm::cl;

static cll::opt<std::string>
    inputFile(cll::Positional, cll::desc("<input file>"), cll::Required);

static cll::opt<std::string>
    transposeFile("transpose", cll::desc("<transpose file>"), cll::Required);

static const char* const name = "Page Rank";
static const char* const desc = "Page Rank of a graph of web pages";
static const char* const url  = "pagerank";

static const double PAGE_RANK_INIT = 1.0;

struct PData {
  float value;
  unsigned outdegree;

  PData(void) : value(PAGE_RANK_INIT), outdegree(0) {}

  PData(unsigned outdegree) : value(PAGE_RANK_INIT), outdegree(outdegree) {}

  double getPageRank() const { return value; }
};

template <typename IG>
class PageRankBase {

protected:
  typedef typename galois::graphs::LC_InOut_Graph<IG> Graph;
  typedef typename Graph::GraphNode GNode;
  typedef typename Graph::node_data_type NodeData;

  static const unsigned DEFAULT_CHUNK_SIZE = 1;

  Graph graph;
  galois::GAccumulator<size_t> numIter;

  void readGraph(void) {
    galois::graphs::readGraph(graph, inputFile, transposeFile);

    galois::do_all_choice(
        galois::runtime::makeLocalRange(graph),
        [&](GNode n) {
          auto& pdata     = graph.getData(n, galois::MethodFlag::UNPROTECTED);
          pdata.outdegree = std::distance(
              graph.edge_begin(n, galois::MethodFlag::UNPROTECTED),
              graph.edge_end(n, galois::MethodFlag::UNPROTECTED));
        },
        "init-pdata", galois::chunk_size<DEFAULT_CHUNK_SIZE>());

    std::printf("Graph read with %zd nodes and %zd edges\n", graph.size(),
                graph.sizeEdges());
  }

  void visitNhood(GNode src) {
    graph.getData(src, galois::MethodFlag::WRITE);

    for (auto i     = graph.in_edge_begin(src, galois::MethodFlag::READ),
              end_i = graph.in_edge_end(src, galois::MethodFlag::READ);
         i != end_i; ++i) {
    }

    // for (auto i = graph.edge_begin (src, galois::MethodFlag::WRITE)
    // , end_i = graph.edge_end (src, galois::MethodFlag::WRITE); i != end_i;
    // ++i) {
    // }
  }

  template <typename C, bool use_onWL = false, bool doLock = false>
  void applyOperator(GNode src, C& ctx) {

    if (doLock) {
      graph.getData(src, galois::MethodFlag::WRITE);
    }

    if (use_onWL) {
      auto& sdata = graph.getData(src, galois::MethodFlag::UNPROTECTED);
      sdata.onWL  = 0;
    }

    numIter += 1;
    double sum = 0;

    if (doLock) {
      for (auto jj = graph.in_edge_begin(src, galois::MethodFlag::READ),
                ej = graph.in_edge_end(src, galois::MethodFlag::READ);
           jj != ej; ++jj) {
        GNode dst   = graph.getInEdgeDst(jj);
        auto& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
        sum += ddata.value / ddata.outdegree;
      }

    } else {

      for (auto jj = graph.in_edge_begin(src, galois::MethodFlag::UNPROTECTED),
                ej = graph.in_edge_end(src, galois::MethodFlag::UNPROTECTED);
           jj != ej; ++jj) {
        GNode dst   = graph.getInEdgeDst(jj);
        auto& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
        sum += ddata.value / ddata.outdegree;
      }
    }

    float value = (1.0 - alpha) * sum + alpha;
    auto& sdata = graph.getData(src, galois::MethodFlag::UNPROTECTED);
    float diff  = std::fabs(value - sdata.value);

    if (diff >= tolerance) {
      sdata.value = value;

      for (auto jj = graph.edge_begin(src, galois::MethodFlag::UNPROTECTED),
                ej = graph.edge_end(src, galois::MethodFlag::UNPROTECTED);
           jj != ej; ++jj) {
        GNode dst = graph.getEdgeDst(jj);

        if (use_onWL) {

          auto& ddata  = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
          unsigned old = 0;
          if (ddata.onWL.cas(old, 1)) {
            ctx.push(dst);
          }

        } else {
          ctx.push(dst);
        }
      }
    }
  }

  bool checkConvergence(void) {
    galois::GReduceLogicalAND allConverged;

    galois::do_all_choice(
        galois::runtime::makeLocalRange(graph),
        [&](GNode src) {
          double sum = 0;

          for (auto
                   jj = graph.in_edge_begin(src,
                                            galois::MethodFlag::UNPROTECTED),
                   ej = graph.in_edge_end(src, galois::MethodFlag::UNPROTECTED);
               jj != ej; ++jj) {
            GNode dst   = graph.getInEdgeDst(jj);
            auto& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
            sum += ddata.value / ddata.outdegree;
          }

          float value = (1.0 - alpha) * sum + alpha;
          auto& sdata = graph.getData(src, galois::MethodFlag::UNPROTECTED);
          float diff  = std::fabs(value - sdata.value);

          if (diff >= tolerance) {
            allConverged.update(false);
            // std::fprintf (stderr, "ERROR: convergence failed on node %d,
            // error=%f, tolerance=%f\n", src, diff, tolerance);
          }
        },
        "check-convergence", galois::chunk_size<DEFAULT_CHUNK_SIZE>());

    return allConverged.reduceRO();
  }

  void verify(void) {
    if (skipVerify) {
      std::printf("WARNING, Verification skipped\n");
      return;
    }

    if (!checkConvergence()) {
      std::fprintf(stderr, "ERROR: Convergence check FAILED\n");
    } else {
      std::printf("OK: Convergence check PASSED\n");
    }

    printTop(graph, 10);
  }

  virtual void runPageRank(void) = 0;

public:
  int run(int argc, char* argv[]) {
    LonestarStart(argc, argv, name, desc, url);
    return run();
  }

  int run(void) {

    galois::StatManager sm;

    readGraph();

    galois::preAlloc(
        galois::getActiveThreads() +
        (4 * sizeof(NodeData) * graph.size() + 2 * graph.sizeEdges()) /
            galois::runtime::pagePoolSize());
    galois::reportPageAlloc("MeminfoPre");

    galois::StatTimer t;

    t.start();
    runPageRank();
    t.stop();

    std::printf(
        "Total number of updates/iterations performed by PageRank: %zd\n",
        numIter.reduceRO());

    galois::reportPageAlloc("MeminfoPost");

    verify();

    return 0;
  }
};

#endif // PAGERANK_DETERMINISTIC_H
