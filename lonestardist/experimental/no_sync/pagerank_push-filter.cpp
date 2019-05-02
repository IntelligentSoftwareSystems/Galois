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

#include <iostream>
#include <limits>
#include <algorithm>
#include <vector>
#include "galois/DistGalois.h"
#include "DistBenchStart.h"
#include "galois/gstl.h"

#include "galois/runtime/CompilerHelperFunctions.h"
#include "galois/runtime/Tracer.h"

#include "galois/runtime/dGraph_edgeCut.h"
#include "galois/runtime/dGraph_cartesianCut.h"
#include "galois/runtime/dGraph_hybridCut.h"

#include "galois/DReducible.h"

#include "galois/runtime/dGraphLoader.h"

static const char* const name =
    "PageRank - Compiler Generated Distributed Heterogeneous";
static const char* const desc = "Residual PageRank on Distributed Galois.";
static const char* const url  = 0;

/******************************************************************************/
/* Declaration of command line arguments */
/******************************************************************************/
namespace cll = llvm::cl;

static cll::opt<float> tolerance("tolerance",
                                 cll::desc("tolerance for residual"),
                                 cll::init(0.000001));
static cll::opt<unsigned int>
    maxIterations("maxIterations",
                  cll::desc("Maximum iterations: Default 1000"),
                  cll::init(1000));
static cll::opt<bool> verify("verify",
                             cll::desc("Verify ranks by printing to file"),
                             cll::init(false));

/******************************************************************************/
/* Graph structure declarations + other initialization */
/******************************************************************************/

static const float alpha = (1.0 - 0.85);
struct NodeData {
  float value;
  std::atomic<uint32_t> nout;
  float delta;
  std::atomic<float> residual;
};

typedef galois::graphs::DistGraph<NodeData, void> Graph;
typedef typename Graph::GraphNode GNode;
typedef GNode WorkItem;

/******************************************************************************/
/* Algorithm structures */
/******************************************************************************/

// Reset all fields of all nodes to 0
struct ResetGraph {
  Graph* graph;

  ResetGraph(Graph* _graph) : graph(_graph) {}
  void static go(Graph& _graph) {
    auto& allNodes = _graph.allNodesRange();
    galois::do_all(
        allNodes.begin(), allNodes.end(), ResetGraph{&_graph},
        galois::no_stats(),
        galois::loopname(_graph.get_run_identifier("ResetGraph").c_str()));
  }

  void operator()(GNode src) const {
    NodeData& sdata = graph->getData(src);
    sdata.value     = 0;
    sdata.nout      = 0;
    sdata.residual  = 0;
    sdata.delta     = 0;
  }
};

// Initialize residual at nodes with outgoing edges + find nout for
// nodes with outgoing edges
struct InitializeGraph {
  const float& local_alpha;
  Graph* graph;

  InitializeGraph(const float& _alpha, Graph* _graph)
      : local_alpha(_alpha), graph(_graph) {}

  void static go(Graph& _graph) {
    // first initialize all fields to 0 via ResetGraph (can't assume all zero
    // at start)
    ResetGraph::go(_graph);

    auto& nodesWithEdges = _graph.allNodesWithEdgesRange();

    {
      // regular do all without stealing; just initialization of nodes with
      // outgoing edges
      galois::do_all(nodesWithEdges.begin(), nodesWithEdges.end(),
                     InitializeGraph{alpha, &_graph}, galois::no_stats(),
                     galois::loopname(
                         _graph.get_run_identifier("InitializeGraph").c_str()));
    }
  }

  void operator()(GNode src) const {
    NodeData& sdata = graph->getData(src);
    sdata.residual  = local_alpha;
    galois::atomicAdd(
        sdata.nout,
        (uint32_t)std::distance(graph->edge_begin(src), graph->edge_end(src)));
  }
};

struct PageRank {
  const float& local_alpha;
  cll::opt<float>& local_tolerance;
  Graph* graph;
  galois::DGAccumulator<unsigned int>& DGAccumulator_accum;

  PageRank(const float& _local_alpha, cll::opt<float>& _local_tolerance,
           Graph* _g, galois::DGAccumulator<unsigned int>& _dga)
      : local_alpha(_local_alpha), local_tolerance(_local_tolerance), graph(_g),
        DGAccumulator_accum(_dga) {}

  void static go(Graph& _graph, galois::DGAccumulator<unsigned int>& dga) {
    unsigned _num_iterations = 0;
    auto& nodesWithEdges     = _graph.allNodesWithEdgesRange();

    do {
      _graph.set_num_round(_num_iterations);
      PageRank_delta::go(_graph);
      dga.reset();
      {
        galois::do_all(
            nodesWithEdges, PageRank{alpha, tolerance, &_graph, dga},
            galois::no_stats(),
            galois::loopname(_graph.get_run_identifier("PageRank").c_str()),
            galois::steal());
      }

      galois::runtime::reportStat(
          "(NULL)", "NumWorkItems_" + (_graph.get_run_identifier()),
          (unsigned long)dga.read_local(), 0);

      ++_num_iterations;
    } while ((_num_iterations < maxIterations) && dga.reduce());

    if (galois::runtime::getSystemNetworkInterface().ID == 0) {
      galois::runtime::reportStat(
          "(NULL)", "NumIterations_" + std::to_string(_graph.get_run_num()),
          (unsigned long)_num_iterations, 0);
    }
  }

  void operator()(WorkItem src) const {
    NodeData& sdata = graph->getData(src);

    if (sdata.residual > this->local_tolerance) {
      DGAccumulator_accum += 1;

      float residual_old = sdata.residual;
      sdata.residual     = 0;
      sdata.value += residual_old;

      // delta calculation
      if (sdata.nout > 0) {
        sdata.delta = residual_old * (1 - local_alpha) / sdata.nout;
      }
    }

    if (sdata.delta > 0) {
      float _delta = sdata.delta;
      sdata.delta  = 0;

      for (auto nbr = graph->edge_begin(src), ee = graph->edge_end(src);
           nbr != ee; ++nbr) {
        GNode dst       = graph->getEdgeDst(nbr);
        NodeData& ddata = graph->getData(dst);

        galois::atomicAdd(ddata.residual, _delta);
      }
    }
  }
};

/******************************************************************************/
/* Main */
/******************************************************************************/

int main(int argc, char** argv) {
  try {
    galois::DistMemSys G(getStatsFile());
    DistBenchStart(argc, argv, name, desc, url);

    auto& net = galois::runtime::getSystemNetworkInterface();

    {
      if (net.ID == 0) {
        galois::runtime::reportStat("(NULL)", "Max Iterations",
                                    (unsigned long)maxIterations, 0);
        std::ostringstream ss;
        ss << tolerance;
        galois::runtime::reportStat("(NULL)", "Tolerance", ss.str(), 0);
      }
      galois::StatTimer StatTimer_init("TIMER_GRAPH_INIT"),
          StatTimer_total("TimerTotal"), StatTimer_hg_init("TIMER_HG_INIT");

      StatTimer_total.start();

      std::vector<unsigned> scalefactor;
      StatTimer_hg_init.start();
      Graph* hg = nullptr;
      hg        = constructGraph<NodeData, void>(scalefactor);

      residual.allocateInterleaved(hg->size());
      delta.allocateInterleaved(hg->size());

      StatTimer_hg_init.stop();

      std::cout << "[" << net.ID << "] InitializeGraph::go called\n";
      StatTimer_init.start();
      InitializeGraph::go((*hg));
      StatTimer_init.stop();

      galois::DGAccumulator<unsigned int> PageRank_accum;

      for (auto run = 0; run < numRuns; ++run) {
        std::cout << "[" << net.ID << "] PageRank::go run " << run
                  << " called\n";
        std::string timer_str("Timer_" + std::to_string(run));
        galois::StatTimer StatTimer_main(timer_str.c_str());

        StatTimer_main.start();
        PageRank::go(*hg, PageRank_accum);
        StatTimer_main.stop();

        if ((run + 1) != numRuns) {
          // galois::runtime::getHostBarrier().wait();
          (*hg).reset_num_iter(run + 1);
          InitializeGraph::go(*hg);
        }
      }

      StatTimer_total.stop();

      // Verify
      if (verify) {
        for (auto ii = (*hg).begin(); ii != (*hg).end(); ++ii) {
          if ((*hg).isOwned((*hg).getGID(*ii)))
            galois::runtime::printOutput("% %\n", (*hg).getGID(*ii),
                                         (*hg).getData(*ii).value);
        }
      }
    }
    galois::runtime::getHostBarrier().wait();

    return 0;
  } catch (const char* c) {
    std::cerr << "Error: " << c << "\n";
    return 1;
  }
}
