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
#include "galois/DistGalois.h"
#include "galois/gstl.h"
#include "DistBenchStart.h"
#include "galois/DReducible.h"
#include "galois/runtime/Tracer.h"

// For resilience
#include "resilience.h"

#ifdef __GALOIS_HET_CUDA__
#include "gen_cuda.h"
struct CUDA_Context* cuda_ctx;
#endif

constexpr static const char* const REGION_NAME = "ConnectedComp";

/******************************************************************************/
/* Declaration of command line arguments */
/******************************************************************************/

namespace cll = llvm::cl;
static cll::opt<unsigned int> maxIterations("maxIterations",
                                            cll::desc("Maximum iterations: "
                                                      "Default 1000"),
                                            cll::init(1000));

/******************************************************************************/
/* Graph structure declarations + other initialization */
/******************************************************************************/

struct NodeData {
  std::atomic<uint32_t> comp_current;
  uint32_t comp_old;
};

galois::DynamicBitSet bitset_comp_current;

typedef galois::graphs::DistGraph<NodeData, void> Graph;
typedef typename Graph::GraphNode GNode;

#include "gen_sync.hh"

/******************************************************************************/
/* Algorithm structures */
/******************************************************************************/

const uint32_t infinity = std::numeric_limits<uint32_t>::max();

struct InitializeGraph {
  Graph* graph;

  InitializeGraph(Graph* _graph) : graph(_graph) {}

  void static go(Graph& _graph) {
    const auto& allNodes = _graph.allNodesRange();

#ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      std::string impl_str("InitializeGraph_" + (_graph.get_run_identifier()));
      galois::StatTimer StatTimer_cuda(impl_str.c_str(), REGION_NAME);
      StatTimer_cuda.start();
      InitializeGraph_allNodes_cuda(cuda_ctx);
      StatTimer_cuda.stop();
    } else if (personality == CPU)
#endif
    {
      galois::do_all(galois::iterate(allNodes.begin(), allNodes.end()),
                     InitializeGraph{&_graph}, galois::no_stats(),
                     galois::loopname(
                         _graph.get_run_identifier("InitializeGraph").c_str()));
    }
  }

  void operator()(GNode src) const {
    NodeData& sdata    = graph->getData(src);
    sdata.comp_current = graph->getGID(src);
    sdata.comp_old     = infinity; // graph->getGID(src);
  }
};

struct InitializeGraph_crashed {
  Graph* graph;

  InitializeGraph_crashed(Graph* _graph) : graph(_graph) {}

  void static go(Graph& _graph) {
    const auto& allNodes = _graph.allNodesRange();

#ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      std::string impl_str("InitializeGraph_" + (_graph.get_run_identifier()));
      galois::StatTimer StatTimer_cuda(impl_str.c_str(), REGION_NAME);
      StatTimer_cuda.start();
      InitializeGraph_allNodes_cuda(cuda_ctx);
      StatTimer_cuda.stop();
    } else if (personality == CPU)
#endif
    {
      galois::do_all(
          galois::iterate(allNodes.begin(), allNodes.end()),
          InitializeGraph_crashed{&_graph}, galois::no_stats(),
          galois::loopname(
              _graph.get_run_identifier("InitializeGraph_crashed").c_str()));
    }
  }

  void operator()(GNode src) const {
    NodeData& sdata    = graph->getData(src);
    sdata.comp_current = graph->getGID(src);
    sdata.comp_old     = infinity; // graph->getGID(src);
  }
};

/* Recovery to be called by resilience based fault tolerance
 * Null recovery operator
 */
struct recovery {
  Graph* graph;

  recovery(Graph* _graph) : graph(_graph) {}

  void static go(Graph& _graph) {
    _graph.sync<writeAny, readSource, Reduce_min_comp_current,
                Broadcast_comp_current>("RECOVERY");
  }
};
struct FirstItr_ConnectedComp {
  Graph* graph;
  FirstItr_ConnectedComp(Graph* _graph) : graph(_graph) {}

  void static go(Graph& _graph) {
    const auto& nodesWithEdges = _graph.allNodesWithEdgesRange();
    _graph.set_num_round(0);
#ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      std::string impl_str("ConnectedComp_" + (_graph.get_run_identifier()));
      galois::StatTimer StatTimer_cuda(impl_str.c_str(), REGION_NAME);
      StatTimer_cuda.start();
      FirstItr_ConnectedComp_nodesWithEdges_cuda(cuda_ctx);
      StatTimer_cuda.stop();
    } else if (personality == CPU)
#endif
    {
      galois::do_all(
          galois::iterate(nodesWithEdges), FirstItr_ConnectedComp{&_graph},
          galois::steal(), galois::no_stats(),
          galois::loopname(_graph.get_run_identifier("ConnectedComp").c_str()));
    }

    _graph.sync<writeDestination, readSource, Reduce_min_comp_current,
                Broadcast_comp_current, Bitset_comp_current>("ConnectedComp");

    galois::runtime::reportStat_Tsum(
        REGION_NAME, "NumWorkItems_" + (_graph.get_run_identifier()),
        _graph.allNodesRange().end() - _graph.allNodesRange().begin());
  }

  void operator()(GNode src) const {
    NodeData& snode = graph->getData(src);
    snode.comp_old  = snode.comp_current;

    for (auto jj : graph->edges(src)) {
      GNode dst         = graph->getEdgeDst(jj);
      auto& dnode       = graph->getData(dst);
      uint32_t new_dist = snode.comp_current;
      uint32_t old_dist = galois::atomicMin(dnode.comp_current, new_dist);
      if (old_dist > new_dist)
        bitset_comp_current.set(dst);
    }
  }
};

struct ConnectedComp {
  Graph* graph;
  galois::DGAccumulator<unsigned int>& DGAccumulator_accum;

  ConnectedComp(Graph* _graph, galois::DGAccumulator<unsigned int>& _dga)
      : graph(_graph), DGAccumulator_accum(_dga) {}

  void static go(Graph& _graph, galois::DGAccumulator<unsigned int>& dga) {
    using namespace galois::worklists;

    FirstItr_ConnectedComp::go(_graph);

    unsigned _num_iterations = 1;

    const auto& nodesWithEdges = _graph.allNodesWithEdgesRange();

    do {

      // Checkpointing the all the node data
      if (enableFT && (recoveryScheme == CP || recoveryScheme == HR)) {
        saveCheckpointToDisk(_num_iterations - 1, _graph);
      }

      _graph.set_num_round(_num_iterations);
      dga.reset();
#ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) {
        std::string impl_str("ConnectedComp_" + (_graph.get_run_identifier()));
        galois::StatTimer StatTimer_cuda(impl_str.c_str(), REGION_NAME);
        StatTimer_cuda.start();
        unsigned int __retval = 0;
        ConnectedComp_nodesWithEdges_cuda(__retval, cuda_ctx);
        dga += __retval;
        StatTimer_cuda.stop();
      } else if (personality == CPU)
#endif
      {
        galois::do_all(galois::iterate(nodesWithEdges),
                       ConnectedComp(&_graph, dga), galois::no_stats(),
                       galois::steal(),
                       galois::loopname(
                           _graph.get_run_identifier("ConnectedComp").c_str()));
      }

      _graph.sync<writeDestination, readSource, Reduce_min_comp_current,
                  Broadcast_comp_current, Bitset_comp_current>("ConnectedComp");

      /**************************CRASH SITE : start
       * *****************************************/
      if (enableFT && (_num_iterations == crashIteration)) {
        crashSite<recovery, InitializeGraph_crashed>(_graph);
        dga += 1;
      }
      /**************************CRASH SITE : end
       * *****************************************/

      galois::runtime::reportStat_Tsum(
          REGION_NAME, "NumWorkItems_" + (_graph.get_run_identifier()),
          (unsigned long)dga.read_local());
      ++_num_iterations;
    } while ((_num_iterations < maxIterations) &&
             dga.reduce(_graph.get_run_identifier()));

    if (galois::runtime::getSystemNetworkInterface().ID == 0) {
      galois::runtime::reportStat_Single(
          REGION_NAME, "NumIterations_" + std::to_string(_graph.get_run_num()),
          (unsigned long)_num_iterations);
    }
  }

  void operator()(GNode src) const {
    NodeData& snode = graph->getData(src);

    if (snode.comp_old > snode.comp_current) {
      snode.comp_old = snode.comp_current;

      DGAccumulator_accum += 1;

      for (auto jj : graph->edges(src)) {
        GNode dst         = graph->getEdgeDst(jj);
        auto& dnode       = graph->getData(dst);
        uint32_t new_dist = snode.comp_current;
        uint32_t old_dist = galois::atomicMin(dnode.comp_current, new_dist);
        if (old_dist > new_dist)
          bitset_comp_current.set(dst);
      }
    }
  }
};

/******************************************************************************/
/* Sanity check operators */
/******************************************************************************/

/* Get/print the number of components */
struct ConnectedCompSanityCheck {
  Graph* graph;

  galois::DGAccumulator<uint64_t>& DGAccumulator_accum;

  ConnectedCompSanityCheck(Graph* _graph, galois::DGAccumulator<uint64_t>& _dga)
      : graph(_graph), DGAccumulator_accum(_dga) {}

  void static go(Graph& _graph, galois::DGAccumulator<uint64_t>& dga) {
    dga.reset();

#ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      uint64_t sum;
      ConnectedCompSanityCheck_masterNodes_cuda(sum, cuda_ctx);
      dga += sum;
    } else
#endif
      galois::do_all(galois::iterate(_graph.masterNodesRange().begin(),
                                     _graph.masterNodesRange().end()),
                     ConnectedCompSanityCheck(&_graph, dga), galois::no_stats(),
                     galois::loopname("ConnectedCompSanityCheck"));

    // uint64_t local_num_components = dga.read_local();
    // auto& net = galois::runtime::getSystemNetworkInterface();
    // galois::gPrint("HostID : ", net.ID, " Number of local components is ",
    // local_num_components, "\n");
    uint64_t num_components = dga.reduce();

    // Only node 0 will print the number visited
    if (galois::runtime::getSystemNetworkInterface().ID == 0) {
      galois::gPrint("Number of components is ", num_components, "\n");
    }
  }

  /* Check if a node's component is the same as its ID.
   * if yes, then increment an accumulator */
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    if (src_data.comp_current == graph->getGID(src)) {
      DGAccumulator_accum += 1;
    }
  }
};

/******************************************************************************/
/* Main */
/******************************************************************************/

constexpr static const char* const name = "ConnectedComp - Distributed "
                                          "Heterogeneous with filter.";
constexpr static const char* const desc =
    "ConnectedComp on Distributed Galois.";
constexpr static const char* const url = 0;

int main(int argc, char** argv) {
  galois::DistMemSys G;
  DistBenchStart(argc, argv, name, desc, url);

  auto& net = galois::runtime::getSystemNetworkInterface();

  if (net.ID == 0) {
    galois::runtime::reportParam(REGION_NAME, "Max Iterations",
                                 (unsigned long)maxIterations);
    galois::runtime::reportParam(REGION_NAME, "ENABLE_FT", (enableFT));
  }

  galois::StatTimer StatTimer_total("TimerTotal", REGION_NAME);

  StatTimer_total.start();

#ifdef __GALOIS_HET_CUDA__
  Graph* hg = symmetricDistGraphInitialization<NodeData, void>(&cuda_ctx);
#else
  Graph* hg = symmetricDistGraphInitialization<NodeData, void>();
#endif

  bitset_comp_current.resize(hg->size());

  galois::gPrint("[", net.ID, "] InitializeGraph::go called\n");

  galois::StatTimer StatTimer_init("TIMER_GRAPH_INIT", REGION_NAME);
  StatTimer_init.start();
  InitializeGraph::go((*hg));
  StatTimer_init.stop();
  galois::runtime::getHostBarrier().wait();

  galois::DGAccumulator<unsigned int> DGAccumulator_accum;
  galois::DGAccumulator<uint64_t> DGAccumulator_accum64;

  for (auto run = 0; run < numRuns; ++run) {
    galois::gPrint("[", net.ID, "] ConnectedComp::go run ", run, " called\n");
    std::string timer_str("Timer_" + std::to_string(run));
    galois::StatTimer StatTimer_main(timer_str.c_str(), REGION_NAME);

    StatTimer_main.start();
    ConnectedComp::go(*hg, DGAccumulator_accum);
    StatTimer_main.stop();

    ConnectedCompSanityCheck::go(*hg, DGAccumulator_accum64);

    if ((run + 1) != numRuns) {
#ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) {
        bitset_comp_current_reset_cuda(cuda_ctx);
      } else
#endif
        bitset_comp_current.reset();

      (*hg).set_num_run(run + 1);
      InitializeGraph::go((*hg));
      galois::runtime::getHostBarrier().wait();
    }
  }

  StatTimer_total.stop();

  // Verify
  if (verify) {
#ifdef __GALOIS_HET_CUDA__
    if (personality == CPU) {
#endif
      for (auto ii = (*hg).masterNodesRange().begin();
           ii != (*hg).masterNodesRange().end(); ++ii) {
        galois::runtime::printOutput("% %\n", (*hg).getGID(*ii),
                                     (*hg).getData(*ii).comp_current);
      }
#ifdef __GALOIS_HET_CUDA__
    } else if (personality == GPU_CUDA) {
      for (auto ii = (*hg).masterNodesRange().begin();
           ii != (*hg).masterNodesRange().end(); ++ii) {
        galois::runtime::printOutput("% %\n", (*hg).getGID(*ii),
                                     get_node_comp_current_cuda(cuda_ctx, *ii));
      }
    }
#endif
  }

  return 0;
}
