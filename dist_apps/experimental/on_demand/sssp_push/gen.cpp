/**
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of XYZ License (a
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

#ifdef __GALOIS_HET_CUDA__
#include "gen_cuda.h"
struct CUDA_Context* cuda_ctx;
#endif

constexpr static const char* const REGION_NAME = "SSSP";

/******************************************************************************/
/* Declaration of command line arguments */
/******************************************************************************/

namespace cll = llvm::cl;

static cll::opt<unsigned int> maxIterations("maxIterations",
                                            cll::desc("Maximum iterations: "
                                                      "Default 1000"),
                                            cll::init(1000));
static cll::opt<unsigned long long>
    src_node("startNode", // not uint64_t due to a bug in llvm cl
             cll::desc("ID of the source node"), cll::init(0));

/******************************************************************************/
/* Graph structure declarations + other initialization */
/******************************************************************************/

const uint32_t infinity = std::numeric_limits<uint32_t>::max() / 4;

struct NodeData {
  std::atomic<uint32_t> dist_current;
  uint32_t dist_old;
};

#if __OPT_VERSION__ >= 3
galois::DynamicBitSet bitset_dist_current;
#endif

typedef galois::graphs::DistGraph<NodeData, unsigned int> Graph;
typedef typename Graph::GraphNode GNode;

#include "gen_sync.hh"

/******************************************************************************/
/* Algorithm structures */
/******************************************************************************/

struct InitializeGraph {
  const uint32_t& local_infinity;
  cll::opt<unsigned long long>& local_src_node;
  Graph* graph;

  InitializeGraph(cll::opt<unsigned long long>& _src_node,
                  const uint32_t& _infinity, Graph* _graph)
      : local_infinity(_infinity), local_src_node(_src_node), graph(_graph) {}

  void static go(Graph& _graph) {
    const auto& allNodes = _graph.allNodesRange();

#ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      std::string impl_str("InitializeGraph_" + (_graph.get_run_identifier()));
      galois::StatTimer StatTimer_cuda(impl_str.c_str(), REGION_NAME);
      StatTimer_cuda.start();
      InitializeGraph_allNodes_cuda(infinity, src_node, cuda_ctx);
      StatTimer_cuda.stop();
    } else if (personality == CPU)
#endif
    {
      galois::do_all(galois::iterate(allNodes.begin(), allNodes.end()),
                     InitializeGraph{src_node, infinity, &_graph},
                     galois::no_stats(),
                     galois::loopname(
                         _graph.get_run_identifier("InitializeGraph").c_str()));
    }
  }

  void operator()(GNode src) const {
    NodeData& sdata = graph->getData(src);
    sdata.dist_current =
        (graph->getGID(src) == local_src_node) ? 0 : local_infinity;
    sdata.dist_old =
        (graph->getGID(src) == local_src_node) ? 0 : local_infinity;
  }
};

struct FirstItr_SSSP {
  Graph* graph;
  FirstItr_SSSP(Graph* _graph) : graph(_graph) {}

  void static go(Graph& _graph) {
    uint32_t __begin, __end;
    if (_graph.isLocal(src_node)) {
      __begin = _graph.getLID(src_node);
      __end   = __begin + 1;
    } else {
      __begin = 0;
      __end   = 0;
    }

#if __OPT_VERSION__ == 5
    _graph.sync_on_demand<readSource, Reduce_min_dist_current,
                          Broadcast_dist_current, Bitset_dist_current>(
        Flags_dist_current, "SSSP");
#endif

    _graph.set_num_round(0);
#ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      std::string impl_str("SSSP_" + (_graph.get_run_identifier()));
      galois::StatTimer StatTimer_cuda(impl_str.c_str(), REGION_NAME);
      StatTimer_cuda.start();
      FirstItr_SSSP_cuda(__begin, __end, cuda_ctx);
      StatTimer_cuda.stop();
    } else if (personality == CPU)
#endif
    {
      // one node
      galois::do_all(
          galois::iterate(__begin, __end), FirstItr_SSSP{&_graph},
          galois::no_stats(),
          galois::loopname(_graph.get_run_identifier("SSSP").c_str()));
    }

#if __OPT_VERSION__ == 5
    Flags_dist_current.set_write_dst();
#endif

#if __OPT_VERSION__ == 1
    // naive sync of everything after operator
    _graph.sync<writeAny, readAny, Reduce_min_dist_current,
                Broadcast_dist_current>("SSSP");
#elif __OPT_VERSION__ == 2
    // sync of touched fields (same as v1 in this case)
    _graph.sync<writeAny, readAny, Reduce_min_dist_current,
                Broadcast_dist_current>("SSSP");
#elif __OPT_VERSION__ == 3
    // with bitset
    _graph.sync<writeAny, readAny, Reduce_min_dist_current,
                Broadcast_dist_current, Bitset_dist_current>("SSSP");
#elif __OPT_VERSION__ == 4
    // write aware (not read aware, i.e. conservative)
    _graph.sync<writeDestination, readAny, Reduce_min_dist_current,
                Broadcast_dist_current, Bitset_dist_current>("SSSP");
#endif

    galois::runtime::reportStat_Tsum(
        "SSSP", "NumWorkItems_" + (_graph.get_run_identifier()),
        __end - __begin);
  }

  void operator()(GNode src) const {
    NodeData& snode = graph->getData(src);
    snode.dist_old  = snode.dist_current;

    for (auto jj : graph->edges(src)) {
      GNode dst         = graph->getEdgeDst(jj);
      auto& dnode       = graph->getData(dst);
      uint32_t new_dist = graph->getEdgeData(jj) + snode.dist_current;
      uint32_t old_dist = galois::atomicMin(dnode.dist_current, new_dist);
#if __OPT_VERSION__ >= 3
      if (old_dist > new_dist)
        bitset_dist_current.set(dst);
#endif
    }
  }
};

struct SSSP {
  Graph* graph;
  galois::DGAccumulator<unsigned int>& DGAccumulator_accum;

  SSSP(Graph* _graph, galois::DGAccumulator<unsigned int>& _dga)
      : graph(_graph), DGAccumulator_accum(_dga) {}

  void static go(Graph& _graph, galois::DGAccumulator<unsigned int>& dga) {
    using namespace galois::worklists;

    FirstItr_SSSP::go(_graph);

    unsigned _num_iterations = 1;

    const auto& nodesWithEdges = _graph.allNodesWithEdgesRange();

    do {
      _graph.set_num_round(_num_iterations);
      dga.reset();

#if __OPT_VERSION__ == 5
      _graph.sync_on_demand<readSource, Reduce_min_dist_current,
                            Broadcast_dist_current, Bitset_dist_current>(
          Flags_dist_current, "SSSP");
#endif

#ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) {
        std::string impl_str("SSSP_" + (_graph.get_run_identifier()));
        galois::StatTimer StatTimer_cuda(impl_str.c_str(), REGION_NAME);
        StatTimer_cuda.start();
        unsigned int __retval = 0;
        SSSP_nodesWithEdges_cuda(__retval, cuda_ctx);
        dga += __retval;
        StatTimer_cuda.stop();
      } else if (personality == CPU)
#endif
      {
        galois::do_all(
            galois::iterate(nodesWithEdges), SSSP{&_graph, dga},
            galois::no_stats(),
            galois::loopname(_graph.get_run_identifier("SSSP").c_str()),
            galois::steal());
      }

#if __OPT_VERSION__ == 5
      Flags_dist_current.set_write_dst();
#endif

#if __OPT_VERSION__ == 1
      // naive sync of everything after operator
      _graph.sync<writeAny, readAny, Reduce_min_dist_current,
                  Broadcast_dist_current>("SSSP");
#elif __OPT_VERSION__ == 2
      // sync of touched fields (same as v1 in this case)
      _graph.sync<writeAny, readAny, Reduce_min_dist_current,
                  Broadcast_dist_current>("SSSP");
#elif __OPT_VERSION__ == 3
      // with bitset
      _graph.sync<writeAny, readAny, Reduce_min_dist_current,
                  Broadcast_dist_current, Bitset_dist_current>("SSSP");
#elif __OPT_VERSION__ == 4
      // write aware (not read aware, i.e. conservative)
      _graph.sync<writeDestination, readAny, Reduce_min_dist_current,
                  Broadcast_dist_current, Bitset_dist_current>("SSSP");
#endif

      galois::runtime::reportStat_Tsum(
          "SSSP", "NumWorkItems_" + (_graph.get_run_identifier()),
          (unsigned long)dga.read_local());
      ++_num_iterations;
    } while ((_num_iterations < maxIterations) &&
             dga.reduce(_graph.get_run_identifier()));

    if (galois::runtime::getSystemNetworkInterface().ID == 0) {
      galois::runtime::reportStat_Single(
          "SSSP", "NumIterations_" + std::to_string(_graph.get_run_num()),
          (unsigned long)_num_iterations);
    }
  }

  void operator()(GNode src) const {
    NodeData& snode = graph->getData(src);

    if (snode.dist_old > snode.dist_current) {
      snode.dist_old = snode.dist_current;

      DGAccumulator_accum += 1;

      for (auto jj : graph->edges(src)) {
        GNode dst         = graph->getEdgeDst(jj);
        auto& dnode       = graph->getData(dst);
        uint32_t new_dist = graph->getEdgeData(jj) + snode.dist_current;
        uint32_t old_dist = galois::atomicMin(dnode.dist_current, new_dist);
#if __OPT_VERSION__ >= 3
        if (old_dist > new_dist)
          bitset_dist_current.set(dst);
#endif
      }
    }
  }
};

/******************************************************************************/
/* Sanity check operators */
/******************************************************************************/

/* Prints total number of nodes visited + max distance */
struct SSSPSanityCheck {
  const uint32_t& local_infinity;
  Graph* graph;

  galois::DGAccumulator<uint64_t>& DGAccumulator_sum;
  galois::DGReduceMax<uint32_t>& DGMax;

  SSSPSanityCheck(const uint32_t& _infinity, Graph* _graph,
                  galois::DGAccumulator<uint64_t>& dgas,
                  galois::DGReduceMax<uint32_t>& dgm)
      : local_infinity(_infinity), graph(_graph), DGAccumulator_sum(dgas),
        DGMax(dgm) {}

  void static go(Graph& _graph, galois::DGAccumulator<uint64_t>& dgas,
                 galois::DGReduceMax<uint32_t>& dgm) {
    dgas.reset();
    dgm.reset();

#ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      uint64_t sum;
      uint32_t max;
      SSSPSanityCheck_masterNodes_cuda(sum, max, infinity, cuda_ctx);
      dgas += sum;
      dgm.update(max);
    } else
#endif
    {
      galois::do_all(galois::iterate(_graph.masterNodesRange().begin(),
                                     _graph.masterNodesRange().end()),
                     SSSPSanityCheck(infinity, &_graph, dgas, dgm),
                     galois::no_stats(), galois::loopname("SSSPSanityCheck"));
    }

    uint64_t num_visited  = dgas.reduce();
    uint32_t max_distance = dgm.reduce();

    // Only host 0 will print the info
    if (galois::runtime::getSystemNetworkInterface().ID == 0) {
      galois::gPrint("Number of nodes visited from source ", src_node, " is ",
                     num_visited, "\n");
      galois::gPrint("Max distance from source ", src_node, " is ",
                     max_distance, "\n");
    }
  }

  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    if (src_data.dist_current < local_infinity) {
      DGAccumulator_sum += 1;
      DGMax.update(src_data.dist_current);
    }
  }
};

/******************************************************************************/
/* Main */
/******************************************************************************/

constexpr static const char* const name = "SSSP - Distributed Heterogeneous "
                                          "with worklist.";
constexpr static const char* const desc = "Variant of Chaotic relaxation SSSP "
                                          "on Distributed Galois.";
constexpr static const char* const url = 0;

int main(int argc, char** argv) {
  galois::DistMemSys G;
  DistBenchStart(argc, argv, name, desc, url);

  auto& net = galois::runtime::getSystemNetworkInterface();

  if (net.ID == 0) {
    galois::runtime::reportParam("SSSP", "Max Iterations",
                                 (unsigned long)maxIterations);
    galois::runtime::reportParam("SSSP", "Source Node ID",
                                 (unsigned long)src_node);

#if __OPT_VERSION__ == 1
    printf("Version 1 of optimization\n");
#elif __OPT_VERSION__ == 2
    printf("Version 2 of optimization\n");
#elif __OPT_VERSION__ == 3
    printf("Version 3 of optimization\n");
#elif __OPT_VERSION__ == 4
    printf("Version 4 of optimization\n");
#elif __OPT_VERSION__ == 5
    printf("Version 5 of optimization\n");
#endif
  }

  galois::StatTimer StatTimer_total("TimerTotal", REGION_NAME);

  StatTimer_total.start();

#ifdef __GALOIS_HET_CUDA__
  Graph* hg = distGraphInitialization<NodeData, unsigned int>(&cuda_ctx);
#else
  Graph* hg = distGraphInitialization<NodeData, unsigned int>();
#endif

#if __OPT_VERSION__ >= 3
  bitset_dist_current.resize(hg->size());
#endif

  galois::gPrint("[", net.ID, "] InitializeGraph::go called\n");

  galois::StatTimer StatTimer_init("TIMER_GRAPH_INIT", REGION_NAME);

  StatTimer_init.start();
  InitializeGraph::go((*hg));
  StatTimer_init.stop();
  galois::runtime::getHostBarrier().wait();

  // accumulators for use in operators
  galois::DGAccumulator<unsigned int> DGAccumulator_accum;
  galois::DGAccumulator<uint64_t> DGAccumulator_sum;
  galois::DGReduceMax<uint32_t> m;

  for (auto run = 0; run < numRuns; ++run) {
    galois::gPrint("[", net.ID, "] SSSP::go run ", run, " called\n");
    std::string timer_str("Timer_" + std::to_string(run));
    galois::StatTimer StatTimer_main(timer_str.c_str(), REGION_NAME);

    StatTimer_main.start();
    SSSP::go(*hg, DGAccumulator_accum);
    StatTimer_main.stop();

    SSSPSanityCheck::go(*hg, DGAccumulator_sum, m);

    if ((run + 1) != numRuns) {
#ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) {
#if __OPT_VERSION__ >= 3
        bitset_dist_current_reset_cuda(cuda_ctx);
#endif
      } else
#endif
#if __OPT_VERSION__ >= 3
        bitset_dist_current.reset();
#endif

#if __OPT_VERSION__ == 5
      Flags_dist_current.clear_all();
#endif

      (*hg).set_num_run(run + 1);
      InitializeGraph::go(*hg);
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
                                     (*hg).getData(*ii).dist_current);
      }
#ifdef __GALOIS_HET_CUDA__
    } else if (personality == GPU_CUDA) {
      for (auto ii = (*hg).masterNodesRange().begin();
           ii != (*hg).masterNodesRange().end(); ++ii) {
        galois::runtime::printOutput("% %\n", (*hg).getGID(*ii),
                                     get_node_dist_current_cuda(cuda_ctx, *ii));
      }
    }
#endif
  }

  return 0;
}
