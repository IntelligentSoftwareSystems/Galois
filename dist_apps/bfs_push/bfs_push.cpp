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

// NOTE: must be before DistBenchStart.h as that relies on some cuda
// calls

#include <iostream>
#include <limits>
#include "galois/DistGalois.h"
#include "galois/gstl.h"
#include "DistBenchStart.h"
#include "galois/DReducible.h"
#ifdef __GALOIS_HET_ASYNC__
#include "galois/DTerminationDetector.h"
#endif
#include "galois/runtime/Tracer.h"

#ifdef __GALOIS_HET_CUDA__
#include "bfs_push_cuda.h"
struct CUDA_Context* cuda_ctx;
#endif

constexpr static const char* const regionname = "BFS";

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

galois::DynamicBitSet bitset_dist_current;

typedef galois::graphs::DistGraph<NodeData, void> Graph;
typedef typename Graph::GraphNode GNode;

#include "bfs_push_sync.hh"

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
      std::string impl_str(_graph.get_run_identifier("InitializeGraph_"));
      galois::StatTimer StatTimer_cuda(impl_str.c_str(), regionname);
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

struct FirstItr_BFS {
  Graph* graph;

  FirstItr_BFS(Graph* _graph) : graph(_graph) {}

  void static go(Graph& _graph) {
    uint32_t __begin, __end;
    if (_graph.isLocal(src_node)) {
      __begin = _graph.getLID(src_node);
      __end   = __begin + 1;
    } else {
      __begin = 0;
      __end   = 0;
    }
    _graph.set_num_round(0);
#ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      std::string impl_str(_graph.get_run_identifier("BFS"));
      galois::StatTimer StatTimer_cuda(impl_str.c_str(), regionname);
      StatTimer_cuda.start();
      FirstItr_BFS_cuda(__begin, __end, cuda_ctx);
      StatTimer_cuda.stop();
    } else if (personality == CPU)
#endif
    {
      // one node
      galois::do_all(
          galois::iterate(__begin, __end), FirstItr_BFS{&_graph},
          galois::no_stats(),
          galois::loopname(_graph.get_run_identifier("BFS").c_str()));
    }

    _graph.sync<writeDestination, readSource, Reduce_min_dist_current,
                Broadcast_dist_current, Bitset_dist_current>("BFS");

    galois::runtime::reportStat_Tsum(
        regionname, _graph.get_run_identifier("NumWorkItems"), __end - __begin);
  }

  void operator()(GNode src) const {
    NodeData& snode = graph->getData(src);
    snode.dist_old  = snode.dist_current;

    for (auto jj : graph->edges(src)) {
      GNode dst         = graph->getEdgeDst(jj);
      auto& dnode       = graph->getData(dst);
      uint32_t new_dist = 1 + snode.dist_current;
      uint32_t old_dist = galois::atomicMin(dnode.dist_current, new_dist);
      if (old_dist > new_dist)
        bitset_dist_current.set(dst);
    }
  }
};

struct BFS {
  Graph* graph;
#ifdef __GALOIS_HET_ASYNC__
  using DGAccumulatorTy = galois::DGTerminator<unsigned int>;
#else
  using DGAccumulatorTy = galois::DGAccumulator<unsigned int>;
#endif

  DGAccumulatorTy& DGAccumulator_accum;

  BFS(Graph* _graph, DGAccumulatorTy& _dga)
      : graph(_graph), DGAccumulator_accum(_dga) {}

  void static go(Graph& _graph, DGAccumulatorTy& dga) {
    FirstItr_BFS::go(_graph);

    unsigned _num_iterations = 1;

    const auto& nodesWithEdges = _graph.allNodesWithEdgesRange();

    do {
      _graph.set_num_round(_num_iterations);
      dga.reset();
#ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) {
        std::string impl_str(_graph.get_run_identifier("BFS"));
        galois::StatTimer StatTimer_cuda(impl_str.c_str(), regionname);
        StatTimer_cuda.start();
        unsigned int __retval = 0;
        BFS_nodesWithEdges_cuda(__retval, cuda_ctx);
        dga += __retval;
        StatTimer_cuda.stop();
      } else if (personality == CPU)
#endif
      {
        galois::do_all(
            galois::iterate(nodesWithEdges), BFS(&_graph, dga), galois::steal(),
            galois::no_stats(),
            galois::loopname(_graph.get_run_identifier("BFS").c_str()));
      }
#ifdef __GALOIS_HET_ASYNC__
      _graph.sync<writeDestination, readSource, Reduce_min_dist_current,
                  Broadcast_dist_current, Bitset_dist_current, true>("BFS");
#else
      _graph.sync<writeDestination, readSource, Reduce_min_dist_current,
                  Broadcast_dist_current, Bitset_dist_current>("BFS");
#endif

      galois::runtime::reportStat_Tsum(
          regionname, _graph.get_run_identifier("NumWorkItems"),
          (unsigned long)dga.read_local());

      ++_num_iterations;
    } while (
#ifndef __GALOIS_HET_ASYNC__
             (_num_iterations < maxIterations) &&
#endif
             dga.reduce(_graph.get_run_identifier()));

    if (galois::runtime::getSystemNetworkInterface().ID == 0) {
      galois::runtime::reportStat_Single(
          regionname, "NumIterations_" + std::to_string(_graph.get_run_num()),
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
        uint32_t new_dist = 1 + snode.dist_current;
        uint32_t old_dist = galois::atomicMin(dnode.dist_current, new_dist);
        if (old_dist > new_dist)
          bitset_dist_current.set(dst);
      }
    }
  }
};

/******************************************************************************/
/* Sanity check operators */
/******************************************************************************/

/* Prints total number of nodes visited + max distance */
struct BFSSanityCheck {
  const uint32_t& local_infinity;
  Graph* graph;

  galois::DGAccumulator<uint64_t>& DGAccumulator_sum;
  galois::DGReduceMax<uint32_t>& DGMax;

  BFSSanityCheck(const uint32_t& _infinity, Graph* _graph,
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
      BFSSanityCheck_masterNodes_cuda(sum, max, infinity, cuda_ctx);
      dgas += sum;
      dgm.update(max);
    } else
#endif
    {
      galois::do_all(galois::iterate(_graph.masterNodesRange().begin(),
                                     _graph.masterNodesRange().end()),
                     BFSSanityCheck(infinity, &_graph, dgas, dgm),
                     galois::no_stats(), galois::loopname("BFSSanityCheck"));
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

constexpr static const char* const name =
    "BFS - Distributed Heterogeneous with "
    "worklist.";
constexpr static const char* const desc = "BFS on Distributed Galois.";
constexpr static const char* const url  = 0;

int main(int argc, char** argv) {
  galois::DistMemSys G;
  DistBenchStart(argc, argv, name, desc, url);

  const auto& net = galois::runtime::getSystemNetworkInterface();
  if (net.ID == 0) {
    galois::runtime::reportParam(regionname, "Max Iterations",
                                 (unsigned long)maxIterations);
    galois::runtime::reportParam(regionname, "Source Node ID",
                                 (unsigned long long)src_node);
  }

  galois::StatTimer StatTimer_total("TimerTotal", regionname);

  StatTimer_total.start();
#ifdef __GALOIS_HET_CUDA__
  Graph* hg = distGraphInitialization<NodeData, void>(&cuda_ctx);
#else
  Graph* hg = distGraphInitialization<NodeData, void>();
#endif

  // bitset comm setup
  bitset_dist_current.resize(hg->size());

  galois::gPrint("[", net.ID, "] InitializeGraph::go called\n");

  InitializeGraph::go((*hg));
  galois::runtime::getHostBarrier().wait();

  // accumulators for use in operators
#ifdef __GALOIS_HET_ASYNC__
  galois::DGTerminator<unsigned int> DGAccumulator_accum;
#else
  galois::DGAccumulator<unsigned int> DGAccumulator_accum;
#endif
  galois::DGAccumulator<uint64_t> DGAccumulator_sum;
  galois::DGReduceMax<uint32_t> m;

  for (auto run = 0; run < numRuns; ++run) {
    galois::gPrint("[", net.ID, "] BFS::go run ", run, " called\n");
    std::string timer_str("Timer_" + std::to_string(run));
    galois::StatTimer StatTimer_main(timer_str.c_str(), regionname);

    StatTimer_main.start();
    BFS::go(*hg, DGAccumulator_accum);
    StatTimer_main.stop();

    // sanity check
    BFSSanityCheck::go(*hg, DGAccumulator_sum, m);

    if ((run + 1) != numRuns) {
#ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) {
        bitset_dist_current_reset_cuda(cuda_ctx);
      } else
#endif
        bitset_dist_current.reset();

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
