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
#ifdef __GALOIS_HET_ASYNC__
#include "galois/DTerminationDetector.h"
#endif
#include "galois/runtime/Tracer.h"

#ifdef __GALOIS_HET_CUDA__
#include "sssp_push_cuda.h"
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

static cll::opt<uint32_t>
    delta("delta",
             cll::desc("Shift value for the delta step (default value 0)"),
             cll::init(0));

/******************************************************************************/
/* Graph structure declarations + other initialization */
/******************************************************************************/

const uint32_t infinity = std::numeric_limits<uint32_t>::max() / 4;

struct NodeData {
  std::atomic<uint32_t> dist_current;
  uint32_t dist_old;
};

galois::DynamicBitSet bitset_dist_current;
uint32_t numThreadBlocks;

typedef galois::graphs::DistGraph<NodeData, unsigned int> Graph;
typedef typename Graph::GraphNode GNode;

#include "sssp_push_sync.hh"

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
    sdata.dist_current = (graph->getGID(src) == local_src_node) ? 0 : local_infinity;
    sdata.dist_old = (graph->getGID(src) == local_src_node) ? 0 : local_infinity;
  }
};

#ifdef __GALOIS_HET_CUDA__
#if DIST_PER_ROUND_TIMER
void ReportThreadBlockWork(uint32_t iteration_num, std::string run_identifier, std::string tb_identifer){

	std::string str = get_thread_block_work_into_string(cuda_ctx);
	galois::runtime::reportParam(REGION_NAME, run_identifier, str);

	if (galois::runtime::getSystemNetworkInterface().ID == 0 && iteration_num == 0) {
		//Assumption: The number of thread blocks in all the iterations
		std::string num_thread_blocks = get_num_thread_blocks(cuda_ctx);
		galois::runtime::reportParam(REGION_NAME, tb_identifer, num_thread_blocks);
	}
}
#endif
#endif

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
    _graph.set_num_round(0);
#ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      std::string impl_str("SSSP_" + (_graph.get_run_identifier()));
      galois::StatTimer StatTimer_cuda(impl_str.c_str(), REGION_NAME);
      StatTimer_cuda.start();
#if DIST_PER_ROUND_TIMER
      unsigned int active_vertices = 0;	
      FirstItr_SSSP_cuda(__begin, __end, active_vertices, cuda_ctx);
#else
      FirstItr_SSSP_cuda(__begin, __end, cuda_ctx);
#endif
      StatTimer_cuda.stop();
#if DIST_PER_ROUND_TIMER
      std::string identifer(_graph.get_run_identifier("GPUThreadBlocksWork_Host", galois::runtime::getSystemNetworkInterface().ID));
      std::string tb_identifer(_graph.get_run_identifier("ThreadBlocks_Host", galois::runtime::getSystemNetworkInterface().ID));
      ReportThreadBlockWork(0, identifer, tb_identifer);
      std::string acive_identifer(_graph.get_run_identifier("NumActiveVertices"));
      galois::runtime::reportParam(REGION_NAME, acive_identifer, std::to_string(active_vertices));
#endif

    } else if (personality == CPU)
#endif
    {
      // one node
      galois::do_all(
          galois::iterate(__begin, __end), FirstItr_SSSP{&_graph},
          galois::no_stats(),
          galois::loopname(_graph.get_run_identifier("SSSP").c_str()));
    }

    _graph.sync<writeDestination, readSource, Reduce_min_dist_current,
                Bitset_dist_current>("SSSP");

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
      if (old_dist > new_dist)
        bitset_dist_current.set(dst);
    }
  }
};

struct SSSP {
  uint32_t local_priority;
  Graph* graph;
#ifdef __GALOIS_HET_ASYNC__
  using DGAccumulatorTy = galois::DGTerminator<unsigned int>;
#else
  using DGAccumulatorTy = galois::DGAccumulator<unsigned int>;
#endif

  DGAccumulatorTy& active_vertices;
  DGAccumulatorTy& work_edges;

  SSSP(uint32_t _local_priority, Graph* _graph, 
      DGAccumulatorTy& _dga, DGAccumulatorTy& _work_edges)
      : local_priority(_local_priority), graph(_graph), 
      active_vertices(_dga), work_edges(_work_edges) {}

  void static go(Graph& _graph, DGAccumulatorTy& dga) {
    using namespace galois::worklists;

    FirstItr_SSSP::go(_graph);

    unsigned _num_iterations = 1;

    const auto& nodesWithEdges = _graph.allNodesWithEdgesRange();

    uint32_t priority;
    if (delta == 0) priority = std::numeric_limits<uint32_t>::max();
    else priority = 0;
    DGAccumulatorTy work_edges;

    do {

      //if (work_edges.reduce() == 0) 
      priority += delta;

      _graph.set_num_round(_num_iterations);
      dga.reset();
      work_edges.reset();
#ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) {
        std::string impl_str("SSSP_" + (_graph.get_run_identifier()));
        galois::StatTimer StatTimer_cuda(impl_str.c_str(), REGION_NAME);
        StatTimer_cuda.start();
        unsigned int __retval = 0;
        unsigned int __retval2 = 0;
#if DIST_PER_ROUND_TIMER
        unsigned int active_vertices = 0;
        SSSP_nodesWithEdges_cuda(__retval, __retval2, active_vertices, priority, cuda_ctx);
#else
        SSSP_nodesWithEdges_cuda(__retval, __retval2, priority, cuda_ctx);
#endif
        dga += __retval;
        work_edges += __retval2;
        StatTimer_cuda.stop();
#if DIST_PER_ROUND_TIMER
        std::string identifer(_graph.get_run_identifier("GPUThreadBlocksWork_Host", galois::runtime::getSystemNetworkInterface().ID));
        std::string tb_identifer(_graph.get_run_identifier("ThreadBlocks_Host", galois::runtime::getSystemNetworkInterface().ID));
        ReportThreadBlockWork(_num_iterations, identifer, tb_identifer);

        std::string acive_identifer(_graph.get_run_identifier("NumActiveVertices"));
        galois::runtime::reportParam(REGION_NAME, acive_identifer, std::to_string(active_vertices));
#endif
      } else if (personality == CPU)
#endif
      {
        galois::do_all(
            galois::iterate(nodesWithEdges), SSSP{priority, &_graph, dga, work_edges},
            galois::no_stats(),
            galois::loopname(_graph.get_run_identifier("SSSP").c_str()),
            galois::steal());
      }

#ifdef __GALOIS_HET_ASYNC__
      _graph.sync<writeDestination, readSource, Reduce_min_dist_current,
                  Bitset_dist_current, true>("SSSP");
#else
      _graph.sync<writeDestination, readSource, Reduce_min_dist_current,
                  Bitset_dist_current>("SSSP");
#endif

      galois::runtime::reportStat_Tsum(
          "SSSP", "NumWorkItems_" + (_graph.get_run_identifier()),
          (unsigned long)work_edges.reduce());
      ++_num_iterations;
    } while (
#ifndef __GALOIS_HET_ASYNC__
             (_num_iterations < maxIterations) &&
#endif
             dga.reduce(_graph.get_run_identifier()));

    galois::runtime::reportStat_Tmax(
        "SSSP", "NumIterations_" + std::to_string(_graph.get_run_num()),
        (unsigned long)_num_iterations);
  }

  void operator()(GNode src) const {
    NodeData& snode = graph->getData(src);

    if (snode.dist_old > snode.dist_current) {
      active_vertices += 1;

      if (local_priority > snode.dist_current) {
        snode.dist_old = snode.dist_current;

        for (auto jj : graph->edges(src)) {
          work_edges += 1;

          GNode dst         = graph->getEdgeDst(jj);
          auto& dnode       = graph->getData(dst);
          uint32_t new_dist = graph->getEdgeData(jj) + snode.dist_current;
          uint32_t old_dist = galois::atomicMin(dnode.dist_current, new_dist);
          if (old_dist > new_dist)
            bitset_dist_current.set(dst);
        }
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
  galois::DGAccumulator<uint64_t>& dg_avg;

  SSSPSanityCheck(const uint32_t& _infinity, Graph* _graph,
                  galois::DGAccumulator<uint64_t>& dgas,
                  galois::DGReduceMax<uint32_t>& dgm,
                  galois::DGAccumulator<uint64_t>& _dg_avg)
      : local_infinity(_infinity), graph(_graph), DGAccumulator_sum(dgas),
        DGMax(dgm), dg_avg(_dg_avg) {}

  void static go(Graph& _graph, galois::DGAccumulator<uint64_t>& dgas,
                 galois::DGReduceMax<uint32_t>& dgm,
                 galois::DGAccumulator<uint64_t>& dgag) {
    dgas.reset();
    dgm.reset();
    dgag.reset();

#ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      uint64_t sum, avg;
      uint32_t max;
      SSSPSanityCheck_masterNodes_cuda(sum, avg, max, infinity, cuda_ctx);
      dgas += sum;
      dgm.update(max);
      dgag += avg;
    } else
#endif
    {
      galois::do_all(galois::iterate(_graph.masterNodesRange().begin(),
                                     _graph.masterNodesRange().end()),
                     SSSPSanityCheck(infinity, &_graph, dgas, dgm, dgag),
                     galois::no_stats(), galois::loopname("SSSPSanityCheck"));
    }

    uint64_t num_visited  = dgas.reduce();
    uint32_t max_distance = dgm.reduce();

    float visit_average = ((float)dgag.reduce()) / num_visited;

    // Only host 0 will print the info
    if (galois::runtime::getSystemNetworkInterface().ID == 0) {
      galois::gPrint("Number of nodes visited from source ", src_node, " is ",
                     num_visited, "\n");
      galois::gPrint("Max distance from source ", src_node, " is ",
                     max_distance, "\n");
      galois::gPrint("Average distances on visited nodes is ", visit_average,
                     "\n");
    }
  }

  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    if (src_data.dist_current < local_infinity) {
      DGAccumulator_sum += 1;
      DGMax.update(src_data.dist_current);
      dg_avg += src_data.dist_current;
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
  }

  galois::StatTimer StatTimer_total("TimerTotal", REGION_NAME);

  StatTimer_total.start();

#ifdef __GALOIS_HET_CUDA__
  Graph* hg = distGraphInitialization<NodeData, unsigned int>(&cuda_ctx);
#else
  Graph* hg = distGraphInitialization<NodeData, unsigned int>();
#endif

  bitset_dist_current.resize(hg->size());

  galois::gPrint("[", net.ID, "] InitializeGraph::go called\n");

  InitializeGraph::go((*hg));
  galois::runtime::getHostBarrier().wait();

  // accumulators for use in operators
#ifdef __GALOIS_HET_ASYNC__
  galois::DGTerminator<unsigned int> active_vertices;
#else
  galois::DGAccumulator<unsigned int> active_vertices;
#endif
  galois::DGAccumulator<uint64_t> DGAccumulator_sum;
  galois::DGAccumulator<uint64_t> dg_avge;
  galois::DGReduceMax<uint32_t> m;

  for (auto run = 0; run < numRuns; ++run) {
    galois::gPrint("[", net.ID, "] SSSP::go run ", run, " called\n");
    std::string timer_str("Timer_" + std::to_string(run));
    galois::StatTimer StatTimer_main(timer_str.c_str(), REGION_NAME);

    StatTimer_main.start();
    SSSP::go(*hg, active_vertices);
    StatTimer_main.stop();

    SSSPSanityCheck::go(*hg, DGAccumulator_sum, m, dg_avge);

    if ((run + 1) != numRuns) {
#ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) {
        bitset_dist_current_reset_cuda(cuda_ctx);
      } else
#endif
        bitset_dist_current.reset();

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
