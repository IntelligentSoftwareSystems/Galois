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
#include "galois/DTerminationDetector.h"
#include "galois/runtime/Tracer.h"

#ifdef __GALOIS_HET_CUDA__
#include "bfs_pull_cuda.h"
struct CUDA_Context* cuda_ctx;
#else
enum { CPU, GPU_CUDA };
int personality = CPU;
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

enum Exec { Sync, Async };

static cll::opt<Exec> execution(
    "exec",
    cll::desc("Distributed Execution Model (default value Async):"),
    cll::values(clEnumVal(Sync, "Bulk-synchronous Parallel (BSP)"), 
    clEnumVal(Async, "Bulk-asynchronous Parallel (BASP)")),
    cll::init(Async));

/******************************************************************************/
/* Graph structure declarations + other initialization */
/******************************************************************************/

const uint32_t infinity = std::numeric_limits<uint32_t>::max() / 4;

struct NodeData {
  uint32_t dist_current;
};

typedef galois::graphs::DistGraph<NodeData, void> Graph;
typedef typename Graph::GraphNode GNode;

galois::DynamicBitSet bitset_dist_current;

galois::graphs::GluonSubstrate<Graph>* syncSubstrate;

#include "bfs_pull_sync.hh"

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
    if (personality == GPU_CUDA) {
#ifdef __GALOIS_HET_CUDA__
      std::string impl_str("InitializeGraph_" + (syncSubstrate->get_run_identifier()));
      galois::StatTimer StatTimer_cuda(impl_str.c_str(), regionname);
      StatTimer_cuda.start();
      InitializeGraph_allNodes_cuda(infinity, src_node, cuda_ctx);
      StatTimer_cuda.stop();
#else
      abort();
#endif
    } else if (personality == CPU) {
      galois::do_all(galois::iterate(allNodes),
                     InitializeGraph(src_node, infinity, &_graph),
                     galois::no_stats(),
                     galois::loopname(
                         syncSubstrate->get_run_identifier("InitializeGraph").c_str()));
    }
  }

  void operator()(GNode src) const {
    NodeData& sdata = graph->getData(src);
    sdata.dist_current = (graph->getGID(src) == local_src_node) ? 0 : local_infinity;
  }
};

template <bool async>
struct BFS {
  Graph* graph;
  using DGTerminatorDetector = typename std::conditional<async, 
          galois::DGTerminator<unsigned int>,
          galois::DGAccumulator<unsigned int>>::type;

  DGTerminatorDetector& active_vertices;

  BFS(Graph* _graph, DGTerminatorDetector& _dga)
      : graph(_graph), active_vertices(_dga) {}

  void static go(Graph& _graph) {
    unsigned _num_iterations = 0;
    DGTerminatorDetector dga;

    const auto& nodesWithEdges = _graph.allNodesWithEdgesRange();
    do {
      syncSubstrate->set_num_round(_num_iterations);
      dga.reset();
      if (personality == GPU_CUDA) {
#ifdef __GALOIS_HET_CUDA__
        std::string impl_str("BFS_" + (syncSubstrate->get_run_identifier()));
        galois::StatTimer StatTimer_cuda(impl_str.c_str(), regionname);
        StatTimer_cuda.start();
        unsigned int __retval = 0;
        BFS_nodesWithEdges_cuda(__retval, cuda_ctx);
        dga += __retval;
        StatTimer_cuda.stop();
#else
        abort();
#endif
      } else if (personality == CPU) {
        galois::do_all(
            galois::iterate(nodesWithEdges), BFS(&_graph, dga),
            galois::no_stats(), galois::steal(),
            galois::loopname(syncSubstrate->get_run_identifier("BFS").c_str()));
      }
      syncSubstrate->sync<writeSource, readDestination, Reduce_min_dist_current,
                          Bitset_dist_current, async>("BFS");

      galois::runtime::reportStat_Tsum(
          regionname, syncSubstrate->get_run_identifier("NumWorkItems"),
          (unsigned long)dga.read_local());
      ++_num_iterations;
    } while (
             (async || (_num_iterations < maxIterations)) &&
             dga.reduce(syncSubstrate->get_run_identifier()));

    if (galois::runtime::getSystemNetworkInterface().ID == 0) {
      galois::runtime::reportStat_Single(
          regionname, "NumIterations_" + std::to_string(syncSubstrate->get_run_num()),
          (unsigned long)_num_iterations);
    }
  }

  void operator()(GNode src) const {
    NodeData& snode = graph->getData(src);

    for (auto jj : graph->edges(src)) {
      GNode dst         = graph->getEdgeDst(jj);
      auto& dnode       = graph->getData(dst);
      uint32_t new_dist = dnode.dist_current + 1;
      uint32_t old_dist = galois::min(snode.dist_current, new_dist);
      if (old_dist > new_dist) {
        bitset_dist_current.set(src);
        active_vertices += 1;
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

    if (personality == GPU_CUDA) {
#ifdef __GALOIS_HET_CUDA__
      uint64_t sum;
      uint32_t max;
      BFSSanityCheck_masterNodes_cuda(sum, max, infinity, cuda_ctx);
      dgas += sum;
      dgm.update(max);
#else
      abort();
#endif
    } else {
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

static const char* const name = "BFS pull - Distributed Heterogeneous";
static const char* const desc = "BFS pull on Distributed Galois.";
static const char* const url  = 0;

int main(int argc, char** argv) {
  galois::DistMemSys G;
  DistBenchStart(argc, argv, name, desc, url);

  auto& net = galois::runtime::getSystemNetworkInterface();
  if (net.ID == 0) {
    galois::runtime::reportParam(regionname, "Max Iterations",
                                 (unsigned long)maxIterations);
    galois::runtime::reportParam(regionname, "Source Node ID",
                                 (unsigned long long)src_node);
  }
  galois::StatTimer StatTimer_total("TimerTotal", regionname);

  StatTimer_total.start();

  Graph* hg;
#ifdef __GALOIS_HET_CUDA__
  std::tie(hg, syncSubstrate) = distGraphInitialization<NodeData, void, false>(&cuda_ctx);
#else
  std::tie(hg, syncSubstrate) = distGraphInitialization<NodeData, void,
                                                        false>();
#endif

  bitset_dist_current.resize(hg->size());

  galois::gPrint("[", net.ID, "] InitializeGraph::go called\n");

  InitializeGraph::go((*hg));
  galois::runtime::getHostBarrier().wait();

  // accumulators for use in operators
  galois::DGAccumulator<uint64_t> DGAccumulator_sum;
  galois::DGReduceMax<uint32_t> m;

  for (auto run = 0; run < numRuns; ++run) {
    galois::gPrint("[", net.ID, "] BFS::go run ", run, " called\n");
    std::string timer_str("Timer_" + std::to_string(run));
    galois::StatTimer StatTimer_main(timer_str.c_str(), regionname);

    StatTimer_main.start();
    if (execution == Async) {
      BFS<true>::go(*hg);
    } else {
      BFS<false>::go(*hg);
    }
    StatTimer_main.stop();

    // sanity check
    BFSSanityCheck::go(*hg, DGAccumulator_sum, m);

    if ((run + 1) != numRuns) {
      if (personality == GPU_CUDA) {
#ifdef __GALOIS_HET_CUDA__
        bitset_dist_current_reset_cuda(cuda_ctx);
#else
        abort();
#endif
      } else {
        bitset_dist_current.reset();
      }

      (*syncSubstrate).set_num_run(run + 1);
      InitializeGraph::go((*hg));
      galois::runtime::getHostBarrier().wait();
    }
  }

  StatTimer_total.stop();

  // Verify
  if (verify) {
    if (personality == CPU) {
      for (auto ii = (*hg).masterNodesRange().begin();
           ii != (*hg).masterNodesRange().end(); ++ii) {
        galois::runtime::printOutput("% %\n", (*hg).getGID(*ii),
                                     (*hg).getData(*ii).dist_current);
      }
    } else if (personality == GPU_CUDA) {
#ifdef __GALOIS_HET_CUDA__
      for (auto ii = (*hg).masterNodesRange().begin();
           ii != (*hg).masterNodesRange().end(); ++ii) {
        if ((*hg).isOwned((*hg).getGID(*ii)))
          galois::runtime::printOutput(
              "% %\n", (*hg).getGID(*ii),
              get_node_dist_current_cuda(cuda_ctx, *ii));
      }
#else
    abort();
#endif
    }
  }
  return 0;
}
