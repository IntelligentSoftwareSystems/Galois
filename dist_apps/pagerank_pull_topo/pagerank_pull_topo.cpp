/** Page Rank Pull Topological -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
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
 *
 * @section Description
 *
 * Compute pagerank pull topological version on distributed Galois.
 *
 * @author Loc Hoang <l_hoang@utexas.edu>
 */

#include "galois/DistGalois.h"
#include "DistBenchStart.h"
#include "galois/gstl.h"
#include "galois/runtime/Tracer.h"
#include "galois/DReducible.h"

#ifdef __GALOIS_HET_CUDA__
#include "pagerank_pull_topo_cuda.h"
struct CUDA_Context *cuda_ctx;
#endif

#include <iostream>
#include <limits>
#include <algorithm>
#include <vector>

constexpr static const char* const REGION_NAME = "PageRank";

/******************************************************************************/
/* Declaration of command line arguments */
/******************************************************************************/
namespace cll = llvm::cl;

static cll::opt<float> tolerance("tolerance", 
                                 cll::desc("tolerance for residual"), 
                                 cll::init(0.000001));
static cll::opt<unsigned int> maxIterations("maxIterations", 
                                cll::desc("Maximum iterations: Default 1000"),
                                cll::init(1000));

/******************************************************************************/
/* Graph structure declarations + other initialization */
/******************************************************************************/

static const float alpha = (1.0 - 0.85);
struct NodeData {
  float value;
  std::atomic<uint32_t> nout;
  float partialSum;
};

galois::DynamicBitSet bitset_nout;
galois::DynamicBitSet bitset_partialSum;

typedef galois::graphs::DistGraph<NodeData, void> Graph;
typedef typename Graph::GraphNode GNode;

#include "pagerank_pull_topo_sync.hh"

/******************************************************************************/
/* Algorithm structures */
/******************************************************************************/

/**
 * All fields reset to default values
 */
struct ResetGraph {
  const float& local_alpha;
  Graph* graph;

  ResetGraph(const float& _local_alpha, Graph* _graph) : 
      local_alpha(_local_alpha), graph(_graph) {}

  void static go(Graph& _graph) {
    const auto& allNodes = _graph.allNodesRange();
    #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      std::string impl_str("CUDA_DO_ALL_IMPL_ResetGraph_" + (_graph.get_run_identifier()));
      galois::StatTimer StatTimer_cuda(impl_str.c_str(), REGION_NAME);
      StatTimer_cuda.start();
      ResetGraph_allNodes_cuda(alpha, cuda_ctx);
      StatTimer_cuda.stop();
    } else if (personality == CPU)
    #endif
    galois::do_all(
      galois::iterate(allNodes.begin(), allNodes.end()),
      ResetGraph{ alpha, &_graph },
      galois::no_stats(), 
      galois::loopname(_graph.get_run_identifier("ResetGraph").c_str())
    );
  }

  void operator()(GNode src) const {
    auto& sdata = graph->getData(src);
    sdata.value = local_alpha;
    sdata.nout = 0;
    sdata.partialSum = 0;
  }
};

/**
 * Determine nout
 */
struct InitializeGraph {
  Graph* graph;

  InitializeGraph(Graph* _graph) : graph(_graph) {}

  void static go(Graph& _graph) {
    // init graph
    ResetGraph::go(_graph);

    const auto& nodesWithEdges = _graph.allNodesWithEdgesRange();

    #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      std::string impl_str("CUDA_DO_ALL_IMPL_InitializeGraph_" + 
        (_graph.get_run_identifier()));
      galois::StatTimer StatTimer_cuda(impl_str.c_str(), REGION_NAME);
      StatTimer_cuda.start();
      InitializeGraph_nodesWithEdges_cuda(cuda_ctx);
      StatTimer_cuda.stop();
    } else if (personality == CPU)
    #endif
    // doing a local do all because we are looping over edges
    galois::do_all(
      galois::iterate(nodesWithEdges),
      InitializeGraph{ &_graph },
      galois::steal(), 
      galois::no_stats(), 
      galois::loopname(_graph.get_run_identifier("InitializeGraph").c_str()));

    _graph.sync<writeDestination, readAny, Reduce_add_nout, Broadcast_nout,
                Bitset_nout>("InitializeGraph");
  }

  // Calculate "outgoing" edges for destination nodes (note we are using
  // the tranpose graph for pull algorithms)
  void operator()(GNode src) const {
    for (auto nbr : graph->edges(src)) {
      GNode dst = graph->getEdgeDst(nbr);
      auto& ddata = graph->getData(dst);
      galois::atomicAdd(ddata.nout, (uint32_t)1);
      bitset_nout.set(dst);
    }
  }
};

/**
 * Determine if calculated pagerank is above tolerance; if so, replace old
 * pagerank value with new one
 */
struct PageRankSum {
  const float local_alpha;
  cll::opt<float> local_tolerance;
  Graph* graph;
  galois::DGAccumulator<unsigned int>& DGAccumulator_accum;

  PageRankSum(const float a, cll::opt<float> _local_tolerance, Graph* _graph,
              galois::DGAccumulator<unsigned int>& _dga)
    : local_alpha(a), local_tolerance(_local_tolerance), graph(_graph), 
      DGAccumulator_accum(_dga) {}

  void static go(Graph& _graph, galois::DGAccumulator<unsigned int>& dga) {
    const auto& allNodes = _graph.allNodesRange();

    #ifdef __GALOIS_HET_CUDA__
    //if (personality == GPU_CUDA) {
    //} else if (personality == CPU)
    #endif
    galois::do_all(
      galois::iterate(allNodes.begin(), allNodes.end()),
      PageRankSum{ alpha, tolerance, &_graph, dga },
      galois::no_stats(), 
      galois::loopname(_graph.get_run_identifier("PageRank").c_str())
    );
  }

  // Check if partial sum is greater than current
  void operator()(GNode src) const {
    auto& sdata = graph->getData(src);
    sdata.partialSum += local_alpha;

    if (std::fabs(sdata.value - sdata.partialSum) > local_tolerance) {
      DGAccumulator_accum += 1;
      sdata.value = sdata.partialSum;
    }

    sdata.partialSum = 0;
  }
};

/**
 * Calculate page rank on all nodes with edges then reduce.
 */
struct PageRank {
  const float local_alpha;
  Graph* graph;

  PageRank(const float a, Graph* _graph) : local_alpha(a), graph(_graph) {}

  void static go(Graph& _graph, galois::DGAccumulator<unsigned int>& dga) {
    unsigned _num_iterations = 0;
    const auto& nodesWithEdges = _graph.allNodesWithEdgesRange();

    do {
      _graph.set_num_iter(_num_iterations);

      //#ifdef __GALOIS_HET_CUDA__
      //if (personality == GPU_CUDA) {
      //  std::string impl_str("CUDA_DO_ALL_IMPL_PageRank_" + 
      //    (_graph.get_run_identifier()));
      //  galois::StatTimer StatTimer_cuda(impl_str.c_str(), REGION_NAME);
      //  StatTimer_cuda.start();
      //  PageRank_nodesWithEdges_cuda(cuda_ctx);
      //  StatTimer_cuda.stop();
      //} else if (personality == CPU)
      //#endif
      galois::do_all(
        galois::iterate(nodesWithEdges),
        PageRank{ alpha, &_graph },
        galois::steal(),
        galois::no_stats(), 
        galois::loopname(_graph.get_run_identifier("PageRank").c_str())
      );

      _graph.sync<writeSource, readAny, Reduce_add_partialSum, 
                  Broadcast_partialSum, Bitset_partialSum>("PageRank");
      
      dga.reset();
      PageRankSum::go(_graph, dga);

      galois::runtime::reportStat_Tsum(REGION_NAME, 
        "NUM_WORK_ITEMS_" + (_graph.get_run_identifier()), 
        (unsigned long)dga.read_local()
      );

      ++_num_iterations;
    } while ((_num_iterations < maxIterations) && dga.reduce(_graph.get_run_identifier()));

    if (galois::runtime::getSystemNetworkInterface().ID == 0) {
      galois::runtime::reportStat_Single(REGION_NAME, 
        "NUM_ITERATIONS_" + std::to_string(_graph.get_run_num()), 
        (unsigned long)_num_iterations);
    }
  }

  // Calculate value to get from neighbors, add to partial sum
  void operator()(GNode src) const {
    auto& sdata = graph->getData(src);

    for (auto nbr : graph->edges(src)) {
      GNode dst = graph->getEdgeDst(nbr);
      auto& ddata = graph->getData(dst);
      sdata.partialSum += (ddata.value * (1 - local_alpha)) / ddata.nout;
    }

    if (sdata.partialSum > 0) {
      bitset_partialSum.set(src);
    }
  }
};

/******************************************************************************/
/* Sanity check operators */
/******************************************************************************/

// Gets various values from the pageranks values/residuals of the graph
struct PageRankSanity {
  cll::opt<float>& local_tolerance;
  Graph* graph;

  galois::DGAccumulator<float>& DGAccumulator_sum;
  galois::DGReduceMax<float>& max_value;
  galois::DGReduceMin<float>& min_value;

  PageRankSanity(cll::opt<float>& _local_tolerance, Graph* _graph,
      galois::DGAccumulator<float>& _DGAccumulator_sum,
      galois::DGReduceMax<float>& _max_value,
      galois::DGReduceMin<float>& _min_value
  ) : local_tolerance(_local_tolerance), graph(_graph),
      DGAccumulator_sum(_DGAccumulator_sum),
      max_value(_max_value),
      min_value(_min_value) {}

  void static go(Graph& _graph,
    galois::DGAccumulator<float>& DGA_sum,
    galois::DGReduceMax<float>& max_value,
    galois::DGReduceMin<float>& min_value
  ) {
    DGA_sum.reset();
    max_value.reset();
    min_value.reset();

  #ifdef __GALOIS_HET_CUDA__
    //if (personality == GPU_CUDA) {
    //} else
  #endif
    {
      galois::do_all(
        galois::iterate(_graph.masterNodesRange().begin(), 
                        _graph.masterNodesRange().end()), 
        PageRankSanity(tolerance, &_graph, DGA_sum, max_value, min_value), 
        galois::no_stats(), 
        galois::loopname("PageRankSanity")
      );
    }

    float max_rank = max_value.reduce();
    float min_rank = min_value.reduce();
    float rank_sum = DGA_sum.reduce();

    // Only host 0 will print data
    if (galois::runtime::getSystemNetworkInterface().ID == 0) {
      galois::gPrint("Max rank is ", max_rank, "\n");
      galois::gPrint("Min rank is ", min_rank, "\n");
      galois::gPrint("Rank sum is ", rank_sum, "\n");
    }
  }
  
  /**
   * Max, min, sum ranks
   */
  void operator()(GNode src) const {
    NodeData& sdata = graph->getData(src);

    max_value.update(sdata.value);
    min_value.update(sdata.value);
    DGAccumulator_sum += sdata.value;
  }
};

/******************************************************************************/
/* Main */
/******************************************************************************/

constexpr static const char* const name = "PageRank - Compiler Generated "
                                          "Distributed Heterogeneous";
constexpr static const char* const desc = "PageRank Pull Topological version on "
                                          "Distributed Galois.";
constexpr static const char* const url = 0;

int main(int argc, char** argv) {
  galois::DistMemSys G;
  DistBenchStart(argc, argv, name, desc, url);

  auto& net = galois::runtime::getSystemNetworkInterface();

  if (net.ID == 0) {
    galois::runtime::reportParam(REGION_NAME, "Max Iterations", 
                                (unsigned long)maxIterations);
    std::ostringstream ss;
    ss << tolerance;
    galois::runtime::reportParam(REGION_NAME, "Tolerance", ss.str());
  }

  galois::StatTimer StatTimer_total("TIMER_TOTAL", REGION_NAME);

  StatTimer_total.start();

  #ifdef __GALOIS_HET_CUDA__
  Graph* hg = distGraphInitialization<NodeData, void, false>(&cuda_ctx);
  #else
  Graph* hg = distGraphInitialization<NodeData, void, false>();
  #endif

  bitset_partialSum.resize(hg->size());
  bitset_nout.resize(hg->size());

  galois::gPrint("[", net.ID, "] InitializeGraph::go called\n");
  galois::StatTimer StatTimer_init("TIMER_GRAPH_INIT", REGION_NAME);
  StatTimer_init.start();
    InitializeGraph::go(*hg);
  StatTimer_init.stop();
  galois::runtime::getHostBarrier().wait();

  galois::DGAccumulator<unsigned int> PageRank_accum;

  galois::DGAccumulator<float> DGA_sum;
  galois::DGReduceMax<float> max_value;
  galois::DGReduceMin<float> min_value;

  for (auto run = 0; run < numRuns; ++run) {
    galois::gPrint("[", net.ID, "] PageRank::go run ", run, " called\n");
    std::string timer_str("TIMER_" + std::to_string(run));
    galois::StatTimer StatTimer_main(timer_str.c_str(), REGION_NAME);

    StatTimer_main.start();
    PageRank::go(*hg, PageRank_accum);
    StatTimer_main.stop();

    // sanity check
    PageRankSanity::go(*hg, DGA_sum, max_value, min_value);

    if ((run + 1) != numRuns) {
      #ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) { 
        bitset_residual_reset_cuda(cuda_ctx);
        bitset_nout_reset_cuda(cuda_ctx);
      } else
      #endif
      { 
        bitset_partialSum.reset();
        bitset_nout.reset(); 
      }

      (*hg).set_num_run(run+1);
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
                ii != (*hg).masterNodesRange().end(); 
                ++ii) {
        galois::runtime::printOutput("% %\n", (*hg).getGID(*ii), 
          (*hg).getData(*ii).value);
      }
    #ifdef __GALOIS_HET_CUDA__
    } else if (personality == GPU_CUDA) {
      for (auto ii = (*hg).masterNodesRange().begin(); 
                ii != (*hg).masterNodesRange().end(); 
                ++ii) {
        galois::runtime::printOutput("% %\n", (*hg).getGID(*ii), 
          get_node_value_cuda(cuda_ctx, *ii));
      }
    }
    #endif
  }

  return 0;
}
