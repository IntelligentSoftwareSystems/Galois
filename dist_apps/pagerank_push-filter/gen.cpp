/** Residual based Page Rank -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
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
 * Compute pageRank using residual on distributed Galois.
 *
 * @author Gurbinder Gill <gurbinder533@gmail.com>
 * @author Roshan Dathathri <roshan@cs.utexas.edu>
 * @author Loc Hoang <l_hoang@utexas.edu> (sanity check operators)
 */
#include "galois/DistGalois.h"
#include "DistBenchStart.h"
#include "galois/gstl.h"
#include "galois/runtime/Tracer.h"
#include "galois/DistAccumulator.h"

#ifdef __GALOIS_HET_CUDA__
#include "gen_cuda.h"
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
  float delta;
  std::atomic<float> residual;
};

galois::DynamicBitSet bitset_residual;
galois::DynamicBitSet bitset_nout;

typedef hGraph<NodeData, void> Graph;
typedef typename Graph::GraphNode GNode;
typedef GNode WorkItem;

#include "gen_sync.hh"

/******************************************************************************/
/* Algorithm structures */
/******************************************************************************/

// Reset all fields of all nodes to 0
struct ResetGraph {
  Graph* graph;

  ResetGraph(Graph* _graph) : graph(_graph){}
  void static go(Graph& _graph) {
    const auto& allNodes = _graph.allNodesRange();
    #ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) {
        std::string impl_str("CUDA_DO_ALL_IMPL_ResetGraph_" + (_graph.get_run_identifier()));
        galois::StatTimer StatTimer_cuda(impl_str.c_str(), REGION_NAME);
        StatTimer_cuda.start();
        ResetGraph_cuda(*allNodes.begin(), *allNodes.end(), cuda_ctx);
        StatTimer_cuda.stop();
      } else if (personality == CPU)
    #endif
    galois::do_all(
      galois::iterate(allNodes.begin(), allNodes.end()),
      ResetGraph{ &_graph },
      galois::loopname(_graph.get_run_identifier("ResetGraph").c_str()),
      galois::timeit(),
      galois::no_stats()
    );
  }

  void operator()(GNode src) const {
    NodeData& sdata = graph->getData(src);
    sdata.value = 0;
    sdata.nout = 0;
    sdata.residual = 0;
    sdata.delta = 0;
  }
};

// Initialize residual at nodes with outgoing edges + find nout for
// nodes with outgoing edges
struct InitializeGraph {
  const float &local_alpha;
  Graph* graph;

  InitializeGraph(const float &_alpha, Graph* _graph) : 
    local_alpha(_alpha), graph(_graph){}

  void static go(Graph& _graph) {
    // first initialize all fields to 0 via ResetGraph (can't assume all zero
    // at start)
    ResetGraph::go(_graph);

    const auto& nodesWithEdges = _graph.allNodesWithEdgesRange();

    #ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) {
        std::string impl_str("CUDA_DO_ALL_IMPL_InitializeGraph_" + 
          (_graph.get_run_identifier()));
        galois::StatTimer StatTimer_cuda(impl_str.c_str(), REGION_NAME);
        StatTimer_cuda.start();
        InitializeGraph_cuda(*nodesWithEdges.begin(), *nodesWithEdges.end(), alpha, 
                             cuda_ctx);
        StatTimer_cuda.stop();
      } else if (personality == CPU)
    #endif
    {
     // regular do all without stealing; just initialization of nodes with
     // outgoing edges
     galois::do_all(
        galois::iterate(nodesWithEdges.begin(), nodesWithEdges.end()),
        InitializeGraph{alpha, &_graph},
        galois::loopname(_graph.get_run_identifier("InitializeGraph").c_str()),
        galois::timeit(),
        galois::no_stats()
      );
    }

    _graph.sync<writeSource, readSource, Reduce_add_nout, Broadcast_nout,
                Bitset_nout>("InitializeGraphNout");
  }

  void operator()(GNode src) const {
    NodeData& sdata = graph->getData(src);
    sdata.residual = local_alpha;
    galois::atomicAdd(sdata.nout, 
      (uint32_t) std::distance(graph->edge_begin(src), 
                               graph->edge_end(src)));
    bitset_nout.set(src);
  }
};

struct PageRank_delta {
  const float & local_alpha;
  cll::opt<float> & local_tolerance;
  Graph* graph;

  PageRank_delta(const float & _local_alpha, cll::opt<float> & _local_tolerance,
                 Graph * _graph) : 
      local_alpha(_local_alpha),
      local_tolerance(_local_tolerance),
      graph(_graph) {}

  void static go(Graph& _graph) {
    const auto& nodesWithEdges = _graph.allNodesWithEdgesRange();

  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      std::string impl_str("CUDA_DO_ALL_IMPL_PageRank_" + (_graph.get_run_identifier()));
      galois::StatTimer StatTimer_cuda(impl_str.c_str(), REGION_NAME);
      StatTimer_cuda.start();
      PageRank_delta_cuda(*nodesWithEdges.begin(), *nodesWithEdges.end(),
                          alpha, tolerance, cuda_ctx);
      StatTimer_cuda.stop();
    } else if (personality == CPU)
  #endif
    {
      galois::do_all(
        galois::iterate(nodesWithEdges.begin(), nodesWithEdges.end()),
        PageRank_delta{ alpha, tolerance, &_graph },
        galois::loopname(_graph.get_run_identifier("PageRank_delta").c_str()),
        galois::timeit(),
        galois::no_stats()
      );
    }
  }

  void operator()(WorkItem src) const {
    NodeData& sdata = graph->getData(src);

    if (sdata.residual > this->local_tolerance) {
      float residual_old = sdata.residual;
      sdata.residual = 0;
      sdata.value += residual_old;
      if (sdata.nout > 0) {
        sdata.delta = residual_old * (1 - local_alpha) / sdata.nout;
      }
    }
  }
};

struct PageRank {
  Graph* graph;
  galois::DGAccumulator<unsigned int>& DGAccumulator_accum;

  PageRank(Graph* _g, galois::DGAccumulator<unsigned int>& _dga): 
    graph(_g), DGAccumulator_accum(_dga) {}

  void static go(Graph& _graph, galois::DGAccumulator<unsigned int>& dga) {
    unsigned _num_iterations = 0;
    const auto& nodesWithEdges = _graph.allNodesWithEdgesRange();

    do { 
      _graph.set_num_iter(_num_iterations);
      PageRank_delta::go(_graph);
      dga.reset();
      #ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) {
        std::string impl_str("CUDA_DO_ALL_IMPL_PageRank_" + (_graph.get_run_identifier()));
        galois::StatTimer StatTimer_cuda(impl_str.c_str(), REGION_NAME);
        StatTimer_cuda.start();
        int __retval = 0;
        PageRank_cuda(*nodesWithEdges.begin(), *nodesWithEdges.end(), 
                      __retval, cuda_ctx);
        dga += __retval;
        StatTimer_cuda.stop();
      } else if (personality == CPU)
      #endif
      {
        galois::do_all(
          galois::iterate(nodesWithEdges),
          PageRank{ &_graph, dga },
          galois::loopname(_graph.get_run_identifier("PageRank").c_str()),
          galois::timeit(),
          galois::no_stats()
        );
      }

      _graph.sync<writeDestination, readSource, Reduce_add_residual, 
                  Broadcast_residual, Bitset_residual>("PageRank");
      
      galois::runtime::reportStat_Tsum(REGION_NAME, 
          "NUM_WORK_ITEMS_" + (_graph.get_run_identifier()), 
          (unsigned long)dga.read_local());

      ++_num_iterations;
    } while ((_num_iterations < maxIterations) && dga.reduce(_graph.get_run_identifier()));

    if (galois::runtime::getSystemNetworkInterface().ID == 0) {
      galois::runtime::reportStat_Single(REGION_NAME, 
        "NUM_ITERATIONS_" + std::to_string(_graph.get_run_num()), 
        (unsigned long)_num_iterations);
    }
  }

  void operator()(WorkItem src) const {
    NodeData& sdata = graph->getData(src);
    if (sdata.delta > 0) {
      float _delta = sdata.delta;
      sdata.delta = 0;
      for(auto nbr = graph->edge_begin(src), ee = graph->edge_end(src); 
          nbr != ee; ++nbr) {
        GNode dst = graph->getEdgeDst(nbr);
        NodeData& ddata = graph->getData(dst);

        galois::atomicAdd(ddata.residual, _delta);

        bitset_residual.set(dst);
      }
      DGAccumulator_accum+= 1; // this should be moved to PagerankCopy operator
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

  static float current_max;
  static float current_min;
  static float current_max_residual;
  static float current_min_residual;

  galois::DGAccumulator<float>& DGAccumulator_max;
  galois::DGAccumulator<float>& DGAccumulator_min;
  galois::DGAccumulator<float>& DGAccumulator_sum;
  galois::DGAccumulator<float>& DGAccumulator_sum_residual;
  galois::DGAccumulator<uint64_t>& DGAccumulator_residual_over_tolerance;
  galois::DGAccumulator<float>& DGAccumulator_max_residual;
  galois::DGAccumulator<float>& DGAccumulator_min_residual;

  PageRankSanity(cll::opt<float>& _local_tolerance, Graph* _graph,
      galois::DGAccumulator<float>& _DGAccumulator_max,
      galois::DGAccumulator<float>& _DGAccumulator_min,
      galois::DGAccumulator<float>& _DGAccumulator_sum,
      galois::DGAccumulator<float>& _DGAccumulator_sum_residual,
      galois::DGAccumulator<uint64_t>& _DGAccumulator_residual_over_tolerance,
      galois::DGAccumulator<float>& _DGAccumulator_max_residual,
      galois::DGAccumulator<float>& _DGAccumulator_min_residual
  ) : 
    local_tolerance(_local_tolerance), graph(_graph),
    DGAccumulator_max(_DGAccumulator_max),
    DGAccumulator_min(_DGAccumulator_min),
    DGAccumulator_sum(_DGAccumulator_sum),
    DGAccumulator_sum_residual(_DGAccumulator_sum_residual),
    DGAccumulator_residual_over_tolerance(_DGAccumulator_residual_over_tolerance),
    DGAccumulator_max_residual(_DGAccumulator_max_residual),
    DGAccumulator_min_residual(_DGAccumulator_min_residual) {}

  void static go(Graph& _graph,
    galois::DGAccumulator<float>& DGA_max,
    galois::DGAccumulator<float>& DGA_min,
    galois::DGAccumulator<float>& DGA_sum,
    galois::DGAccumulator<float>& DGA_sum_residual,
    galois::DGAccumulator<uint64_t>& DGA_residual_over_tolerance,
    galois::DGAccumulator<float>& DGA_max_residual,
    galois::DGAccumulator<float>& DGA_min_residual
  ) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      // TODO currently no GPU support for sanity check operator
      fprintf(stderr, "Warning: No GPU support for sanity check; might get "
                      "wrong results.\n");
    }
  #endif
    DGA_max.reset();
    DGA_min.reset();
    DGA_sum.reset();
    DGA_sum_residual.reset();
    DGA_residual_over_tolerance.reset();
    DGA_max_residual.reset();
    DGA_min_residual.reset();

    galois::do_all(galois::iterate(_graph.allNodesRange().begin(), _graph.allNodesRange().end()), 
                   PageRankSanity(
                     tolerance, 
                     &_graph,
                     DGA_max,
                     DGA_min,
                     DGA_sum,
                     DGA_sum_residual,
                     DGA_residual_over_tolerance,
                     DGA_max_residual,
                     DGA_min_residual
                   ), 
                   galois::loopname("PageRankSanity"),
                   galois::no_stats());

    DGA_max = current_max;
    DGA_min = current_min;
    DGA_max_residual = current_max_residual;
    DGA_min_residual = current_min_residual;

    float max_rank = DGA_max.reduce_max();
    float min_rank = DGA_min.reduce_min();
    float rank_sum = DGA_sum.reduce();
    float residual_sum = DGA_sum_residual.reduce();
    uint64_t over_tolerance = DGA_residual_over_tolerance.reduce();
    float max_residual = DGA_max_residual.reduce_max();
    float min_residual = DGA_min_residual.reduce_min();

    // Only node 0 will print data
    if (galois::runtime::getSystemNetworkInterface().ID == 0) {
      printf("Max rank is %f\n", max_rank);
      printf("Min rank is %f\n", min_rank);
      printf("Rank sum is %f\n", rank_sum);
      printf("Residual sum is %f\n", residual_sum);
      printf("# nodes with residual over tolerance is %lu\n", over_tolerance);
      printf("Max residual is %f\n", max_residual);
      printf("Min residual is %f\n", min_residual);
    }
  }
  
  /* Gets the max, min rank from all owned nodes and
   * also the sum of ranks */
  void operator()(GNode src) const {
    NodeData& sdata = graph->getData(src);

    if (graph->isOwned(graph->getGID(src))) {
      if (current_max < sdata.value) {
        current_max = sdata.value;
      }

      if (current_min > sdata.value) {
        current_min = sdata.value;
      }

      if (current_max_residual < sdata.residual) {
        current_max_residual = sdata.residual;
      }

      if (current_min_residual > sdata.residual) {
        current_min_residual = sdata.residual;
      }

      if (sdata.residual > local_tolerance) {
        DGAccumulator_residual_over_tolerance += 1;
      }

      DGAccumulator_sum += sdata.value;
      DGAccumulator_sum_residual += sdata.residual;
    }
  }
};
float PageRankSanity::current_max = 0;
float PageRankSanity::current_min = std::numeric_limits<float>::max() / 4;
float PageRankSanity::current_max_residual = 0;
float PageRankSanity::current_min_residual = std::numeric_limits<float>::max() / 4;

/******************************************************************************/
/* Main */
/******************************************************************************/

constexpr static const char* const name = "PageRank - Compiler Generated "
                                          "Distributed Heterogeneous";
constexpr static const char* const desc = "Residual PageRank on Distributed "
                                          "Galois.";
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
  Graph* hg = distGraphInitialization<NodeData, void>(&cuda_ctx);
  #else
  Graph* hg = distGraphInitialization<NodeData, void>();
  #endif

  bitset_residual.resize(hg->size());
  bitset_nout.resize(hg->size());

  galois::gPrint("[", net.ID, "] InitializeGraph::go called\n");
  galois::StatTimer StatTimer_init("TIMER_GRAPH_INIT", REGION_NAME);
  StatTimer_init.start();
    InitializeGraph::go((*hg));
  StatTimer_init.stop();
  galois::runtime::getHostBarrier().wait();

  galois::DGAccumulator<unsigned int> PageRank_accum;

  galois::DGAccumulator<float> DGA_max;
  galois::DGAccumulator<float> DGA_min;
  galois::DGAccumulator<float> DGA_sum;
  galois::DGAccumulator<float> DGA_sum_residual;
  galois::DGAccumulator<uint64_t> DGA_residual_over_tolerance;
  galois::DGAccumulator<float> DGA_max_residual;
  galois::DGAccumulator<float> DGA_min_residual;

  for (auto run = 0; run < numRuns; ++run) {
    galois::gPrint("[", net.ID, "] PageRank::go run ", run, " called\n");
    std::string timer_str("TIMER_" + std::to_string(run));
    galois::StatTimer StatTimer_main(timer_str.c_str(), REGION_NAME);

    StatTimer_main.start();
      PageRank::go(*hg, PageRank_accum);
    StatTimer_main.stop();

    // sanity check
    PageRankSanity::current_max = 0;
    PageRankSanity::current_min = std::numeric_limits<float>::max() / 4;

    PageRankSanity::current_max_residual = 0;
    PageRankSanity::current_min_residual = std::numeric_limits<float>::max() / 4;

    PageRankSanity::go(
      *hg, DGA_max, DGA_min, DGA_sum, DGA_sum_residual,
      DGA_residual_over_tolerance, DGA_max_residual, DGA_min_residual
    );

    if ((run + 1) != numRuns){
      #ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) { 
        bitset_residual_reset_cuda(cuda_ctx);
        bitset_nout_reset_cuda(cuda_ctx);
      } else
      #endif
      { bitset_residual.reset();
      bitset_nout.reset(); }

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
