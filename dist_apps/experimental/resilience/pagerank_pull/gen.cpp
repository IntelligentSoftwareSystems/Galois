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
 * Compute pageRank Pull version using residual on distributed Galois.
 *
 * @author Gurbinder Gill <gurbinder533@gmail.com>
 * @author Roshan Dathathri <roshan@cs.utexas.edu>
 * @author Loc Hoang <l_hoang@utexas.edu> (sanity check operators)
 */

#include "galois/DistGalois.h"
#include "DistBenchStart.h"
#include "galois/gstl.h"
#include "galois/runtime/Tracer.h"
#include "galois/DReducible.h"

//For resilience
#include "resilience.h"

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
  float residual;
  float delta;
};

galois::DynamicBitSet bitset_residual;
galois::DynamicBitSet bitset_nout;

typedef DistGraph<NodeData, void> Graph;
typedef typename Graph::GraphNode GNode;

#include "gen_sync.hh"

/******************************************************************************/
/* Algorithm structures */
/******************************************************************************/

/* (Re)initialize all fields to 0 except for residual which needs to be 0.15
 * everywhere */
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
      ResetGraph_cuda(*allNodes.begin(), *allNodes.end(), alpha, cuda_ctx);
      StatTimer_cuda.stop();
    } else if (personality == CPU)
    #endif
    galois::do_all(
      galois::iterate(allNodes.begin(), allNodes.end()),
      ResetGraph{ alpha, &_graph },
      galois::no_stats(), 
      galois::loopname(_graph.get_run_identifier("ResetGraph").c_str()));
  }

  void operator()(GNode src) const {
    auto& sdata = graph->getData(src);
    sdata.value = 0;
    sdata.nout = 0;
    sdata.delta = 0;
    sdata.residual = local_alpha;
  }
};

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
      InitializeGraph_cuda(*nodesWithEdges.begin(), *nodesWithEdges.end(), 
                           cuda_ctx);
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

/* (Re)initialize all fields to 0 except for value which needs to be 0.15
 * on crashed hosts */
struct ResetGraph_crashed {
  const float& local_alpha;
  Graph* graph;

  ResetGraph_crashed(const float& _local_alpha, Graph* _graph) : 
      local_alpha(_local_alpha), graph(_graph) {}

  void static go(Graph& _graph) {
    const auto& allNodes = _graph.allNodesRange();
    //TODO: not correct for CUDA
    #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      std::string impl_str("CUDA_DO_ALL_IMPL_ResetGraph_" + (_graph.get_run_identifier()));
      galois::StatTimer StatTimer_cuda(impl_str.c_str(), REGION_NAME);
      StatTimer_cuda.start();
      ResetGraph_cuda(*allNodes.begin(), *allNodes.end(), alpha, cuda_ctx);
      StatTimer_cuda.stop();
    } else if (personality == CPU)
    #endif
    galois::do_all(
      galois::iterate(allNodes.begin(), allNodes.end()),
      ResetGraph_crashed{ alpha, &_graph },
      galois::no_stats(),
      galois::loopname(_graph.get_run_identifier("ResetGraph_crashed").c_str()));
  }

  void operator()(GNode src) const {
    auto& sdata = graph->getData(src);
    sdata.value = local_alpha;
    sdata.nout = 0;
    sdata.delta = 0;
    sdata.residual = 0;
  }
};

struct InitializeGraph_crashed {
  Graph* graph;

  InitializeGraph_crashed(Graph* _graph) : graph(_graph) {}

  void static go(Graph& _graph) {
    // init graph
    ResetGraph_crashed::go(_graph);

    const auto& nodesWithEdges = _graph.allNodesWithEdgesRange();

    //TODO:: not correct for CUDA
    #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      std::string impl_str("CUDA_DO_ALL_IMPL_InitializeGraph_" + 
        (_graph.get_run_identifier()));
      galois::StatTimer StatTimer_cuda(impl_str.c_str(), REGION_NAME);
      StatTimer_cuda.start();
      InitializeGraph_cuda(*nodesWithEdges.begin(), *nodesWithEdges.end(), 
                           cuda_ctx);
      StatTimer_cuda.stop();
    } else if (personality == CPU)
    #endif
    // doing a local do all because we are looping over edges
    galois::do_all(
      galois::iterate(nodesWithEdges),
      InitializeGraph_crashed{ &_graph },
      galois::steal(), 
      galois::no_stats(), 
      galois::loopname(_graph.get_run_identifier("InitializeGraph_crashed").c_str()));

    _graph.sync<writeDestination, readAny, Reduce_add_nout, Broadcast_nout,
                Bitset_nout>("InitializeGraph_crashed");
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

/* (Re)initialize only nout. Required only for recovery
 * on healthy hosts */
struct ResetGraph_healthy {
  Graph* graph;

  ResetGraph_healthy(Graph* _graph) : graph(_graph) {}

  void static go(Graph& _graph) {
    const auto& allNodes = _graph.allNodesRange();
    #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      std::string impl_str("CUDA_DO_ALL_IMPL_ResetGraph_" + (_graph.get_run_identifier()));
      galois::StatTimer StatTimer_cuda(impl_str.c_str(), REGION_NAME);
      StatTimer_cuda.start();
      ResetGraph_cuda(*allNodes.begin(), *allNodes.end(), alpha, cuda_ctx);
      StatTimer_cuda.stop();
    } else if (personality == CPU)
    #endif
    galois::do_all(
      galois::iterate(allNodes.begin(), allNodes.end()),
      ResetGraph_healthy{ &_graph },
      galois::no_stats(),
      galois::loopname(_graph.get_run_identifier("ResetGraph_healthy").c_str()));
  }

  void operator()(GNode src) const {
    auto& sdata = graph->getData(src);
    //Reset residual on healthy nodes.
    sdata.residual = 0;
    sdata.nout = 0;
  }
};


struct InitializeGraph_healthy {
  Graph* graph;

  InitializeGraph_healthy(Graph* _graph) : graph(_graph) {}

  void static go(Graph& _graph) {
    // Reset graph field on healthy nodes.
    ResetGraph_healthy::go(_graph);

    const auto& nodesWithEdges = _graph.allNodesWithEdgesRange();

    #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      std::string impl_str("CUDA_DO_ALL_IMPL_InitializeGraph_" + 
        (_graph.get_run_identifier()));
      galois::StatTimer StatTimer_cuda(impl_str.c_str(), REGION_NAME);
      StatTimer_cuda.start();
      InitializeGraph_cuda(*nodesWithEdges.begin(), *nodesWithEdges.end(), 
                           cuda_ctx);
      StatTimer_cuda.stop();
    } else if (personality == CPU)
    #endif
    // doing a local do all because we are looping over edges
    galois::do_all(
      galois::iterate(nodesWithEdges),
      InitializeGraph_healthy{ &_graph },
      galois::steal(), 
      galois::no_stats(), 
      galois::loopname(_graph.get_run_identifier("InitializeGraph_healthy").c_str()));

    _graph.sync<writeDestination, readAny, Reduce_add_nout, Broadcast_nout,
                Bitset_nout>("InitializeGraph_healthy");
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

/* Recovery to be called by resilience based fault tolerance
 */
struct recovery {
  Graph * graph;

  recovery(Graph * _graph) : graph(_graph) {}

  void static go(Graph& _graph) {

    const auto& nodesWithEdges = _graph.allNodesWithEdgesRange();
    //const auto& allNodes = _graph.allNodesRange();
    galois::do_all(
      galois::iterate(nodesWithEdges),
      //galois::iterate(allNodes.begin(), allNodes.end()),
      recovery{&_graph},
      galois::no_stats(),
      galois::loopname(_graph.get_run_identifier("RECOVERY").c_str()));
  }

  void operator()(GNode src) const {
    NodeData& sdata = graph->getData(src);
    if(sdata.nout > 0)
      sdata.delta = sdata.value * (1 - alpha) / sdata.nout;
  }
};

/* Recovery Adjust to be called by resilience based fault tolerance
 */
struct recoveryAdjust {
  Graph * graph;

  recoveryAdjust(Graph * _graph) : graph(_graph) {}

  void static go(Graph& _graph) {

    const auto& nodesWithEdges = _graph.allNodesWithEdgesRange();
    //const auto& allNodes = _graph.allNodesRange();
    galois::do_all(
      //galois::iterate(allNodes.begin(), allNodes.end()),
      galois::iterate(nodesWithEdges),
      recoveryAdjust{&_graph},
      galois::no_stats(),
      galois::loopname(_graph.get_run_identifier("RECOVERY").c_str()));

      //TODO: Is this required??
      //_graph.sync<writeSource, readAny, Reduce_add_residual, Broadcast_residual>("RECOVERY");
  }

  void operator()(GNode src) const {
    NodeData& sdata = graph->getData(src);
    sdata.residual -= (sdata.value - alpha);
  }
};

struct PageRank_delta {
  const float & local_alpha;
  cll::opt<float> & local_tolerance;
  Graph* graph;

  galois::DGAccumulator<unsigned int>& DGAccumulator_accum;

  PageRank_delta(const float & _local_alpha, cll::opt<float> & _local_tolerance,
                 Graph* _graph, galois::DGAccumulator<unsigned int>& _dga) : 
      local_alpha(_local_alpha),
      local_tolerance(_local_tolerance),
      graph(_graph),
      DGAccumulator_accum(_dga) {}

  void static go(Graph& _graph, galois::DGAccumulator<unsigned int>& dga) {
    const auto& allNodes = _graph.allNodesRange();

    #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      std::string impl_str("CUDA_DO_ALL_IMPL_PageRank_" +
        (_graph.get_run_identifier()));
      galois::StatTimer StatTimer_cuda(impl_str.c_str(), REGION_NAME);
      StatTimer_cuda.start();
      int __retval = 0;
      PageRank_delta_cuda(*allNodes.begin(), *allNodes.end(),
                          __retval, alpha, tolerance, cuda_ctx);
      dga += __retval;
      StatTimer_cuda.stop();
    } else if (personality == CPU)
    #endif
    galois::do_all(
      galois::iterate(allNodes.begin(), allNodes.end()),
      PageRank_delta{ alpha, tolerance, &_graph, dga },
      galois::no_stats(),
      galois::loopname(_graph.get_run_identifier("PageRank_delta").c_str()));
  }

  void operator()(GNode src) const {
    auto& sdata = graph->getData(src);
    sdata.delta = 0;

    if (sdata.residual > this->local_tolerance) {
      sdata.value += sdata.residual;
      if (sdata.nout > 0) {
        sdata.delta = sdata.residual * (1 - local_alpha) / sdata.nout;
        DGAccumulator_accum += 1;
      }
      sdata.residual = 0;
    }
  }
};

// TODO: GPU code operator does not match CPU's operator (cpu accumulates sum
// and adds all at once, GPU adds each pulled value individually/atomically)
struct PageRank {
  Graph* graph;

  PageRank(Graph* _graph) : graph(_graph) {}

  void static go(Graph& _graph, galois::DGAccumulator<unsigned int>& dga) {
    unsigned _num_iterations = 0;
    const auto& nodesWithEdges = _graph.allNodesWithEdgesRange();

    //unsigned int reduced = 0;

    do {

       /**************************CRASH SITE : ADJUST : start *****************************************/
      //if(enableFT && (_num_iterations == crashIteration + 1)){
        //galois::gPrint("CALLING crashSiteAdjust\n");
        //crashSiteAdjust<recoveryAdjust>(_graph);
      //}
      /**************************CRASH SITE : ADJUST : end *****************************************/

      //Checkpointing the all the node data
      if(enableFT && recoveryScheme == CP){
        saveCheckpointToDisk(_num_iterations, _graph);
      }

      _graph.set_num_iter(_num_iterations);
      dga.reset();
      PageRank_delta::go(_graph, dga);
      _graph.reset_mirrorField<Reduce_add_residual>();

      #ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) {
        std::string impl_str("CUDA_DO_ALL_IMPL_PageRank_" + 
          (_graph.get_run_identifier()));
        galois::StatTimer StatTimer_cuda(impl_str.c_str(), REGION_NAME);
        StatTimer_cuda.start();
        PageRank_cuda(*nodesWithEdges.begin(), *nodesWithEdges.end(), cuda_ctx);
        StatTimer_cuda.stop();
      } else if (personality == CPU)
      #endif
      galois::do_all(
        galois::iterate(nodesWithEdges),
        PageRank{ &_graph },
        galois::steal(),
        galois::no_stats(),
        galois::loopname(_graph.get_run_identifier("PageRank").c_str()));

      _graph.sync<writeSource, readAny, Reduce_add_residual, Broadcast_residual,
                  Bitset_residual>("PageRank");

       /**************************CRASH SITE : start *****************************************/
      if(enableFT && (_num_iterations == crashIteration)){
        crashSite<recovery, InitializeGraph_crashed, InitializeGraph_healthy>(_graph);
        const auto& net = galois::runtime::getSystemNetworkInterface();
        galois::gPrint(net.ID, " : recovery DONE!!!\n");

        _graph.reset_mirrorField<Reduce_add_residual>();
        galois::do_all(
            galois::iterate(nodesWithEdges),
            PageRank{ &_graph },
            galois::steal(),
            galois::no_stats(),
            galois::loopname(_graph.get_run_identifier("PageRank-afterCrash").c_str()));

        //_graph.sync<writeSource, readAny, Reduce_add_residual, Broadcast_residual,
          //Bitset_residual>("PageRank-afterCrash");

        _graph.sync<writeSource, readAny, Reduce_add_residual, Broadcast_residual>("PageRank-afterCrash");

        crashSiteAdjust<recoveryAdjust>(_graph);

        // Do all and sync
        //_graph.reset_mirrorField<Reduce_add_residual>();
        // Adjust delta
      }
      /**************************CRASH SITE : end *****************************************/

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

  // Pull deltas from neighbor nodes, then add to self-residual
  void operator()(GNode src) const {
    auto& sdata = graph->getData(src);

    for (auto nbr : graph->edges(src)) {
      GNode dst = graph->getEdgeDst(nbr);
      auto& ddata = graph->getData(dst);

      if (ddata.delta > 0) {
        galois::add(sdata.residual, ddata.delta);

        bitset_residual.set(src);
      }
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

  galois::DGAccumulator<float>& DGAccumulator_max;
  galois::DGAccumulator<float>& DGAccumulator_min;
  galois::DGAccumulator<float>& DGAccumulator_sum;
  galois::DGAccumulator<float>& DGAccumulator_sum_residual;
  galois::DGAccumulator<uint64_t>& DGAccumulator_residual_over_tolerance;
  galois::DGAccumulator<float>& DGAccumulator_max_residual;
  galois::DGAccumulator<float>& DGAccumulator_min_residual;

  galois::GReduceMax<float>& max_value;
  galois::GReduceMin<float>& min_value;
  galois::GReduceMax<float>& max_residual;
  galois::GReduceMin<float>& min_residual;

  PageRankSanity(cll::opt<float>& _local_tolerance, Graph* _graph,
      galois::DGAccumulator<float>& _DGAccumulator_max,
      galois::DGAccumulator<float>& _DGAccumulator_min,
      galois::DGAccumulator<float>& _DGAccumulator_sum,
      galois::DGAccumulator<float>& _DGAccumulator_sum_residual,
      galois::DGAccumulator<uint64_t>& _DGAccumulator_residual_over_tolerance,
      galois::DGAccumulator<float>& _DGAccumulator_max_residual,
      galois::DGAccumulator<float>& _DGAccumulator_min_residual,
      galois::GReduceMax<float>& _max_value,
      galois::GReduceMin<float>& _min_value,
      galois::GReduceMax<float>& _max_residual,
      galois::GReduceMin<float>& _min_residual
  ) : 
    local_tolerance(_local_tolerance), graph(_graph),
    DGAccumulator_max(_DGAccumulator_max),
    DGAccumulator_min(_DGAccumulator_min),
    DGAccumulator_sum(_DGAccumulator_sum),
    DGAccumulator_sum_residual(_DGAccumulator_sum_residual),
    DGAccumulator_residual_over_tolerance(_DGAccumulator_residual_over_tolerance),
    DGAccumulator_max_residual(_DGAccumulator_max_residual),
    DGAccumulator_min_residual(_DGAccumulator_min_residual),
    max_value(_max_value),
    min_value(_min_value),
    max_residual(_max_residual),
    min_residual(_min_residual) {}

  void static go(Graph& _graph,
    galois::DGAccumulator<float>& DGA_max,
    galois::DGAccumulator<float>& DGA_min,
    galois::DGAccumulator<float>& DGA_sum,
    galois::DGAccumulator<float>& DGA_sum_residual,
    galois::DGAccumulator<uint64_t>& DGA_residual_over_tolerance,
    galois::DGAccumulator<float>& DGA_max_residual,
    galois::DGAccumulator<float>& DGA_min_residual,
    galois::GReduceMax<float>& max_value,
    galois::GReduceMin<float>& min_value,
    galois::GReduceMax<float>& max_residual,
    galois::GReduceMin<float>& min_residual
  ) {
    DGA_max.reset();
    DGA_min.reset();
    DGA_sum.reset();
    DGA_sum_residual.reset();
    DGA_residual_over_tolerance.reset();
    DGA_max_residual.reset();
    DGA_min_residual.reset();

  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      float _max_value;
      float _min_value;
      float _sum_value;
      float _sum_residual;
      uint32_t num_residual_over_tolerance;
      float _max_residual;
      float _min_residual;
      PageRankSanityCheck_cuda(_max_value, _min_value, _sum_value, _sum_residual, num_residual_over_tolerance, _max_residual, _min_residual, tolerance, cuda_ctx);
      DGA_max = _max_value;
      DGA_min = _min_value;
      DGA_sum += _sum_value;
      DGA_sum_residual += _sum_residual;
      DGA_residual_over_tolerance += num_residual_over_tolerance;
      DGA_max_residual = _max_residual;
      DGA_min_residual = _min_residual;
    }
    else
  #endif
    {
      max_value.reset();
      min_value.reset();
      max_residual.reset();
      min_residual.reset();
      galois::do_all(galois::iterate(_graph.masterNodesRange().begin(), 
                                     _graph.masterNodesRange().end()), 
                     PageRankSanity(
                       tolerance, 
                       &_graph,
                       DGA_max,
                       DGA_min,
                       DGA_sum,
                       DGA_sum_residual,
                       DGA_residual_over_tolerance,
                       DGA_max_residual,
                       DGA_min_residual,
                       max_value,
                       min_value,
                       max_residual,
                       min_residual
                     ), 
                     galois::no_stats(), galois::loopname("PageRankSanity"));

      DGA_max = max_value.reduce();
      DGA_min = min_value.reduce();
      DGA_max_residual = max_residual.reduce();
      DGA_min_residual = min_residual.reduce();
    }

    float max_rank = DGA_max.reduce_max();
    float min_rank = DGA_min.reduce_min();
    float rank_sum = DGA_sum.reduce();
    float residual_sum = DGA_sum_residual.reduce();
    uint64_t over_tolerance = DGA_residual_over_tolerance.reduce();
    float max_res = DGA_max_residual.reduce_max();
    float min_res = DGA_min_residual.reduce_min();

    // Only node 0 will print data
    if (galois::runtime::getSystemNetworkInterface().ID == 0) {
      galois::gPrint("Max rank is ", max_rank, "\n");
      galois::gPrint("Min rank is ", min_rank, "\n");
      galois::gPrint("Rank sum is ", rank_sum, "\n");
      galois::gPrint("Residual sum is ", residual_sum, "\n");
      galois::gPrint("# nodes with residual over ", tolerance, " (tolerance) is ", over_tolerance, "\n");
      galois::gPrint("Max residual is ", max_res, "\n");
      galois::gPrint("Min residual is ", min_res, "\n");
    }
  }
  
  /* Gets the max, min rank from all owned nodes and
   * also the sum of ranks */
  void operator()(GNode src) const {
    NodeData& sdata = graph->getData(src);

    max_value.update(sdata.value);
    min_value.update(sdata.value);
    max_residual.update(sdata.residual);
    min_residual.update(sdata.residual);

    DGAccumulator_sum += sdata.value;
    DGAccumulator_sum_residual += sdata.residual;

    if (sdata.residual > local_tolerance) {
      DGAccumulator_residual_over_tolerance += 1;
    }
  }
};

/******************************************************************************/
/* Main */
/******************************************************************************/

constexpr static const char* const name = "PageRank - Compiler Generated "
                                          "Distributed Heterogeneous";
constexpr static const char* const desc = "PageRank Residual Pull version on "
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

  bitset_residual.resize(hg->size());
  bitset_nout.resize(hg->size());

  galois::gPrint("[", net.ID, "] InitializeGraph::go called\n");
  galois::StatTimer StatTimer_init("TIMER_GRAPH_INIT", REGION_NAME);
  StatTimer_init.start();
    InitializeGraph::go(*hg);
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
  galois::GReduceMax<float> max_value;
  galois::GReduceMin<float> min_value;
  galois::GReduceMax<float> max_residual;
  galois::GReduceMin<float> min_residual;

  for (auto run = 0; run < numRuns; ++run) {
    galois::gPrint("[", net.ID, "] PageRank::go run ", run, " called\n");
    std::string timer_str("TIMER_" + std::to_string(run));
    galois::StatTimer StatTimer_main(timer_str.c_str(), REGION_NAME);

    StatTimer_main.start();
    PageRank::go(*hg, PageRank_accum);
    StatTimer_main.stop();

    // sanity check
    PageRankSanity::go(
      *hg, DGA_max, DGA_min, DGA_sum, DGA_sum_residual,
      DGA_residual_over_tolerance, DGA_max_residual, DGA_min_residual,
      max_value, min_value, max_residual, min_residual
    );

    if ((run + 1) != numRuns) {
      #ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) { 
        bitset_residual_reset_cuda(cuda_ctx);
        bitset_nout_reset_cuda(cuda_ctx);
      } else
      #endif
      { 
        bitset_residual.reset();
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
