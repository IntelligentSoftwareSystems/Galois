/** ConnectedComp -*- C++ -*-
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
 * Compute ConnectedComp on distributed Galois using worklist (filtered).
 *
 * @author Gurbinder Gill <gurbinder533@gmail.com>
 * @author Roshan Dathathri <roshan@cs.utexas.edu>
 * @author Loc Hoang <l_hoang@utexas.edu> (sanity check operators)
 */

#include <iostream>
#include <limits>
#include "galois/DistGalois.h"
#include "galois/gstl.h"
#include "DistBenchStart.h"
#include "galois/runtime/CompilerHelperFunctions.h"

#include "galois/DistAccumulator.h"
#include "galois/runtime/Tracer.h"

#ifdef __GALOIS_HET_CUDA__
#include "gen_cuda.h"
struct CUDA_Context *cuda_ctx;
#endif

static const char* const name = "ConnectedComp - Distributed Heterogeneous "
                                "with filter.";
static const char* const desc = "ConnectedComp on Distributed Galois.";
static const char* const url = 0;

/******************************************************************************/
/* Declaration of command line arguments */
/******************************************************************************/

namespace cll = llvm::cl;
static cll::opt<unsigned int> maxIterations("maxIterations", 
                                            cll::desc("Maximum iterations: "
                                                      "Default 1000"), 
                                            cll::init(1000));
static cll::opt<unsigned long long> src_node("srcNodeId", 
                                             cll::desc("ID of the source node"), 
                                             cll::init(0));

/******************************************************************************/
/* Graph structure declarations + other initialization */
/******************************************************************************/

struct NodeData {
  std::atomic<unsigned long long> comp_current;
  uint32_t comp_old;
};

galois::DynamicBitSet bitset_comp_current;

typedef hGraph<NodeData, void> Graph;
typedef typename Graph::GraphNode GNode;

#include "gen_sync.hh"

/******************************************************************************/
/* Algorithm structures */
/******************************************************************************/

struct InitializeGraph {
  Graph *graph;

  InitializeGraph(Graph* _graph) : graph(_graph){}

  void static go(Graph& _graph) {
    const auto& allNodes = _graph.allNodesRange();

    #ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) {
        std::string impl_str("CUDA_DO_ALL_IMPL_InitializeGraph_" + 
                             (_graph.get_run_identifier()));
        galois::StatTimer StatTimer_cuda(impl_str.c_str());
        StatTimer_cuda.start();
        InitializeGraph_cuda(*(allNodes.begin()), *(allNodes.end()), 
                                 cuda_ctx);
        StatTimer_cuda.stop();
      } else if (personality == CPU)
    #endif
    {
    galois::do_all_local(
      allNodes,
      InitializeGraph{&_graph}, 
      galois::loopname(_graph.get_run_identifier("InitializeGraph").c_str()),
      galois::timeit(),
      galois::no_stats()
    );
    }
  }

  void operator()(GNode src) const {
    NodeData& sdata = graph->getData(src);
    sdata.comp_current = graph->getGID(src);
    sdata.comp_old = graph->getGID(src);
  }
};

struct FirstItr_ConnectedComp{
  Graph * graph;
  FirstItr_ConnectedComp(Graph * _graph):graph(_graph){}

  void static go(Graph& _graph) {
    const auto& nodesWithEdges = _graph.allNodesWithEdgesRange();
#ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      std::string impl_str("CUDA_DO_ALL_IMPL_ConnectedComp_" + 
                             (_graph.get_run_identifier()));
      galois::StatTimer StatTimer_cuda(impl_str.c_str());
      StatTimer_cuda.start();
      FirstItr_ConnectedComp_cuda(*nodesWithEdges.begin(), 
                                  *nodesWithEdges.end(), cuda_ctx);
      StatTimer_cuda.stop();
    } else if (personality == CPU)
#endif
    {
      galois::do_all_local(
        nodesWithEdges,
        FirstItr_ConnectedComp{ &_graph },
        galois::loopname(_graph.get_run_identifier("ConnectedComp").c_str()),
        galois::timeit(),
        galois::no_stats()
      );
    }

    _graph.sync<writeDestination, readSource, Reduce_min_comp_current, 
                Broadcast_comp_current, Bitset_comp_current>("ConnectedComp");
  
    galois::runtime::reportStat_Tsum("ConnectedComp", 
      "NUM_WORK_ITEMS_" + (_graph.get_run_identifier()), 
      _graph.end() - _graph.begin());
  }

  void operator()(GNode src) const {
    NodeData& snode = graph->getData(src);
    snode.comp_old = snode.comp_current;

    for (auto jj = graph->edge_begin(src), ee = graph->edge_end(src); jj != ee; ++jj) {
      GNode dst = graph->getEdgeDst(jj);
      auto& dnode = graph->getData(dst);
      unsigned long long new_dist = snode.comp_current;
      unsigned long long old_dist = galois::atomicMin(dnode.comp_current, new_dist);
      if (old_dist > new_dist) bitset_comp_current.set(dst);
    }
  }

};

struct ConnectedComp {
  Graph* graph;
  galois::DGAccumulator<unsigned int>& DGAccumulator_accum;

  ConnectedComp(Graph* _graph, galois::DGAccumulator<unsigned int>& _dga) : 
    graph(_graph), DGAccumulator_accum(_dga) {}

  void static go(Graph& _graph, galois::DGAccumulator<unsigned int>& dga) {
    using namespace galois::worklists;

    FirstItr_ConnectedComp::go(_graph);
    
    unsigned _num_iterations = 1;
    
    const auto& nodesWithEdges = _graph.allNodesWithEdgesRange();

    do { 
      _graph.set_num_iter(_num_iterations);
      dga.reset();
    #ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) {
        std::string impl_str("CUDA_DO_ALL_IMPL_ConnectedComp_" + 
                             (_graph.get_run_identifier()));
        galois::StatTimer StatTimer_cuda(impl_str.c_str());
        StatTimer_cuda.start();
        int __retval = 0;
        ConnectedComp_cuda(*nodesWithEdges.begin(), *nodesWithEdges.end(),
                           __retval, cuda_ctx);
        dga += __retval;
        StatTimer_cuda.stop();
      } else if (personality == CPU)
    #endif
      {
      galois::do_all_local(
        nodesWithEdges,
        ConnectedComp(&_graph, dga),
        galois::loopname(_graph.get_run_identifier("ConnectedComp").c_str()),
        galois::timeit(),
        galois::no_stats()
      );
      }

      _graph.sync<writeDestination, readSource, Reduce_min_comp_current, 
                  Broadcast_comp_current, Bitset_comp_current>("ConnectedComp");
      
      galois::runtime::reportStat_Tsum("ConnectedComp", 
        "NUM_WORK_ITEMS_" + (_graph.get_run_identifier()), 
        (unsigned long)dga.read_local());
      ++_num_iterations;
    } while((_num_iterations < maxIterations) && dga.reduce(_graph.get_run_identifier()));

    if (galois::runtime::getSystemNetworkInterface().ID == 0) {
      galois::runtime::reportStat_Serial("ConnectedComp", 
        "NUM_ITERATIONS_" + std::to_string(_graph.get_run_num()), 
        (unsigned long)_num_iterations);
    }
  }

  void operator()(GNode src) const {
    NodeData& snode = graph->getData(src);

    if (snode.comp_old > snode.comp_current) {
      snode.comp_old = snode.comp_current;

      for (auto jj = graph->edge_begin(src), ee = graph->edge_end(src); 
           jj != ee; ++jj) {
        GNode dst = graph->getEdgeDst(jj);
        auto& dnode = graph->getData(dst);
        unsigned long long new_dist = snode.comp_current;
        unsigned long long old_dist = galois::atomicMin(dnode.comp_current, new_dist);
        if (old_dist > new_dist) bitset_comp_current.set(dst);
      }

      DGAccumulator_accum+= 1;
    }
  }
};

/******************************************************************************/
/* Sanity check operators */
/******************************************************************************/

/* Get/print the number of members in source node component */
struct SourceComponentSize {
  const unsigned long long src_comp;
  Graph* graph;

  galois::DGAccumulator<unsigned long long>& DGAccumulator_accum;

  SourceComponentSize(const uint64_t _src_comp, Graph* _graph, 
                      galois::DGAccumulator<unsigned long long>& _dga) : 
    src_comp(_src_comp), graph(_graph), DGAccumulator_accum(_dga) {}

  void static go(Graph& _graph, galois::DGAccumulator<unsigned long long>& dga) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      // TODO currently no GPU support for sanity check operator
      fprintf(stderr, "Warning: No GPU support for sanity check; might get "
                      "wrong results.\n");
    }
  #endif

    dga.reset();

    if (_graph.isOwned(src_node)) {
      dga += _graph.getData(_graph.getLID(src_node)).comp_current;
    }

    uint64_t src_comp = dga.reduce();

    dga.reset();

    galois::do_all(_graph.begin(), _graph.end(), 
                   SourceComponentSize(src_comp, &_graph, dga), 
                   galois::loopname("SourceComponentSize"),
                   galois::no_stats());

    uint64_t num_in_component = dga.reduce();

    // Only node 0 will print the number visited
    if (_graph.id == 0) {
      printf("Number of nodes in node %lu's component is %lu\n", (uint64_t)src_node, 
             num_in_component);
    }
  }

  /* Check if an owned node's component is the same as src's.
   * if yes, then increment an accumulator */
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    if (graph->isOwned(graph->getGID(src)) && 
        src_data.comp_current == src_comp) {
      DGAccumulator_accum += 1;
    }
  }
};

/******************************************************************************/
/* Main */
/******************************************************************************/

int main(int argc, char** argv) {
  galois::DistMemSys G;
  DistBenchStart(argc, argv, name, desc, url);

  auto& net = galois::runtime::getSystemNetworkInterface();

  if (net.ID == 0) {
    galois::runtime::reportParam("ConnectedComponents", "Max Iterations", 
      (unsigned long)maxIterations);
  }

  galois::StatTimer StatTimer_total("TIMER_TOTAL");

  StatTimer_total.start();

  #ifdef __GALOIS_HET_CUDA__
  Graph* hg = symmetricDistGraphInitialization<NodeData, void>(&cuda_ctx);
  #else
  Graph* hg = symmetricDistGraphInitialization<NodeData, void>();
  #endif

  bitset_comp_current.resize(hg->get_local_total_nodes());

  std::cout << "[" << net.ID << "] InitializeGraph::go called\n";

  galois::StatTimer StatTimer_init("TIMER_GRAPH_INIT");
  StatTimer_init.start();
    InitializeGraph::go((*hg));
  StatTimer_init.stop();
  galois::runtime::getHostBarrier().wait();

  galois::DGAccumulator<unsigned int> DGAccumulator_accum;
  galois::DGAccumulator<unsigned long long> DGAccumulator_accum64;

  for (auto run = 0; run < numRuns; ++run) {
    std::cout << "[" << net.ID << "] ConnectedComp::go run " << run << " called\n";
    std::string timer_str("TIMER_" + std::to_string(run));
    galois::StatTimer StatTimer_main(timer_str.c_str());

    StatTimer_main.start();
      ConnectedComp::go(*hg, DGAccumulator_accum);
    StatTimer_main.stop();

    SourceComponentSize::go(*hg, DGAccumulator_accum64);

    if ((run + 1) != numRuns) {
      #ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) { 
        bitset_comp_current_reset_cuda(cuda_ctx);
      } else
      #endif
      bitset_comp_current.reset();

      (*hg).reset_num_iter(run+1);
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
      for (auto ii = (*hg).begin(); ii != (*hg).end(); ++ii) {
        if ((*hg).isOwned((*hg).getGID(*ii))) 
          galois::runtime::printOutput("% %\n", (*hg).getGID(*ii), 
            (*hg).getData(*ii).comp_current);
      }
    #ifdef __GALOIS_HET_CUDA__
    } else if (personality == GPU_CUDA)  {
      for(auto ii = (*hg).begin(); ii != (*hg).end(); ++ii) {
        if ((*hg).isOwned((*hg).getGID(*ii))) 
          galois::runtime::printOutput("% %\n", (*hg).getGID(*ii), 
            get_node_comp_current_cuda(cuda_ctx, *ii));
      }
    }
    #endif
  }

  return 0;
}
