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
 * Compute ConnectedComp Pull on distributed Galois.
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
#include "galois/DistAccumulator.h"
#include "galois/runtime/Tracer.h"

#ifdef __GALOIS_HET_CUDA__
#include "gen_cuda.h"
struct CUDA_Context *cuda_ctx;
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

static cll::opt<unsigned long long> src_node("srcNodeId", 
                                             cll::desc("ID of the source node"), 
                                             cll::init(0));

/******************************************************************************/
/* Graph structure declarations + other initialization */
/******************************************************************************/

struct NodeData {
  unsigned long long comp_current;
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
    		galois::StatTimer StatTimer_cuda(impl_str.c_str(), REGION_NAME);
    		StatTimer_cuda.start();

        InitializeGraph_cuda(*(allNodes.begin()), *(allNodes.end()), 
                             cuda_ctx);

    		StatTimer_cuda.stop();
    	} else if (personality == CPU)
    #endif
    {
    galois::do_all(
      galois::iterate(allNodes.begin(), allNodes.end()),
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
  }
};

struct ConnectedComp {
  Graph* graph;
  galois::DGAccumulator<unsigned int>& DGAccumulator_accum;

  ConnectedComp(Graph* _graph, galois::DGAccumulator<unsigned int>& _dga) : 
    graph(_graph), DGAccumulator_accum(_dga) {}

  void static go(Graph& _graph, galois::DGAccumulator<unsigned int>& dga) {
    unsigned _num_iterations = 1;
    
    const auto& nodesWithEdges = _graph.allNodesWithEdgesRange();
    do { 
      _graph.set_num_iter(_num_iterations);
      dga.reset();
    #ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) {
        std::string impl_str("CUDA_DO_ALL_IMPL_ConnectedComp_" + (_graph.get_run_identifier()));
        galois::StatTimer StatTimer_cuda(impl_str.c_str(), REGION_NAME);
        StatTimer_cuda.start();
        int __retval = 0;
        ConnectedComp_cuda(*nodesWithEdges.begin(), *nodesWithEdges.end(),
                           __retval, cuda_ctx);
        dga += __retval;
        StatTimer_cuda.stop();
      } else if (personality == CPU)
    #endif
      galois::do_all(
        galois::iterate(nodesWithEdges),
        ConnectedComp(&_graph, dga),
        galois::loopname(_graph.get_run_identifier("ConnectedComp").c_str()),
        galois::timeit(),
        galois::no_stats()
      );

      _graph.sync<writeSource, readDestination, Reduce_min_comp_current, 
                  Broadcast_comp_current, Bitset_comp_current>("ConnectedComp");
      
      galois::runtime::reportStat_Tsum(REGION_NAME, 
        "NUM_WORK_ITEMS_" + (_graph.get_run_identifier()), 
        (unsigned long)dga.read_local());
      ++_num_iterations;
    } while((_num_iterations < maxIterations) && dga.reduce(_graph.get_run_identifier()));

    if (galois::runtime::getSystemNetworkInterface().ID == 0) {
      galois::runtime::reportStat_Serial(REGION_NAME, 
        "NUM_ITERATIONS_" + std::to_string(_graph.get_run_num()), 
        (unsigned long)_num_iterations);
    }
  }

  void operator()(GNode src) const {
    NodeData& snode = graph->getData(src);

    for (auto jj = graph->edge_begin(src), ee = graph->edge_end(src); jj != ee; ++jj) {
      GNode dst = graph->getEdgeDst(jj);
      auto& dnode = graph->getData(dst);
      unsigned long long new_comp = dnode.comp_current;
      unsigned long long old_comp = galois::min(snode.comp_current, new_comp);
      if (old_comp > new_comp){
        bitset_comp_current.set(src);
        DGAccumulator_accum += 1;
      }
    }
  }
};

/******************************************************************************/
/* Sanity check operators */
/******************************************************************************/

/* Get/print the number of members in node 0's component */
struct SourceComponentSize {
  const unsigned long long src_comp;
  Graph* graph;

  galois::DGAccumulator<uint64_t>& DGAccumulator_accum;

  SourceComponentSize(const unsigned long long _src_comp, Graph* _graph, 
                      galois::DGAccumulator<uint64_t>& _dga) : 
    src_comp(_src_comp), graph(_graph), DGAccumulator_accum(_dga) {}

  void static go(Graph& _graph, galois::DGAccumulator<uint64_t>& dga) {
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

    galois::do_all(galois::iterate(_graph.begin(), _graph.end()), 
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

constexpr static const char* const name = "ConnectedComp Pull - Distributed "
                                          "Heterogeneous";
constexpr static const char* const desc = "ConnectedComp pull on Distributed "
                                          "Galois.";
constexpr static const char* const url = 0;

int main(int argc, char** argv) {
  galois::DistMemSys G;
  DistBenchStart(argc, argv, name, desc, url);

  auto& net = galois::runtime::getSystemNetworkInterface();

  if (net.ID == 0) {
    galois::runtime::reportParam(REGION_NAME, "Max Iterations", 
      (unsigned long)maxIterations);
  }

  galois::StatTimer StatTimer_total("TIMER_TOTAL", REGION_NAME);

  StatTimer_total.start();

  #ifdef __GALOIS_HET_CUDA__
  Graph* hg = symmetricDistGraphInitialization<NodeData, void>(&cuda_ctx);
  #else
  Graph* hg = symmetricDistGraphInitialization<NodeData, void>();
  #endif

  bitset_comp_current.resize(hg->get_local_total_nodes());

  printf("[%d] InitializeGraph::go called\n", net.ID);
  galois::StatTimer StatTimer_init("TIMER_GRAPH_INIT", REGION_NAME);
  StatTimer_init.start();
    InitializeGraph::go((*hg));
  StatTimer_init.stop();
  galois::runtime::getHostBarrier().wait();

  galois::DGAccumulator<unsigned int> DGAccumulator_accum;
  galois::DGAccumulator<uint64_t> DGAccumulator_accum64;

  for (auto run = 0; run < numRuns; ++run) {
    printf("[%d] ConnectedComp::go run %d called\n", net.ID, run);
    std::string timer_str("TIMER_" + std::to_string(run));
    galois::StatTimer StatTimer_main(timer_str.c_str(), REGION_NAME);

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

      (*hg).reset_num_iter(run + 1);
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
      for(auto ii = (*hg).begin(); ii != (*hg).end(); ++ii) {
        if ((*hg).isOwned((*hg).getGID(*ii))) 
          galois::runtime::printOutput("% %\n", (*hg).getGID(*ii), 
            (*hg).getData(*ii).comp_current);
      }
    #ifdef __GALOIS_HET_CUDA__
    } else if(personality == GPU_CUDA)  {
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
