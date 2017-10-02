/** BFS -*- C++ -*-
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
 * Compute BFS pull on distributed Galois.
 *
 * @author Gurbinder Gill <gurbinder533@gmail.com>
 * @author Roshan Dathathri <roshan@cs.utexas.edu>
 * @author Loc Hoang <l_hoang@utexas.edu>
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

constexpr static const char* const regionname = "BFS";

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

const uint32_t infinity = std::numeric_limits<uint32_t>::max()/4;

struct NodeData {
  uint32_t dist_current;
};

typedef hGraph<NodeData, void> Graph;
typedef typename Graph::GraphNode GNode;

galois::DynamicBitSet bitset_dist_current;

#include "gen_sync.hh"

/******************************************************************************/
/* Algorithm structures */
/******************************************************************************/

struct InitializeGraph {
  const uint32_t &local_infinity;
  cll::opt<unsigned long long> &local_src_node;
  Graph *graph;

  InitializeGraph(cll::opt<unsigned long long> &_src_node, 
                  const uint32_t &_infinity, Graph* _graph) : 
                    local_infinity(_infinity), local_src_node(_src_node), 
                    graph(_graph){}

  void static go(Graph& _graph){
    const auto& allNodes = _graph.allNodesRange();
    #ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) {
        std::string impl_str("CUDA_DO_ALL_IMPL_InitializeGraph_" + 
                             (_graph.get_run_identifier()));
        galois::StatTimer StatTimer_cuda(impl_str.c_str(), regionname);
        StatTimer_cuda.start();
        InitializeGraph_cuda(*(allNodes.begin()), *(allNodes.end()),
                             infinity, src_node, cuda_ctx);
        StatTimer_cuda.stop();
      } else if (personality == CPU)
    #endif
    {
    galois::do_all(
      galois::iterate(allNodes),
      InitializeGraph(src_node, infinity, &_graph),
      galois::loopname(_graph.get_run_identifier("InitializeGraph").c_str()),
      galois::timeit(),
      galois::no_stats()
    );
    }
  }

  void operator()(GNode src) const {
    NodeData& sdata = graph->getData(src);
    sdata.dist_current = (graph->getGID(src) == local_src_node) ? 0 : local_infinity;
  }
};

struct BFS {
  Graph* graph;
  galois::DGAccumulator<unsigned int>& DGAccumulator_accum;

  BFS(Graph* _graph, galois::DGAccumulator<unsigned int>& _dga) : 
    graph(_graph), DGAccumulator_accum(_dga) {}

  void static go(Graph& _graph, galois::DGAccumulator<unsigned int>& dga) {
    unsigned _num_iterations = 0;

    const auto& nodesWithEdges = _graph.allNodesWithEdgesRange();
    do {
      _graph.set_num_iter(_num_iterations);
      dga.reset();
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          std::string impl_str("CUDA_DO_ALL_IMPL_BFS_" + (_graph.get_run_identifier()));
          galois::StatTimer StatTimer_cuda(impl_str.c_str(), regionname);
          StatTimer_cuda.start();
          int __retval = 0;
          BFS_cuda(*nodesWithEdges.begin(), *nodesWithEdges.end(),
                   __retval, cuda_ctx);
          dga += __retval;
          StatTimer_cuda.stop();
        } else if (personality == CPU)
      #endif
      {
      galois::do_all(
        galois::iterate(nodesWithEdges),
        BFS(&_graph, dga),
        galois::loopname(_graph.get_run_identifier("BFS").c_str()),
        galois::timeit(),
        galois::no_stats()
      );

      }
      _graph.sync<writeSource, readDestination, Reduce_min_dist_current, 
                  Broadcast_dist_current, Bitset_dist_current>("BFS");

      galois::runtime::reportStat_Tsum(regionname,
        _graph.get_run_identifier("NUM_WORK_ITEMS"), 
        (unsigned long)dga.read_local());
      ++_num_iterations;
    } while ((_num_iterations < maxIterations) && dga.reduce(_graph.get_run_identifier()));

    if (galois::runtime::getSystemNetworkInterface().ID == 0) {
      galois::runtime::reportStat_Serial(regionname, 
        "NUM_ITERATIONS_" + std::to_string(_graph.get_run_num()), 
        (unsigned long)_num_iterations);
    }
  }

  void operator()(GNode src) const {
    NodeData& snode = graph->getData(src);

    for (auto jj = graph->edge_begin(src), ee = graph->edge_end(src); jj != ee; ++jj) {
      GNode dst = graph->getEdgeDst(jj);
      auto& dnode = graph->getData(dst);
      uint32_t new_dist = dnode.dist_current + 1;
      uint32_t old_dist = galois::min(snode.dist_current, new_dist);
      if (old_dist > new_dist){
        bitset_dist_current.set(src);
        DGAccumulator_accum += 1;
      }
    }
  }
};

/******************************************************************************/
/* Sanity check operators */
/******************************************************************************/

/* Prints total number of nodes visited + max distance */
struct BFSSanityCheck {
  const uint32_t &local_infinity;
  Graph* graph;

  static uint32_t current_max;

  galois::DGAccumulator<uint64_t>& DGAccumulator_sum;
  galois::DGAccumulator<uint32_t>& DGAccumulator_max;

  BFSSanityCheck(const uint32_t& _infinity, Graph* _graph, 
                 galois::DGAccumulator<uint64_t>& dgas,
                 galois::DGAccumulator<uint32_t>& dgam) : 
    local_infinity(_infinity), graph(_graph), DGAccumulator_sum(dgas),
    DGAccumulator_max(dgam) {}

  void static go(Graph& _graph, galois::DGAccumulator<uint64_t>& dgas,
                 galois::DGAccumulator<uint32_t>& dgam) {

  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      // TODO currently no GPU support for sanity check operator
      fprintf(stderr, "Warning: No GPU support for sanity check; might get "
                      "wrong results.\n");
    }
  #endif
    dgas.reset();
    dgam.reset();

    galois::do_all(galois::iterate(*_graph.begin(), *_graph.end()),
                   BFSSanityCheck(infinity, &_graph, dgas, dgam), 
                   galois::loopname("BFSSanityCheck"),
                   galois::no_stats());

    uint64_t num_visited = dgas.reduce();

    dgam = current_max;
    uint32_t max_distance = dgam.reduce_max();

    // Only node 0 will print the info
    if (_graph.id == 0) {
      printf("Number of nodes visited is %lu\n", num_visited);
      printf("Max distance is %u\n", max_distance);
    }
  }

  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    if (graph->isOwned(graph->getGID(src)) && 
        src_data.dist_current < local_infinity) {
      DGAccumulator_sum += 1;

      if (current_max < src_data.dist_current) {
        current_max = src_data.dist_current;
      }
    }
  }

};
uint32_t BFSSanityCheck::current_max = 0;

/******************************************************************************/
/* Main */
/******************************************************************************/

static const char* const name = "BFS pull - Distributed Heterogeneous";
static const char* const desc = "BFS pull on Distributed Galois.";
static const char* const url = 0;

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
  galois::StatTimer StatTimer_total("TIMER_TOTAL", regionname); 

  StatTimer_total.start();

  #ifdef __GALOIS_HET_CUDA__
  Graph* hg = distGraphInitialization<NodeData, void, false>(&cuda_ctx);
  #else
  Graph* hg = distGraphInitialization<NodeData, void, false>();
  #endif

  bitset_dist_current.resize(hg->get_local_total_nodes());

  printf("[%d] InitializeGraph::go called\n", net.ID);

  galois::StatTimer StatTimer_init("TIMER_GRAPH_INIT", regionname); 
  StatTimer_init.start();
    InitializeGraph::go((*hg));
  StatTimer_init.stop();
  galois::runtime::getHostBarrier().wait();

  // accumulators for use in operators
  galois::DGAccumulator<unsigned int> DGAccumulator_accum;
  galois::DGAccumulator<uint64_t> DGAccumulator_sum;
  galois::DGAccumulator<uint32_t> DGAccumulator_max;

  for (auto run = 0; run < numRuns; ++run) {
    printf("[%d] BFS::go run %d called\n", net.ID, run);
    std::string timer_str("TIMER_" + std::to_string(run));
    galois::StatTimer StatTimer_main(timer_str.c_str(), regionname);

    StatTimer_main.start();
      BFS::go(*hg, DGAccumulator_accum);
    StatTimer_main.stop();

    // sanity check
    BFSSanityCheck::current_max = 0;
    BFSSanityCheck::go(*hg, DGAccumulator_sum, DGAccumulator_max);

    if ((run + 1) != numRuns) {
      #ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) { 
        bitset_dist_current_reset_cuda(cuda_ctx);
      } else
      #endif
      bitset_dist_current.reset();

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
                                       (*hg).getData(*ii).dist_current);
      }
    #ifdef __GALOIS_HET_CUDA__
    } else if(personality == GPU_CUDA)  {
      for (auto ii = (*hg).begin(); ii != (*hg).end(); ++ii) {
        if ((*hg).isOwned((*hg).getGID(*ii))) 
          galois::runtime::printOutput("% %\n", (*hg).getGID(*ii), 
                                   get_node_dist_current_cuda(cuda_ctx, *ii));
      }
    }
    #endif
  }
  return 0;
}
