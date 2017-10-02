/** KCore -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2017, The University of Texas at Austin. All rights reserved.
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
 * Compute KCore on distributed Galois pull style.
 *
 * @author Loc Hoang <l_hoang@utexas.edu>
 */

/******************************************************************************/
/* Sync code/calls was manually written, not compiler generated */
/******************************************************************************/

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

constexpr static const char* const REGION_NAME = "KCore";

/******************************************************************************/
/* Declaration of command line arguments */
/******************************************************************************/
namespace cll = llvm::cl;
static cll::opt<unsigned int> maxIterations("maxIterations", 
                               cll::desc("Maximum iterations: Default 10000"), 
                               cll::init(10000));
// required k specification for k-core
static cll::opt<unsigned int> k_core_num("kcore",
                                     cll::desc("KCore value"),
                                     cll::Required);

/******************************************************************************/
/* Graph structure declarations + other inits */
/******************************************************************************/

struct NodeData {
  uint32_t current_degree;
  uint32_t trim;
  uint8_t flag;
  uint8_t pull_flag;
};

typedef hGraph<NodeData, void> Graph;
typedef typename Graph::GraphNode GNode;

// bitset for tracking updates
galois::DynamicBitSet bitset_current_degree;
galois::DynamicBitSet bitset_trim;

// add all sync/bitset structs (needs above declarations)
#include "gen_sync.hh"

/******************************************************************************/
/* Functors for running the algorithm */
/******************************************************************************/

/* Degree counting
 * Called by InitializeGraph1 */
struct DegreeCounting {
  Graph *graph;

  DegreeCounting(Graph* _graph) : graph(_graph){}

  /* Initialize the entire graph node-by-node */
  void static go(Graph& _graph) {
    const auto& nodesWithEdges = _graph.allNodesWithEdgesRange();

  #ifdef __GALOIS_HET_CUDA__
    // TODO calls all wrong
    if (personality == GPU_CUDA) {
      std::string impl_str("CUDA_DO_ALL_IMPL_InitializeGraph2_" + 
                           (_graph.get_run_identifier()));
      galois::StatTimer StatTimer_cuda(impl_str.c_str(), REGION_NAME);
      StatTimer_cuda.start();
      InitializeGraph2_cuda(*nodesWithEdges.begin(), *nodesWithEdges.end(),
                            cuda_ctx);
      StatTimer_cuda.stop();
    } else if (personality == CPU)
  #endif
    galois::do_all(
      galois::iterate(nodesWithEdges),
      DegreeCounting{ &_graph },
      galois::loopname(_graph.get_run_identifier("DegreeCounting").c_str()),
      galois::timeit(),
      galois::no_stats()
    );

    _graph.sync<writeSource, readAny, Reduce_add_current_degree, 
      Broadcast_current_degree, Bitset_current_degree>("DegreeCounting");
  }

  /* Calculate degree of nodes by checking how many nodes have it as a dest and
   * adding for every dest (works same way in pull version since it's a symmetric
   * graph) */
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    // technically can use std::dist, but this is more easily recognizable
    // by compiler + this is init so it doesn't matter much
    for (auto current_edge = graph->edge_begin(src), 
              end_edge = graph->edge_end(src);
         current_edge != end_edge;
         current_edge++) {
      src_data.current_degree++;
      bitset_current_degree.set(src);
    }
  }
};


/* Initialize: initial field setup */
struct InitializeGraph {
  Graph *graph;

  InitializeGraph(Graph* _graph) : graph(_graph){}

  /* Initialize the entire graph node-by-node */
  void static go(Graph& _graph) {
    const auto& allNodes = _graph.allNodesRange();

#ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      // TODO calls all wrong
      std::string impl_str("CUDA_DO_ALL_IMPL_InitializeGraph1_" + 
                           (_graph.get_run_identifier()));
      galois::StatTimer StatTimer_cuda(impl_str.c_str(), REGION_NAME);
      StatTimer_cuda.start();
      InitializeGraph1_cuda(*(allNodes.begin()), *(allNodes.end()), 
                            cuda_ctx);
      StatTimer_cuda.stop();
    } else if (personality == CPU)
  #endif
     galois::do_all(
        galois::iterate(allNodes.begin(), allNodes.end()),
        InitializeGraph{ &_graph },
        galois::loopname(_graph.get_run_identifier("InitializeGraph").c_str()),
        galois::timeit(),
        galois::no_stats()
      );

    // degree calculation
    DegreeCounting::go(_graph);
  }

  /* Setup intial fields */
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);
    src_data.flag = true;
    src_data.trim = 0;
    src_data.current_degree = 0;
    src_data.pull_flag = false;
  }
};


/* Updates liveness of a node + updates flag that says if node has been pulled 
 * from */
struct LiveUpdate {
  cll::opt<uint32_t>& local_k_core_num;
  Graph* graph;
  galois::DGAccumulator<unsigned int>& DGAccumulator_accum;

  LiveUpdate(cll::opt<uint32_t>& _kcore, Graph* _graph,
             galois::DGAccumulator<unsigned int>& _dga) : 
    local_k_core_num(_kcore), graph(_graph), DGAccumulator_accum(_dga) {}
  
  void static go(Graph& _graph, galois::DGAccumulator<unsigned int>& dga) {
    const auto& allNodes = _graph.allNodesRange();
    dga.reset();

    // TODO GPU code

    galois::do_all(
      galois::iterate(allNodes.begin(), allNodes.end()),
      LiveUpdate{ k_core_num, &_graph, dga },
      galois::loopname(_graph.get_run_identifier("LiveUpdate").c_str()),
      galois::timeit(),
      galois::no_stats()
    );

    // no sync necessary as all nodes should have updated
  }

  /**
   * Mark a node dead if degree is under kcore number and mark it 
   * available for pulling from.
   *
   * If dead, and pull flag is on, then turn off flag as you don't want to
   * be pulled from more than once.
   */
  void operator()(GNode src) const {
    NodeData& sdata = graph->getData(src);

    if (sdata.flag) {
      if (sdata.trim > 0) {
        sdata.current_degree = sdata.current_degree - sdata.trim;
      }

      if (sdata.current_degree < local_k_core_num) {
        sdata.flag = false;
        DGAccumulator_accum += 1;

        // let neighbors pull from me next round
        assert(sdata.pull_flag == false);
        sdata.pull_flag = true;
      }
    } else {
      // dead
      if (sdata.pull_flag) {
        // do not allow neighbors to pull value from this node anymore
        sdata.pull_flag = false;
      }
    }

    // always reset trim
    sdata.trim = 0;
  }
};

/* Step that determines if a node is dead and updates its neighbors' trim
 * if it is */
struct KCore {
  Graph* graph;

  KCore(Graph* _graph) : graph(_graph) {}

  void static go(Graph& _graph, galois::DGAccumulator<unsigned int>& dga) {
    unsigned iterations = 0;
    
    const auto& nodesWithEdges = _graph.allNodesWithEdgesRange();

    do {
      _graph.set_num_iter(iterations);

      galois::do_all(
        galois::iterate(nodesWithEdges),
        KCore{ &_graph },
        galois::loopname(_graph.get_run_identifier("KCore").c_str()),
        galois::timeit(),
        galois::no_stats()
      );

      _graph.sync<writeSource, readAny, Reduce_add_trim, Broadcast_trim, 
                  Bitset_trim>("KCore");

      // update live/deadness
      LiveUpdate::go(_graph, dga);

      iterations++;
    } while ((iterations < maxIterations) && dga.reduce(_graph.get_run_identifier()));

    if (galois::runtime::getSystemNetworkInterface().ID == 0) {
      galois::runtime::reportStat_Serial(REGION_NAME, 
        "NUM_ITERATIONS_" + std::to_string(_graph.get_run_num()), 
        (unsigned long)iterations);
    }
  }

  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    // only if node is alive we do things
    if (src_data.flag) {
      // if dst node is dead, increment trim by one so we can decrement
      // our degree later
      for (auto current_edge = graph->edge_begin(src), 
                end_edge = graph->edge_end(src);
           current_edge != end_edge; 
           ++current_edge) {
         GNode dst = graph->getEdgeDst(current_edge);
         NodeData& dst_data = graph->getData(dst);

         if (dst_data.pull_flag) {
           galois::add(src_data.trim, (uint32_t)1);
           bitset_trim.set(src);
         }
      }
    }
  }
};

/******************************************************************************/
/* Sanity check operators */
/******************************************************************************/

/* Gets the total number of nodes that are still alive */
struct GetAliveDead {
  Graph* graph;
  galois::DGAccumulator<uint64_t>& DGAccumulator_accum;
  galois::DGAccumulator<uint64_t>& DGAccumulator_accum2;

  GetAliveDead(Graph* _graph, 
               galois::DGAccumulator<uint64_t>& _DGAccumulator_accum,
               galois::DGAccumulator<uint64_t>& _DGAccumulator_accum2) : 
      graph(_graph), DGAccumulator_accum(_DGAccumulator_accum),
      DGAccumulator_accum2(_DGAccumulator_accum2) {}

  void static go(Graph& _graph,
    galois::DGAccumulator<uint64_t>& dga1,
    galois::DGAccumulator<uint64_t>& dga2) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      // TODO currently no GPU support for sanity check operator
      fprintf(stderr, "Warning: No GPU support for sanity check; might get "
                      "wrong results.\n");
    }
  #endif
    dga1.reset();
    dga2.reset();

    galois::do_all(galois::iterate(_graph.begin(), _graph.end()),
                   GetAliveDead(&_graph, dga1, dga2), 
                   galois::loopname("GetAliveDead"),
                   galois::numrun(_graph.get_run_identifier()),
                   galois::no_stats());

    uint32_t num_alive = dga1.reduce();
    uint32_t num_dead = dga2.reduce();

    // Only node 0 will print data
    if (_graph.id == 0) {
      printf("Number of nodes alive is %u, dead is %u\n", num_alive,
             num_dead);
    }
  }

  /* Check if an owned node is alive/dead: increment appropriate accumulator */
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    if (graph->isOwned(graph->getGID(src))) {
      if (src_data.flag) {
        DGAccumulator_accum += 1;
      } else {
        DGAccumulator_accum2 += 1;
      }
    }
  }
};

/******************************************************************************/
/* Main method for running */
/******************************************************************************/

constexpr static const char* const name = "KCore - Distributed Heterogeneous "
                                          "Pull Topological.";
constexpr static const char* const desc = "KCore on Distributed Galois.";
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
  Graph* h_graph = symmetricDistGraphInitialization<NodeData, void>(&cuda_ctx);
  #else
  Graph* h_graph = symmetricDistGraphInitialization<NodeData, void>();
  #endif

  bitset_current_degree.resize(h_graph->get_local_total_nodes());
  bitset_trim.resize(h_graph->get_local_total_nodes());

  printf("[%d] InitializeGraph::go called\n", net.ID);
  galois::StatTimer StatTimer_graph_init("TIMER_GRAPH_INIT", REGION_NAME);

  StatTimer_graph_init.start();
    InitializeGraph::go((*h_graph));
  StatTimer_graph_init.stop();
  galois::runtime::getHostBarrier().wait();

  galois::DGAccumulator<unsigned int> DGAccumulator_accum;
  galois::DGAccumulator<uint64_t> dga1;
  galois::DGAccumulator<uint64_t> dga2;

  for (auto run = 0; run < numRuns; ++run) {
    printf("[%d] KCore::go run %d called\n", net.ID, run);
    std::string timer_str("TIMER_" + std::to_string(run));
    galois::StatTimer StatTimer_main(timer_str.c_str(), REGION_NAME);

    StatTimer_main.start();
      KCore::go(*h_graph, DGAccumulator_accum);
    StatTimer_main.stop();

    // sanity check
    GetAliveDead::go(*h_graph, dga1, dga2);

    // re-init graph for next run
    if ((run + 1) != numRuns) {
      (*h_graph).reset_num_iter(run+1);

      #ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) { 
        bitset_current_degree_reset_cuda(cuda_ctx);
        bitset_trim_reset_cuda(cuda_ctx);
      } else
      #endif
      { bitset_current_degree.reset();
      bitset_trim.reset(); }

      InitializeGraph::go((*h_graph));
      galois::runtime::getHostBarrier().wait();
    }
  }

  StatTimer_total.stop();

  // Verify, i.e. print out graph data for examination
  if (verify) {
    #ifdef __GALOIS_HET_CUDA__
    if (personality == CPU) { 
    #endif
      for (auto ii = (*h_graph).begin(); ii != (*h_graph).end(); ++ii) {
        if ((*h_graph).isOwned((*h_graph).getGID(*ii))) {
          // prints the flag (alive/dead)
          galois::runtime::printOutput("% %\n", (*h_graph).getGID(*ii), 
                                       (bool)(*h_graph).getData(*ii).flag);
        }


        // does a sanity check as well: 
        // degree higher than kcore if node is alive
        if (!((*h_graph).getData(*ii).flag)) {
          assert((*h_graph).getData(*ii).current_degree < k_core_num);
        } 
      }
    #ifdef __GALOIS_HET_CUDA__
    } else if (personality == GPU_CUDA) {
      for (auto ii = (*h_graph).begin(); ii != (*h_graph).end(); ++ii) {
        if ((*h_graph).isOwned((*h_graph).getGID(*ii))) {
          galois::runtime::printOutput("% %\n", (*h_graph).getGID(*ii), 
                                     (bool)get_node_flag_cuda(cuda_ctx, *ii));
        }
      }
    }
    #endif
  }

  return 0;
}
