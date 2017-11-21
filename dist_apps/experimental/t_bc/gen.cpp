/** Betweeness Centrality (Theoretical) -*- C++ -*-
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
 * Compute Betweeness-Centrality on distributed Galois; Vijaya Ramachandran's
 * theoretical BC 
 *
 * @author Loc Hoang <l_hoang@utexas.edu>
 */

#define __USE_BFS__

constexpr static const char* const REGION_NAME = "T_BC";

/******************************************************************************/
/* Sync code/calls was manually written, not compiler generated */
/******************************************************************************/

#include <iostream>
#include <limits>
#include <random>

#include "galois/DistGalois.h"
#include "galois/gstl.h"
#include "DistBenchStart.h"
#include "galois/DistAccumulator.h"
#include "galois/runtime/Tracer.h"

/******************************************************************************/
/* Declaration of command line arguments */
/******************************************************************************/
namespace cll = llvm::cl;
static cll::opt<unsigned int> maxIterations("maxIterations", 
                               cll::desc("Maximum iterations: Default 10000"), 
                               cll::init(10000));
static cll::opt<bool> singleSourceBC("singleSource", 
                                cll::desc("Use for single source BC"),
                                cll::init(false));
static cll::opt<unsigned int> startSource("srcNodeId", 
                                cll::desc("Starting source node used for "
                                          "betweeness-centrality"),
                                cll::init(0));
static cll::opt<unsigned int> numberOfSources("numOfSources", 
                                cll::desc("Number of sources to use for "
                                          "betweeness-centraility"),
                                cll::init(0));

/******************************************************************************/
/* Graph structure declarations */
/******************************************************************************/

const uint32_t infinity = std::numeric_limits<uint32_t>::max() / 4;
static uint64_t current_src_node = 0;

// NOTE: types assume that these values will not reach uint64_t: it may
// need to be changed for very large graphs
struct NodeData {
  std::vector<uint32_t> minDistances;
  std::vector<uint32_t> shortestPathNumbers;
  //// SSSP vars
  //std::atomic<uint32_t> current_length;

  //uint32_t old_length;

  //// Betweeness centrality vars
  //uint32_t num_shortest_paths;
  //uint32_t num_successors;
  //std::atomic<uint32_t> num_predecessors;
  //std::atomic<uint32_t> trim;
  //std::atomic<uint32_t> to_add;

  //float to_add_float;
  //float dependency;

  //float betweeness_centrality;

  //// used to determine if data has been propogated yet
  //uint8_t propogation_flag;
};

static std::set<uint64_t> random_sources = std::set<uint64_t>();

#ifndef __USE_BFS__
typedef hGraph<NodeData, unsigned int, true> Graph;
#else
typedef hGraph<NodeData, void, true> Graph;
#endif

typedef typename Graph::GraphNode GNode;

// bitsets for tracking updates
galois::DynamicBitSet bitset_to_add;
galois::DynamicBitSet bitset_to_add_float;
galois::DynamicBitSet bitset_num_shortest_paths;
galois::DynamicBitSet bitset_num_successors;
galois::DynamicBitSet bitset_num_predecessors;
galois::DynamicBitSet bitset_trim;
galois::DynamicBitSet bitset_current_length;
galois::DynamicBitSet bitset_propogation_flag;
galois::DynamicBitSet bitset_dependency;

// sync structures
#include "gen_sync.hh"

/******************************************************************************/
/* Functors for running the algorithm */
/******************************************************************************/
uint64_t offset;


struct InitializeGraph {
  Graph *graph;

  InitializeGraph(Graph* _graph) : graph(_graph){}

  /* Initialize the graph */
  void static go(Graph& _graph) {
    const auto& allNodes = _graph.allNodesRange();

    galois::do_all(
      // pass in begin/end to not use local thread ranges
      galois::iterate(allNodes.begin(), allNodes.end()), 
      InitializeGraph{&_graph}, 
      galois::loopname("InitializeGraph"), 
      //galois::loopname(_graph.get_run_identifier("InitializeGraph").c_str()), 
      galois::timeit(),
      galois::no_stats()
    );
  }

  /* Functor passed into the Galois operator to carry out initialization;
   * reset everything */
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    // for now 5 sources per run
    src_data.minDistances.resize(5);
    src_data.shortestPathNumbers.resize(5);

    //src_data.betweeness_centrality = 0;
    //src_data.num_shortest_paths = 0;
    //src_data.num_successors = 0;
    //src_data.num_predecessors = 0;
    //src_data.trim = 0;
    //src_data.to_add = 0;
    //src_data.to_add_float = 0;
    //src_data.dependency = 0;
    //src_data.propogation_flag = false;
  }
};

/* This is used to reset node data when switching to a different 5 source set */
struct InitializeIteration {
  const uint32_t &local_infinity;
  const uint64_t &local_current_src_node;
  Graph *graph;

  InitializeIteration(const uint32_t &_local_infinity,
                      const uint64_t &_local_current_src_node,
                      Graph* _graph) : 
                       local_infinity(_local_infinity),
                       local_current_src_node(_local_current_src_node),
                       graph(_graph){}

  /* Reset necessary graph metadata for next iteration of SSSP */
  void static go(Graph& _graph) {
    const auto& allNodes = _graph.allNodesRange();

    galois::do_all(
      galois::iterate(allNodes.begin(), allNodes.end()), 
      InitializeIteration{infinity, current_src_node, &_graph},
      galois::loopname("InitializeIteration"), 
      galois::no_stats()
    );
  }

  /* Functor passed into the Galois operator to carry out reset of node data
   * (aside from betweeness centrality measure */
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    for (unsigned i = 0; i < 5; i++) {
      if ((offset + i) == graph->getGID(src)) {
        src_data.minDistances[i] = 0;
        src_data.shortestPathNumber[i] = 1;
      } else {
        src_data.minDistances[i] = local_infinity;
        src_data.shortestPathNumbers[i] = 0;
      }
    }

    //bool is_source = graph->getGID(src) == local_current_src_node;

    //if (!is_source) {
    //  src_data.current_length = local_infinity;
    //  src_data.old_length = local_infinity;
    //  src_data.num_shortest_paths = 0;
    //  src_data.propogation_flag = false;
    //} else {
    //  src_data.current_length = 0;
    //  src_data.old_length = 0; 
    //  src_data.num_shortest_paths = 1;
    //  src_data.propogation_flag = true;
    //}
    //src_data.num_predecessors = 0;
    //src_data.num_successors = 0;
    //src_data.dependency = 0;

    //assert(src_data.trim.load() == 0);
    //assert(src_data.to_add.load() == 0);
    //assert(src_data.to_add_float == 0);
  }
};

/* Need a separate call for the first iteration as the condition check is 
 * different */
struct FirstIterationSSSP {
  Graph* graph;
  FirstIterationSSSP(Graph* _graph) : graph(_graph){}

  void static go(Graph& _graph){
    unsigned int __begin, __end;
    if (_graph.isLocal(current_src_node)) {
      __begin = _graph.getLID(current_src_node);
      __end = __begin + 1;
    } else {
      __begin = 0;
      __end = 0;
    }

    galois::do_all(
      galois::iterate(__begin, __end), 
      FirstIterationSSSP(&_graph),
      galois::loopname("SSSP"),
      galois::no_stats()
    );

    // Next op will read src, current length
    _graph.sync<writeDestination, readSource, Reduce_min_current_length, 
                Broadcast_current_length, Bitset_current_length>(
                "SSSP");
  }

  /* Does SSSP, push/filter based */
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);
    for (auto current_edge : graph->edges(src)) {
      GNode dst = graph->getEdgeDst(current_edge);

      if (src == dst) {
        continue;
      }

      auto& dst_data = graph->getData(dst);

      #ifndef __USE_BFS__
      uint32_t new_dist = graph->getEdgeData(current_edge) + 
                              src_data.current_length + 1;
      #else
      // BFS 
      uint32_t new_dist = 1 + src_data.current_length;
      #endif

      galois::atomicMin(dst_data.current_length, new_dist);

      bitset_current_length.set(dst);
    }
  }
};

/* Sub struct for running SSSP (beyond 1st iteration) */
struct SSSP {
  Graph* graph;
  galois::DGAccumulator<uint32_t>& DGAccumulator_accum;

  SSSP(Graph* _graph, galois::DGAccumulator<uint32_t>& dga) : 
    graph(_graph), DGAccumulator_accum(dga) { }

  void static go(Graph& _graph, galois::DGAccumulator<uint32_t>& dga) {
    FirstIterationSSSP::go(_graph);

    // starts at 1 since FirstSSSP takes care of the first one
    uint32_t iterations = 1;
    uint32_t accum_result;

    const auto& nodesWithEdges = _graph.allNodesWithEdgesRange();

    do {
      _graph.set_num_iter(iterations);
      dga.reset();

      {
      galois::do_all(
        galois::iterate(nodesWithEdges),
        SSSP(&_graph, dga), 
        galois::loopname("SSSP"), 
        //galois::loopname(_graph.get_run_identifier("SSSP").c_str()), 
        galois::timeit(),
        galois::no_stats()
      );
      }

      iterations++;

      accum_result = dga.reduce();

      if (accum_result) {
        _graph.sync<writeDestination, readSource, Reduce_min_current_length, 
                    Broadcast_current_length, Bitset_current_length>("SSSP");
      } else {
        // write destination, read any, fails.....
        // sync src and dst
        if (_graph.is_vertex_cut()) { // TODO: only needed for cartesian cut
          // no bitset used = sync all; at time of writing, vertex cut
          // syncs cause the bit to be reset prematurely, so using the bitset
          // will lead to incorrect results as it will not sync what is
          // necessary
          _graph.sync<writeDestination, readSource, Reduce_min_current_length, 
                       Broadcast_current_length, Bitset_current_length>("SSSP");
          _graph.sync<writeDestination, readDestination, Reduce_min_current_length, 
                       Broadcast_current_length>("SSSP");
        } else {
          _graph.sync<writeDestination, readAny, Reduce_min_current_length, 
                      Broadcast_current_length, 
                      Bitset_current_length>("SSSP");
        }
      }
    } while (accum_result);
  }

  /* Does SSSP, push/filter based */
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    if (src_data.old_length > src_data.current_length) {
      src_data.old_length = src_data.current_length;

      for (auto current_edge : graph->edges(src)) {
        GNode dst = graph->getEdgeDst(current_edge);

        if (src == dst) {
          continue;
        }

        auto& dst_data = graph->getData(dst);

        #ifndef __USE_BFS__
        uint32_t new_dist = graph->getEdgeData(current_edge) + 
                            src_data.current_length + 1;
        #else
        uint32_t new_dist = 1 + src_data.current_length;
        #endif

        uint32_t old = galois::atomicMin(dst_data.current_length, new_dist);

        if (old > new_dist) {
          bitset_current_length.set(dst);
          DGAccumulator_accum += 1;
        }
      }
    }
  }
};


/******************************************************************************/
/* Sanity check */
/******************************************************************************/

//struct Sanity {
//  Graph* graph;
//
//  static float current_max;
//  static float current_min;
//
//  galois::DGAccumulator<float>& DGAccumulator_max;
//  galois::DGAccumulator<float>& DGAccumulator_min;
//  galois::DGAccumulator<double>& DGAccumulator_sum;
//
//  Sanity(Graph* _graph,
//      galois::DGAccumulator<float>& _DGAccumulator_max,
//      galois::DGAccumulator<float>& _DGAccumulator_min,
//      galois::DGAccumulator<double>& _DGAccumulator_sum
//  ) : 
//    graph(_graph),
//    DGAccumulator_max(_DGAccumulator_max),
//    DGAccumulator_min(_DGAccumulator_min),
//    DGAccumulator_sum(_DGAccumulator_sum) {}
//
//  void static go(Graph& _graph,
//    galois::DGAccumulator<float>& DGA_max,
//    galois::DGAccumulator<float>& DGA_min,
//    galois::DGAccumulator<double>& DGA_sum
//  ) {
//  #ifdef __GALOIS_HET_CUDA__
//    if (personality == GPU_CUDA) {
//      // TODO currently no GPU support for sanity check operator
//      fprintf(stderr, "Warning: No GPU support for sanity check; might get "
//                      "wrong results.\n");
//    }
//  #endif
//
//    DGA_max.reset();
//    DGA_min.reset();
//    DGA_sum.reset();
//
//    galois::do_all(galois::iterate(_graph.allNodesRange().begin(), 
//                                   _graph.allNodesRange().end()),
//                   Sanity(
//                     &_graph,
//                     DGA_max,
//                     DGA_min,
//                     DGA_sum
//                   ), 
//                   galois::loopname("Sanity"),
//                   galois::no_stats());
//
//    DGA_max = current_max;
//    DGA_min = current_min;
//
//    float max_bc = DGA_max.reduce_max();
//    float min_bc = DGA_min.reduce_min();
//    double bc_sum = DGA_sum.reduce();
//
//    // Only node 0 will print data
//    if (galois::runtime::getSystemNetworkInterface().ID == 0) {
//      printf("Max BC is %f\n", max_bc);
//      printf("Min BC is %f\n", min_bc);
//      printf("BC sum is %f\n", bc_sum);
//    }
//  }
//  
//  /* Gets the max, min rank from all owned nodes and
//   * also the sum of ranks */
//  void operator()(GNode src) const {
//    NodeData& sdata = graph->getData(src);
//
//    if (graph->isOwned(graph->getGID(src))) {
//      if (current_max < sdata.betweeness_centrality) {
//        current_max = sdata.betweeness_centrality;
//      }
//
//      if (current_min > sdata.betweeness_centrality) {
//        current_min = sdata.betweeness_centrality;
//      }
//
//      DGAccumulator_sum += sdata.betweeness_centrality;
//    }
//  }
//};
//float Sanity::current_max = 0;
//float Sanity::current_min = std::numeric_limits<float>::max() / 4;

/******************************************************************************/
/* Main method for running */
/******************************************************************************/

constexpr static const char* const name = "Betweeness Centrality"; 
constexpr static const char* const desc = "Betweeness Centrality on Distributed "
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
  Graph* hg = twoWayDistGraphInitialization<NodeData, void>(&cuda_ctx);
  #else
  Graph* hg = twoWayDistGraphInitialization<NodeData, void>();
  #endif

  //// random num generate for sources
  //std::minstd_rand0 r_generator;
  //r_generator.seed(100);
  //std::uniform_int_distribution<uint64_t> r_dist(0, h_graph->globalSize() - 1);

  //if (numberOfSources != 0) {
  //  // uncomment this to have srcnodeid included as well
  //  //random_sources.insert(startSource);

  //  while (random_sources.size() < numberOfSources) {
  //    random_sources.insert(r_dist(r_generator));
  //  }
  //}

  //#ifndef NDEBUG
  //int counter = 0;
  //for (auto i = random_sources.begin(); i != random_sources.end(); i++) {
  //  printf("Source #%d: %lu\n", counter, *i);
  //  counter++;
  //}
  //#endif

  //bitset_to_add.resize(h_graph->size());
  //bitset_to_add_float.resize(h_graph->size());
  //bitset_num_shortest_paths.resize(h_graph->size());
  //bitset_num_successors.resize(h_graph->size());
  //bitset_num_predecessors.resize(h_graph->size());
  //bitset_trim.resize(h_graph->size());
  //bitset_current_length.resize(h_graph->size());
  //bitset_propogation_flag.resize(h_graph->size());
  //bitset_dependency.resize(h_graph->size());

  galois::gPrint("[", net.ID, "] InitializeGraph::go called\n");

  galois::StatTimer StatTimer_graph_init("TIMER_GRAPH_INIT", REGION_NAME);
  StatTimer_graph_init.start();
    InitializeGraph::go((*h_graph));
  StatTimer_graph_init.stop();
  galois::runtime::getHostBarrier().wait();

  // shared DG accumulator among all steps
  galois::DGAccumulator<uint32_t> dga;

  //// sanity dg accumulators
  //galois::DGAccumulator<float> dga_max;
  //galois::DGAccumulator<float> dga_min;
  //galois::DGAccumulator<double> dga_sum;

  for (auto run = 0; run < numRuns; ++run) {
    galois::gPrint("[", net.ID, "] BC::go run ", run, " called\n");
    std::string timer_str("TIMER_" + std::to_string(run));
    galois::StatTimer StatTimer_main(timer_str.c_str(), REGION_NAME);

    StatTimer_main.start();
    //  BC::go(*h_graph, dga);
    StatTimer_main.stop();

    //Sanity::current_max = 0;
    //Sanity::current_min = std::numeric_limits<float>::max() / 4;

    //Sanity::go(
    //  *h_graph,
    //  dga_max,
    //  dga_min,
    //  dga_sum
    //);

    // re-init graph for next run
    if ((run + 1) != numRuns) {
      galois::runtime::getHostBarrier().wait();
      (*h_graph).set_num_run(run + 1);

      //bitset_to_add.reset();
      //bitset_to_add_float.reset();
      //bitset_num_shortest_paths.reset();
      //bitset_num_successors.reset();
      //bitset_num_predecessors.reset();
      //bitset_trim.reset();
      //bitset_current_length.reset();
      //bitset_propogation_flag.reset();
      //bitset_dependency.reset();

      InitializeGraph::go((*h_graph));
      galois::runtime::getHostBarrier().wait();
    }
  }

  StatTimer_total.stop();

  //// Verify, i.e. print out graph data for examination
  //if (verify) {
  //  char *v_out = (char*)malloc(40);
  //  #ifdef __GALOIS_HET_CUDA__
  //  if (personality == CPU) { 
  //  #endif
  //    for (auto ii = (*h_graph).masterNodesRange().begin(); 
  //              ii != (*h_graph).masterNodesRange().end(); 
  //              ++ii) {
  //      // outputs betweenness centrality
  //      sprintf(v_out, "%lu %.9f\n", (*h_graph).getGID(*ii),
  //              (*h_graph).getData(*ii).betweeness_centrality);
  //      galois::runtime::printOutput(v_out);
  //    }
  //  #ifdef __GALOIS_HET_CUDA__
  //  } else if (personality == GPU_CUDA) {
  //    for (auto ii = (*h_graph).masterNodesRange().begin(); 
  //              ii != (*h_graph).masterNodesRange().end(); 
  //              ++ii) {
  //      sprintf(v_out, "%lu %.9f\n", (*h_graph).getGID(*ii),
  //              get_node_betweeness_centrality_cuda(cuda_ctx, *ii));
  //      galois::runtime::printOutput(v_out);
  //      memset(v_out, '\0', 40);
  //    }
  //  }
  //  #endif
  //  free(v_out);
  //}

  return 0;
}
