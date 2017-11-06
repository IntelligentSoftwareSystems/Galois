/** Betweeness Centrality -*- C++ -*-
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
 * Compute Betweeness-Centrality on distributed Galois using SSSP for distances
 *
 * @author Loc Hoang <l_hoang@utexas.edu>
 */

// TODO GPU code needed

constexpr static const char* const REGION_NAME = "BC";

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

#ifdef __GALOIS_HET_CUDA__
#include "gen_cuda.h"
struct CUDA_Context *cuda_ctx;
#endif

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
  // SSSP vars
  uint32_t current_length;

  uint32_t num_shortest_paths;
  std::atomic<uint32_t> num_successors;
  uint32_t num_predecessors;

  std::atomic<uint32_t> trim;
  uint32_t to_add;

  std::atomic<float> to_add_float;
  float dependency;

  float betweeness_centrality;

  // used to determine if data has been propogated yet
  uint8_t propogation_flag;
};

static std::set<uint64_t> random_sources = std::set<uint64_t>();

#ifndef __USE_BFS__
typedef hGraph<NodeData, unsigned int> Graph;
#else
typedef hGraph<NodeData, void> Graph;
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
struct InitializeGraph {
  Graph *graph;

  InitializeGraph(Graph* _graph) : graph(_graph){}

  /* Initialize the graph */
  void static go(Graph& _graph) {
    const auto& allNodes = _graph.allNodesRange();

    #ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) {
        std::string impl_str(
          _graph.get_run_identifier("CUDA_DO_ALL_IMPL_InitializeGraph")
        );
        galois::StatTimer StatTimer_cuda(impl_str.c_str(), REGION_NAME);
        StatTimer_cuda.start();
        InitializeGraph_cuda(*allNodes.begin(), *allNodes.end(), cuda_ctx);
        StatTimer_cuda.stop();
      } else if (personality == CPU)
    #endif
    galois::do_all(
      galois::iterate(allNodes.begin(), allNodes.end()),
      InitializeGraph{&_graph}, 
      galois::loopname("InitializeGraph"));
  }

  /* Functor passed into the Galois operator to carry out initialization;
   * reset everything */
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    src_data.betweeness_centrality = 0;

    src_data.num_shortest_paths = 0;
    src_data.num_successors = 0;
    src_data.num_predecessors = 0;
    src_data.trim = 0;
    src_data.to_add = 0;
    src_data.to_add_float = 0;
    src_data.dependency = 0;
    src_data.propogation_flag = false;
  }
};

/* This is used to reset node data when switching to a difference source */
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

    #ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) {
        std::string impl_str(
          _graph.get_run_identifier("CUDA_DO_ALL_IMPL_InitializeIteration")
        );

        galois::StatTimer StatTimer_cuda(impl_str.c_str(), REGION_NAME);
        StatTimer_cuda.start();
        InitializeIteration_cuda(
          *allNodes.begin(), *allNodes.end(),
          infinity, current_src_node, cuda_ctx
        );
        StatTimer_cuda.stop();
      } else if (personality == CPU)
    #endif
    galois::do_all(
      galois::iterate(allNodes.begin(), allNodes.end()),
      InitializeIteration{infinity, current_src_node, &_graph},
      galois::loopname("InitializeIteration"));
  }

  /* Functor passed into the Galois operator to carry out reset of node data
   * (aside from betweeness centrality measure */
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    bool is_source = graph->getGID(src) == local_current_src_node;

    if (!is_source) {
      src_data.current_length = local_infinity;
      src_data.num_shortest_paths = 0;
      src_data.propogation_flag = false;
    } else {
      src_data.current_length = 0;
      src_data.num_shortest_paths = 1;
      src_data.propogation_flag = true;
    }

    src_data.num_successors = 0;
    src_data.num_predecessors = 0;

    assert(src_data.trim == 0);
    assert(src_data.to_add == 0);
    assert(src_data.to_add_float.load() == 0);
  }
};

/* Sub struct for running SSSP (beyond 1st iteration) */
struct SSSP {
  Graph* graph;
  galois::DGAccumulator<uint32_t>& DGAccumulator_accum;

  SSSP(Graph* _graph, galois::DGAccumulator<uint32_t>& dga) : 
    graph(_graph), DGAccumulator_accum(dga) { }

  void static go(Graph& _graph, galois::DGAccumulator<uint32_t>& dga) {
    uint32_t iterations = 0;
    uint32_t accum_result;

    const auto& nodesWithEdges = _graph.allNodesWithEdgesRange();

    do {
      _graph.set_num_iter(iterations);
      dga.reset();

    #ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) {
        //std::string impl_str("CUDA_DO_ALL_IMPL_SSSP_0");
        std::string impl_str(
          _graph.get_run_identifier("CUDA_DO_ALL_IMPL_SSSP")
        );
        galois::StatTimer StatTimer_cuda(impl_str.c_str(), REGION_NAME);
        StatTimer_cuda.start();
        int __retval = 0;
        SSSP_cuda(*nodesWithEdges.begin(), *nodesWithEdges.end(), __retval, cuda_ctx);
        dga += __retval;
        StatTimer_cuda.stop();
      } else if (personality == CPU)
    #endif
      {
      galois::do_all(
        galois::iterate(nodesWithEdges),
        SSSP(&_graph, dga), 
        galois::loopname("SSSP"));
      }

      iterations++;

      accum_result = dga.reduce();

      if (accum_result) {
        _graph.sync<writeSource, readDestination, Reduce_min_current_length, 
                    Broadcast_current_length, Bitset_current_length>("SSSP");
      } else {
        // write destination, read any, fails if vertex cut + bitset.....
        // sync src and dst
        if (_graph.is_vertex_cut()) { // TODO: only needed for cartesian cut
          // no bitset used = sync all; at time of writing, vertex cut
          // syncs cause the bit to be reset prematurely, so using the bitset
          // will lead to incorrect results as it will not sync what is
          // necessary
          // TODO reason about if this still applies to pull style version;
          // I know it happened in push style....
          _graph.sync<writeSource, readDestination, Reduce_min_current_length, 
                       Broadcast_current_length, Bitset_current_length>("SSSP");
          _graph.sync<writeSource, readSource, Reduce_min_current_length, 
                       Broadcast_current_length>("SSSP");
        } else {
          _graph.sync<writeSource, readAny, Reduce_min_current_length, 
                      Broadcast_current_length, 
          //            Bitset_current_length>("SSSP_cur_len_any");
                      Bitset_current_length>("SSSP");
        }
      }
    } while (accum_result);
  }

  /* Does SSSP */
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);
    for (auto current_edge : graph->edges(src)) {
      GNode dst = graph->getEdgeDst(current_edge);
      auto& dst_data = graph->getData(dst);

      #ifndef __USE_BFS__
      uint32_t new_dist = graph->getEdgeData(current_edge) + 
                              dst_data.current_length + 1;
      #else
      uint32_t new_dist = 1 + dst_data.current_length;
      #endif

      uint32_t old = galois::min(src_data.current_length, new_dist);

      if (old > new_dist) {
        bitset_current_length.set(src);
        DGAccumulator_accum += 1;
      }
    }
  }
};

/* Struct to get pred and succ on the SSSP DAG */
struct PredAndSucc {
  const uint32_t &local_infinity;
  Graph* graph;

  PredAndSucc(const uint32_t &_local_infinity, Graph* _graph) : 
      local_infinity(_local_infinity), graph(_graph) {}

  void static go(Graph& _graph){
    const auto& nodesWithEdges = _graph.allNodesWithEdgesRange();
    
    #ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) {
        //std::string impl_str("CUDA_DO_ALL_IMPL_PredAndSucc");
        std::string impl_str(
          _graph.get_run_identifier("CUDA_DO_ALL_IMPL_PredAndSucc")
        );
        galois::StatTimer StatTimer_cuda(impl_str.c_str(), REGION_NAME);
        StatTimer_cuda.start();
        PredAndSucc_cuda(*nodesWithEdges.begin(), *nodesWithEdges.end(),
                         infinity, cuda_ctx);
        StatTimer_cuda.stop();
      } else if (personality == CPU)
    #endif
    {
    galois::do_all(
      galois::iterate(nodesWithEdges),
      PredAndSucc(infinity, &_graph), 
      galois::loopname("PredAndSucc"),
      galois::no_stats()
    );
    }

    _graph.sync<writeSource, readAny, Reduce_add_num_predecessors, 
                Broadcast_num_predecessors, 
                Bitset_num_predecessors>("PredAndSucc");
    _graph.sync<writeDestination, readAny, Reduce_add_num_successors, 
                Broadcast_num_successors, 
                Bitset_num_successors>("PredAndSucc");
  }

  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    if (src_data.current_length != local_infinity) {
      for (auto current_edge : graph->edges(src)) {
        GNode dst = graph->getEdgeDst(current_edge);
        auto& dst_data = graph->getData(dst);


        #ifndef __USE_BFS__
        uint32_t edge_weight = graph->getEdgeData(current_edge) + 1;
        #else
        uint32_t edge_weight = 1;
        #endif

        if ((dst_data.current_length + edge_weight) == src_data.current_length) {
          // dest on shortest path with this node as successor
          galois::add(src_data.num_predecessors, (unsigned int)1);
          galois::atomicAdd(dst_data.num_successors, (unsigned int)1);

          bitset_num_successors.set(dst);
          bitset_num_predecessors.set(src);
        }
      }
    }
  }
};

/* Uses an incremented trim value to decrement the predecessor: the trim value
 * has to be synchronized across ALL nodes (including mirrors) 
 * Increment num_shortest_paths using the to_add variable which should be 
 * sync'd among source nodes */
struct NumShortestPathsChanges {
  const uint32_t &local_infinity;
  Graph* graph;

  NumShortestPathsChanges(const uint32_t &_local_infinity, Graph* _graph) : 
      local_infinity(_local_infinity), graph(_graph) {}

  void static go(Graph& _graph) {
    const auto& allNodes = _graph.allNodesRange();

    #ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) {
        //std::string impl_str("CUDA_DO_ALL_IMPL_NumShortestPathsChanges");
        std::string impl_str(
          _graph.get_run_identifier("CUDA_DO_ALL_IMPL_NumShortestPathsChanges")
        );
        galois::StatTimer StatTimer_cuda(impl_str.c_str(), REGION_NAME);
        StatTimer_cuda.start();
        NumShortestPathsChanges_cuda(
          *allNodes.begin(),
          *allNodes.end(),
          cuda_ctx
        );
        StatTimer_cuda.stop();
      } else if (personality == CPU)
    #endif
    galois::do_all(
      galois::iterate(allNodes.begin(), allNodes.end()),
      NumShortestPathsChanges{infinity, &_graph}, 
      galois::loopname("NumShortestPathsChanges"), 
      galois::no_stats()
    );
  }

  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    if (src_data.current_length != local_infinity) {
      if (src_data.num_predecessors == 0 && src_data.propogation_flag) {
        assert(src_data.trim == 0);
        src_data.propogation_flag = false;
      } else if (src_data.trim > 0) {
        // decrement predecessor by trim then reset

        //if (src_data.trim > src_data.num_predecessors) {
        //  uint64_t num_e = graph->edge_end(src) - graph->edge_begin(src);
        //  printf("ERROR: src %lu trim is %u, pred is %u e %lu\n",
        //         graph->L2G(src), src_data.trim.load(), src_data.num_predecessors,
        //         num_e);
        //}

        assert(src_data.trim <= src_data.num_predecessors); 

        src_data.num_predecessors = src_data.num_predecessors - src_data.trim;
        src_data.trim = 0;

        // if I hit 0 predecessors after trim, set the flag to true (i.e. says
        // I'm ready to have my value pulled)
        if (src_data.num_predecessors == 0) {
          assert(!src_data.propogation_flag);
          src_data.propogation_flag = true;
        }
      }
    }

    // increment num_shortest_paths by to_add then reset
    if (src_data.to_add > 0) {
      src_data.num_shortest_paths += src_data.to_add;
      src_data.to_add = 0;
    }
  }
};

/* Calculate the number of shortest paths for each node */
struct NumShortestPaths {
  const uint32_t &local_infinity;
  Graph* graph;
  galois::DGAccumulator<uint32_t>& DGAccumulator_accum;

  NumShortestPaths(const uint32_t &_local_infinity,
                   Graph* _graph, galois::DGAccumulator<uint32_t>& dga) : 
     local_infinity(_local_infinity), graph(_graph), DGAccumulator_accum(dga) {}

  void static go(Graph& _graph, galois::DGAccumulator<uint32_t>& dga) {
    uint32_t iterations = 0;
    uint32_t accum_result;

    const auto& nodesWithEdges = _graph.allNodesWithEdgesRange();

    do {
      _graph.set_num_iter(iterations);
      dga.reset();

    #ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) {
        //std::string impl_str("CUDA_DO_ALL_IMPL_NumShortestPaths");
        std::string impl_str(
          _graph.get_run_identifier("CUDA_DO_ALL_IMPL_NumShortestPaths")
        );
        galois::StatTimer StatTimer_cuda(impl_str.c_str(), REGION_NAME);
        StatTimer_cuda.start();
        int __retval = 0;
        NumShortestPaths_cuda(*nodesWithEdges.begin(), *nodesWithEdges.end(),
                              __retval, infinity, cuda_ctx);
        dga += __retval;
        StatTimer_cuda.stop();
      } else if (personality == CPU)
    #endif
      { 
        galois::do_all(
          galois::iterate(nodesWithEdges),
          NumShortestPaths(infinity, &_graph, dga), 
          galois::loopname("NumShortestPaths"),
          galois::no_stats()
        );
      }

      // read any because destinations need it too
      _graph.sync<writeSource, readAny, Reduce_add_trim, 
                  Broadcast_trim, Bitset_trim>("NumShortestPaths");
      _graph.sync<writeSource, readAny, Reduce_add_to_add, 
                  Broadcast_to_add, Bitset_to_add>("NumShortestPaths");

      // do predecessor decrementing using trim + dependency changes with
      // to_add
      NumShortestPathsChanges::go(_graph);

      iterations++;

      accum_result = dga.reduce();
      // all nodes have everything (loops over all nodes)
    } while (accum_result);
  }



  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    if (src_data.current_length != local_infinity) {
      if (src_data.num_predecessors > 0) {
        for (auto current_edge : graph->edges(src)) {
          GNode dst = graph->getEdgeDst(current_edge);
          auto& dst_data = graph->getData(dst);

          #ifndef __USE_BFS__
          uint32_t edge_weight = graph->getEdgeData(current_edge) + 1;
          #else
          uint32_t edge_weight = 1;
          #endif

          // only operate if a dst flag is set (i.e. no more pred, finalized
          // short paths to take)
          if (dst_data.propogation_flag) {
            // dest on shortest path with this node as successor
            if ((dst_data.current_length + edge_weight) == src_data.current_length) {
              galois::add(src_data.trim, (uint32_t)1);
              galois::add(src_data.to_add, dst_data.num_shortest_paths);

              bitset_trim.set(src);
              bitset_to_add.set(src);

              DGAccumulator_accum += 1;
            }
          }
        }
      }
    }
  }
};

struct FlagPrep {
  const uint32_t &local_infinity;
  Graph* graph;

  FlagPrep(const uint32_t &_local_infinity, Graph* _graph) : 
    local_infinity(_local_infinity), graph(_graph) {}

  void static go(Graph& _graph) {
    const auto& allNodes = _graph.allNodesRange();

    galois::do_all(
      galois::iterate(allNodes.begin(), allNodes.end()),
      FlagPrep{infinity, &_graph}, 
      galois::loopname("FlagPrep"), 
      galois::no_stats()
    );
  }

  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    if (src_data.current_length != local_infinity) {
      if (src_data.num_successors == 0) {
        src_data.propogation_flag = true;
      } else {
        assert(src_data.propogation_flag == false);
      }
    }
  }
};

/* Uses an incremented trim value to decrement the successor: the trim value
 * has to be synchronized across ALL nodes (including mirrors)
 * Use to_add_float to increment the dependency value */
struct DependencyPropChanges {
  const uint32_t &local_infinity;
  Graph* graph;

  DependencyPropChanges(const uint32_t &_local_infinity,
               Graph* _graph) : local_infinity(_local_infinity), graph(_graph){}

  void static go(Graph& _graph) {
    const auto& nodesWithEdges = _graph.allNodesWithEdgesRange();

    #ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) {
        //std::string impl_str("CUDA_DO_ALL_IMPL_DependencyPropChanges");
        std::string impl_str(
          _graph.get_run_identifier("CUDA_DO_ALL_IMPL_DependencyPropChanges")
        );
        galois::StatTimer StatTimer_cuda(impl_str.c_str(), REGION_NAME);
        StatTimer_cuda.start();
        DependencyPropChanges_cuda(
          *nodesWithEdges.begin(), 
          *nodesWithEdges.end(),
          infinity, cuda_ctx
        );
        StatTimer_cuda.stop();
      } else if (personality == CPU)
    #endif
    galois::do_all(
      galois::iterate(nodesWithEdges.begin(), nodesWithEdges.end()),
      DependencyPropChanges{infinity, &_graph}, 
      galois::loopname("DependencyPropChanges"),
      galois::no_stats()
    );
  }

  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    if (src_data.current_length != local_infinity) {
      // increment dependency using to_add_float then reset
      if (src_data.to_add_float > 0.0) {
        src_data.dependency += src_data.to_add_float;
        src_data.to_add_float = 0.0;
      }

      if (src_data.trim > 0) {
        // decrement successor by trim then reset
        assert(src_data.trim <= src_data.num_successors);

        src_data.num_successors = src_data.num_successors - src_data.trim;
        src_data.trim = 0;

        if (src_data.num_successors == 0) {
          src_data.propogation_flag = true;
        }
      }
    }
  }
};

/* Do dependency propogation which is required for betweeness centraility
 * calculation */
struct DependencyPropagation {
  const uint32_t &local_infinity;
  const uint64_t &local_current_src_node;
  Graph* graph;
  galois::DGAccumulator<uint32_t>& DGAccumulator_accum;

  DependencyPropagation(const uint32_t &_local_infinity,
                        const uint64_t &_local_current_src_node,
                        Graph* _graph, galois::DGAccumulator<uint32_t>& dga) : 
      local_infinity(_local_infinity),
      local_current_src_node(_local_current_src_node),
      graph(_graph),
      DGAccumulator_accum(dga) {}

  /* Look at all nodes to do propogation until no more work is done */
  void static go(Graph& _graph, galois::DGAccumulator<uint32_t>& dga) {
    uint32_t iterations = 0;
    uint32_t accum_result;

    do {
      _graph.set_num_iter(iterations);
      dga.reset();

      const auto& nodesWithEdges = _graph.allNodesWithEdgesRange();

    #ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) {
        //std::string impl_str("CUDA_DO_ALL_IMPL_DependencyPropagation");
        std::string impl_str(
          _graph.get_run_identifier("CUDA_DO_ALL_IMPL_DependencyPropagation")
        );
        galois::StatTimer StatTimer_cuda(impl_str.c_str(), REGION_NAME);
        StatTimer_cuda.start();
        int __retval = 0;
        DependencyPropagation_cuda(
          *nodesWithEdges.begin(), *nodesWithEdges.end(),
          __retval, infinity, current_src_node, cuda_ctx
        );
        dga += __retval;
        StatTimer_cuda.stop();
      } else if (personality == CPU)
    #endif
    {
      galois::do_all(
        galois::iterate(nodesWithEdges),
        DependencyPropagation(infinity, current_src_node, &_graph, dga), 
        galois::loopname("DependencyPropagation"),
        galois::no_stats()
      );
    }
                    
      _graph.sync<writeDestination, readSource, Reduce_add_trim, 
                  Broadcast_trim, Bitset_trim>("DependencyPropagation");
      _graph.sync<writeDestination, readSource, Reduce_add_to_add_float, 
                  Broadcast_to_add_float, 
                  Bitset_to_add_float>("DependencyPropagation");

      // use trim + to add to do appropriate changes
      DependencyPropChanges::go(_graph);

      iterations++;
      accum_result = dga.reduce();
    } while (accum_result);
  }


  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    if (src_data.current_length != local_infinity) {
      if (src_data.propogation_flag) {
        assert(src_data.num_successors == 0);
  
        for (auto current_edge : graph->edges(src)) {
          GNode dst = graph->getEdgeDst(current_edge);
  
          // ignore current source node of bc iteration
          if (graph->getGID(dst) == local_current_src_node) {
            continue;
          }
  
          auto& dst_data = graph->getData(dst);

          #ifndef __USE_BFS__
          uint32_t edge_weight = graph->getEdgeData(current_edge) + 1;
          #else
          uint32_t edge_weight = 1;
          #endif

          uint32_t dep = src_data.dependency;

          // I am successor to destination
          if ((dst_data.current_length + edge_weight) == src_data.current_length) {
            galois::atomicAdd(dst_data.trim, (uint32_t)1);
            galois::atomicAdd(dst_data.to_add_float, 
                (((float)dst_data.num_shortest_paths / 
                      (float)src_data.num_shortest_paths) * 
                 (float)(1.0 + dep))
            );

            bitset_to_add_float.set(dst);
            bitset_trim.set(dst);

            DGAccumulator_accum += 1;
          }
        }

        // set flag so that it doesn't propogate its info more than once
        src_data.propogation_flag = false;
      }
    }
  }
};

struct BC {
  Graph* graph;

  BC(Graph* _graph) : graph(_graph){}

  void static go(Graph& _graph, galois::DGAccumulator<uint32_t>& dga){
    uint64_t loop_end = 1;
    bool use_random = false;

    auto random_sources_iterator = random_sources.begin();

    if (!singleSourceBC) {
      if (numberOfSources != 0) {
        loop_end = numberOfSources;
        use_random = true;
      } else {
        loop_end = _graph.globalSize();
      }
    }

    for (uint64_t i = 0; i < loop_end; i++) {
      if (singleSourceBC) {
        // only 1 source; specified start source in command line
        assert(loop_end == 1);
        galois::gDebug("This is single source node BC");
        current_src_node = startSource;
      } else if (use_random) {
        // number of sources non-zero, so use random sources
        current_src_node = *random_sources_iterator;
        random_sources_iterator++;
      } else {
        // all sources
        current_src_node = i;
      }

      //galois::gDebug("Current source node for BC is ", current_src_node);

      #ifndef NDEBUG
      if (galois::runtime::getSystemNetworkInterface().ID == 0) {
        if (i % 5000 == 0) {
          galois::gPrint("SSSP source node #", i, "\n");
        }
      }
      #endif

      _graph.set_num_iter(0);

      // reset the graph aside from the between-cent measure
      InitializeIteration::go(_graph);
      //galois::gDebug("Init done");

      // get SSSP on the current graph
      SSSP::go(_graph, dga);
      //galois::gDebug("SSSP done");

      _graph.set_num_iter(0);

      // calculate the succ/pred for all nodes in the SSSP DAG
      PredAndSucc::go(_graph);
      //galois::gDebug("PredAndSucc done");

      // calculate the number of shortest paths for each node
      NumShortestPaths::go(_graph, dga);
      //galois::gDebug("NumShortestPaths done");

      _graph.set_num_iter(0);

      // setup flags for dep prop round
      FlagPrep::go(_graph);

      // do between-cent calculations for this iteration 
      DependencyPropagation::go(_graph, dga);
      //galois::gDebug("DepPropagation done");

      _graph.set_num_iter(0);

      const auto& nodesWithEdges = _graph.allNodesWithEdgesRange();

      // finally, since dependencies are finalized for this round at this 
      // point, add them to the betweeness centrality measure on each node
    #ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) {
        //std::string impl_str("CUDA_DO_ALL_IMPL_BC");
        std::string impl_str(_graph.get_run_identifier("CUDA_DO_ALL_IMPL_BC"));
        galois::StatTimer StatTimer_cuda(impl_str.c_str(), REGION_NAME);
        StatTimer_cuda.start();
        BC_cuda(*nodesWithEdges.begin(), *nodesWithEdges.end(), cuda_ctx);
        StatTimer_cuda.stop();
      } else if (personality == CPU)
    #endif
      // nodes with edges because those include masters
      galois::do_all(
        galois::iterate(nodesWithEdges.begin(), nodesWithEdges.end()),
        BC(&_graph), 
        galois::loopname("BC"));
    }
  }

  /* adds dependency measure to BC measure (dependencies should be finalized,
   * i.e. no unprocessed successors on the node) */
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    if (src_data.dependency > 0) {
      galois::add(src_data.betweeness_centrality, src_data.dependency);
      src_data.dependency = 0;
    }

  }
};

/******************************************************************************/
/* Sanity check */
/******************************************************************************/

struct Sanity {
  Graph* graph;

  static float current_max;
  static float current_min;

  galois::DGAccumulator<float>& DGAccumulator_max;
  galois::DGAccumulator<float>& DGAccumulator_min;
  galois::DGAccumulator<double>& DGAccumulator_sum;

  Sanity(Graph* _graph,
      galois::DGAccumulator<float>& _DGAccumulator_max,
      galois::DGAccumulator<float>& _DGAccumulator_min,
      galois::DGAccumulator<double>& _DGAccumulator_sum
  ) : 
    graph(_graph),
    DGAccumulator_max(_DGAccumulator_max),
    DGAccumulator_min(_DGAccumulator_min),
    DGAccumulator_sum(_DGAccumulator_sum) {}

  void static go(Graph& _graph,
    galois::DGAccumulator<float>& DGA_max,
    galois::DGAccumulator<float>& DGA_min,
    galois::DGAccumulator<double>& DGA_sum
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

    galois::do_all(galois::iterate(_graph.allNodesRange().begin(), _graph.allNodesRange().end()),
                   Sanity(
                     &_graph,
                     DGA_max,
                     DGA_min,
                     DGA_sum
                   ), 
                   galois::loopname("Sanity"));

    DGA_max = current_max;
    DGA_min = current_min;

    float max_bc = DGA_max.reduce_max();
    float min_bc = DGA_min.reduce_min();
    double bc_sum = DGA_sum.reduce();

    // Only node 0 will print data
    if (galois::runtime::getSystemNetworkInterface().ID == 0) {
      printf("Max BC is %f\n", max_bc);
      printf("Min BC is %f\n", min_bc);
      printf("BC sum is %f\n", bc_sum);
    }
  }
  
  /* Gets the max, min rank from all owned nodes and
   * also the sum of ranks */
  void operator()(GNode src) const {
    NodeData& sdata = graph->getData(src);

    if (graph->isOwned(graph->getGID(src))) {
      if (current_max < sdata.betweeness_centrality) {
        current_max = sdata.betweeness_centrality;
      }

      if (current_min > sdata.betweeness_centrality) {
        current_min = sdata.betweeness_centrality;
      }

      DGAccumulator_sum += sdata.betweeness_centrality;
    }
  }
};
float Sanity::current_max = 0;
float Sanity::current_min = std::numeric_limits<float>::max() / 4;
 
/******************************************************************************/
/* Main method for running */
/******************************************************************************/

constexpr static const char* const name = "Betweeness Centrality - "
                                          "Pull Distributed Heterogeneous.";
constexpr static const char* const desc = "Betweeness Centrality Pull on "
                                          "Distributed Galois.";
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

  #ifndef __USE_BFS__

  #ifdef __GALOIS_HET_CUDA__
  Graph* h_graph = distGraphInitialization<NodeData, unsigned int,
                                           false>(&cuda_ctx);
  #else
  Graph* h_graph = distGraphInitialization<NodeData, unsigned int,
                                           false>();
  #endif

  #else

  #ifdef __GALOIS_HET_CUDA__
  Graph* h_graph = distGraphInitialization<NodeData, void, false>(&cuda_ctx);
  #else
  Graph* h_graph = distGraphInitialization<NodeData, void, false>();
  #endif

  #endif

  // random num generate for sources
  std::minstd_rand0 r_generator;
  r_generator.seed(100);
  std::uniform_int_distribution<uint64_t> r_dist(0, h_graph->globalSize() - 1);

  if (numberOfSources != 0) {
    //random_sources.insert(startSource);
    while (random_sources.size() < numberOfSources) {
      random_sources.insert(r_dist(r_generator));
    }
  }

  #ifndef NDEBUG
  int counter = 0;
  for (auto i = random_sources.begin(); i != random_sources.end(); i++) {
    printf("Source #%d: %lu\n", counter, *i);
    counter++;
  }
  #endif

  bitset_to_add.resize(h_graph->size());
  bitset_to_add_float.resize(h_graph->size());
  bitset_num_shortest_paths.resize(h_graph->size());
  bitset_num_successors.resize(h_graph->size());
  bitset_num_predecessors.resize(h_graph->size());
  bitset_trim.resize(h_graph->size());
  bitset_current_length.resize(h_graph->size());
  bitset_propogation_flag.resize(h_graph->size());
  bitset_dependency.resize(h_graph->size());

  galois::gPrint("[", net.ID, "] InitializeGraph::go called\n");

  galois::StatTimer StatTimer_graph_init("TIMER_GRAPH_INIT", REGION_NAME);
  StatTimer_graph_init.start();
    InitializeGraph::go((*h_graph));
  StatTimer_graph_init.stop();
  galois::runtime::getHostBarrier().wait();

  // shared DG accumulator among all steps
  galois::DGAccumulator<uint32_t> dga;

  // sanity dg accumulators
  galois::DGAccumulator<float> dga_max;
  galois::DGAccumulator<float> dga_min;
  galois::DGAccumulator<double> dga_sum;

  for (auto run = 0; run < numRuns; ++run) {
    galois::gPrint("[", net.ID, "] BC::go run ", run, " called\n");
    std::string timer_str("TIMER_" + std::to_string(run));
    galois::StatTimer StatTimer_main(timer_str.c_str(), REGION_NAME);

    StatTimer_main.start();
      BC::go(*h_graph, dga);
    StatTimer_main.stop();

    Sanity::current_max = 0;
    Sanity::current_min = std::numeric_limits<float>::max() / 4;

    Sanity::go(
      *h_graph,
      dga_max,
      dga_min,
      dga_sum
    );

    // re-init graph for next run
    if ((run + 1) != numRuns) {
      galois::runtime::getHostBarrier().wait();
      (*h_graph).set_num_run(run + 1);

    #ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) { 
        bitset_to_add_reset_cuda(cuda_ctx);
        bitset_to_add_float_reset_cuda(cuda_ctx);
        bitset_num_shortest_paths_reset_cuda(cuda_ctx);
        bitset_num_successors_reset_cuda(cuda_ctx);
        bitset_num_predecessors_reset_cuda(cuda_ctx);
        bitset_trim_reset_cuda(cuda_ctx);
        bitset_current_length_reset_cuda(cuda_ctx);
        bitset_old_length_reset_cuda(cuda_ctx);
        bitset_propogation_flag_reset_cuda(cuda_ctx);
        bitset_dependency_reset_cuda(cuda_ctx);
      } else
    #endif
      {
      bitset_to_add.reset();
      bitset_to_add_float.reset();
      bitset_num_shortest_paths.reset();
      bitset_num_successors.reset();
      bitset_num_predecessors.reset();
      bitset_trim.reset();
      bitset_current_length.reset();
      bitset_propogation_flag.reset();
      bitset_dependency.reset();
      }

      InitializeGraph::go((*h_graph));
      galois::runtime::getHostBarrier().wait();
    }
  }

  StatTimer_total.stop();

  // Verify, i.e. print out graph data for examination
  if (verify) {
    char *v_out = (char*)malloc(40);
    #ifdef __GALOIS_HET_CUDA__
    if (personality == CPU) { 
    #endif
      for (auto ii = (*h_graph).masterNodesRange().begin(); 
                ii != (*h_graph).masterNodesRange().end(); 
                ++ii) {
          // outputs betweenness centrality
        sprintf(v_out, "%lu %.9f\n", (*h_graph).getGID(*ii),
                (*h_graph).getData(*ii).betweeness_centrality);
        galois::runtime::printOutput(v_out);
      }
    #ifdef __GALOIS_HET_CUDA__
    } else if (personality == GPU_CUDA) {
      for (auto ii = (*h_graph).masterNodesRange().begin(); 
                ii != (*h_graph).masterNodesRange().end(); 
                ++ii) {
        galois::runtime::printOutput(v_out);
        memset(v_out, '\0', 40);
      }
    }
    #endif
    free(v_out);
  }

  return 0;
}
