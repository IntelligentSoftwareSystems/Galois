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

// TODO rebuild gpu code (new things added)

/******************************************************************************/
/* Sync code/calls was manually written, not compiler generated */
/******************************************************************************/

#include <iostream>
#include <limits>
#include <random>

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

static const char* const name = "Betweeness Centrality - "
                                "Distributed Heterogeneous.";
static const char* const desc = "Betweeness Centrality on Distributed Galois.";
static const char* const url = 0;

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
  std::atomic<uint32_t> current_length;

  uint32_t old_length;

  // Betweeness centrality vars
  uint32_t num_shortest_paths;
  uint32_t num_successors;
  std::atomic<uint32_t> num_predecessors;
  std::atomic<uint32_t> trim;
  std::atomic<uint32_t> to_add;

  float to_add_float;
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
    auto& allNodes = _graph.allNodesRange();

    #ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) {
        std::string impl_str(
          //_graph.get_run_identifier("CUDA_DO_ALL_IMPL_InitializeGraph")
          "CUDA_DO_ALL_IMPL_InitializeGraph"
        );
        galois::StatTimer StatTimer_cuda(impl_str.c_str());
        StatTimer_cuda.start();
        InitializeGraph_cuda(*allNodes.begin(), *allNodes.end(), cuda_ctx);
        StatTimer_cuda.stop();
      } else if (personality == CPU)
    #endif
    galois::do_all(
      allNodes.begin(), allNodes.end(), 
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
    auto& allNodes = _graph.allNodesRange();

    #ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) {
        std::string impl_str(
          //_graph.get_run_identifier("CUDA_DO_ALL_IMPL_InitializeIteration")
          "CUDA_DO_ALL_IMPL_InitializeIteration"
        );

        galois::StatTimer StatTimer_cuda(impl_str.c_str());
        StatTimer_cuda.start();
        InitializeIteration_cuda(
          *allNodes.begin(), *allNodes.end(),
          infinity, current_src_node, cuda_ctx
        );
        StatTimer_cuda.stop();
      } else if (personality == CPU)
    #endif
    galois::do_all(
      allNodes.begin(), allNodes.end(), 
      InitializeIteration{infinity, current_src_node, &_graph},
      galois::loopname("InitializeIteration"), 
      //galois::loopname(_graph.get_run_identifier("InitializeIteration").c_str()), 
      galois::timeit(),
      galois::no_stats()
    );
  }

  /* Functor passed into the Galois operator to carry out reset of node data
   * (aside from betweeness centrality measure */
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    bool is_source = graph->getGID(src) == local_current_src_node;

    if (!is_source) {
      src_data.current_length = local_infinity;
      src_data.old_length = local_infinity;
      src_data.num_shortest_paths = 0;
      src_data.propogation_flag = false;
    } else {
      src_data.current_length = 0;
      src_data.old_length = 0; 
      src_data.num_shortest_paths = 1;
      src_data.propogation_flag = true;
    }
    src_data.num_predecessors = 0;
    src_data.num_successors = 0;
    src_data.dependency = 0;

    assert(src_data.trim.load() == 0);
    assert(src_data.to_add.load() == 0);
    assert(src_data.to_add_float == 0);
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

    #ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) {
        std::string impl_str(
          //_graph.get_run_identifier("CUDA_DO_ALL_IMPL_FirstIterationSSSP")
          "CUDA_DO_ALL_IMPL_SSSP"
        );
        galois::StatTimer StatTimer_cuda(impl_str.c_str());
        StatTimer_cuda.start();
        FirstIterationSSSP_cuda(__begin, __end, cuda_ctx);
        StatTimer_cuda.stop();
      } else if (personality == CPU)
    #endif
    galois::do_all(
      boost::make_counting_iterator(__begin), 
      boost::make_counting_iterator(__end), 
      FirstIterationSSSP(&_graph),
      galois::loopname("SSSP"),
      //galois::loopname(_graph.get_run_identifier("FirstIterationSSSP").c_str()),
      galois::timeit(),
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
    for (auto current_edge = graph->edge_begin(src), 
              end_edge = graph->edge_end(src); 
         current_edge != end_edge; 
         ++current_edge) {
      GNode dst = graph->getEdgeDst(current_edge);

      if (src == dst) {
        continue;
      }

      auto& dst_data = graph->getData(dst);

      #ifndef __USE_BFS__
      // For SSSP (uses the edge weight; you need to change the graph edge
      // type as well in the declaration above)
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

    auto& nodesWithEdges = _graph.allNodesWithEdgesRange();

    do {
      _graph.set_num_iter(iterations);
      dga.reset();

    #ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) {
        //std::string impl_str("CUDA_DO_ALL_IMPL_SSSP_0");
        std::string impl_str(
          //_graph.get_run_identifier("CUDA_DO_ALL_IMPL_SSSP")
          "CUDA_DO_ALL_IMPL_SSSP"
        );
        galois::StatTimer StatTimer_cuda(impl_str.c_str());
        StatTimer_cuda.start();
        int __retval = 0;
        SSSP_cuda(*nodesWithEdges.begin(), *nodesWithEdges.end(), __retval, cuda_ctx);
        dga += __retval;
        StatTimer_cuda.stop();
      } else if (personality == CPU)
    #endif
      {
      galois::do_all_local(
        nodesWithEdges,
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

      for (auto current_edge = graph->edge_begin(src), 
                end_edge = graph->edge_end(src); 
           current_edge != end_edge; 
           ++current_edge) {
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

/* Struct to get pred and succ on the SSSP DAG */
struct PredAndSucc {
  const uint32_t &local_infinity;
  Graph* graph;

  PredAndSucc(const uint32_t &_local_infinity, Graph* _graph) : 
      local_infinity(_local_infinity), graph(_graph) {}

  void static go(Graph& _graph){
    auto& nodesWithEdges = _graph.allNodesWithEdgesRange();
    
    #ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) {
        //std::string impl_str("CUDA_DO_ALL_IMPL_PredAndSucc");
        std::string impl_str(
          //_graph.get_run_identifier("CUDA_DO_ALL_IMPL_PredAndSucc")
          "CUDA_DO_ALL_IMPL_PredAndSucc"
        );
        galois::StatTimer StatTimer_cuda(impl_str.c_str());
        StatTimer_cuda.start();
        PredAndSucc_cuda(*nodesWithEdges.begin(), *nodesWithEdges.end(),
                         infinity, cuda_ctx);
        StatTimer_cuda.stop();
      } else if (personality == CPU)
    #endif
    {
    galois::do_all_local(
      nodesWithEdges,
      PredAndSucc(infinity, &_graph), 
      galois::loopname("PredAndSucc"),
      //galois::loopname(_graph.get_run_identifier("PredAndSucc").c_str()),
      galois::timeit(),
      galois::no_stats()
    );
    }

    // sync for use in NumShortPath calculation
    _graph.sync<writeDestination, readSource, Reduce_add_num_predecessors, 
                Broadcast_num_predecessors, 
                Bitset_num_predecessors>("PredAndSucc");

    // sync now for later DependencyPropogation use 
    _graph.sync<writeSource, readSource, Reduce_add_num_successors, 
                Broadcast_num_successors, 
                Bitset_num_successors>("PredAndSucc");
  }

  /* Summary:
   * Look at outgoing edges; see if dest is on a shortest path from src node.
   * If it is, increment the number of successors on src by 1 and
   * increment # of pred on dest by 1 
   */
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    if (src_data.current_length != local_infinity) {
      for (auto current_edge = graph->edge_begin(src),  
                end_edge = graph->edge_end(src); 
           current_edge != end_edge; 
           ++current_edge) {
        GNode dst = graph->getEdgeDst(current_edge);
        // ignore self loops
        if (dst == src) {
          continue;
        }

        auto& dst_data = graph->getData(dst);

        #ifndef __USE_BFS__
        // SSSP
        uint32_t edge_weight = graph->getEdgeData(current_edge) + 1;
        #else
        // BFS
        uint32_t edge_weight = 1;
        #endif

        if ((src_data.current_length + edge_weight) == dst_data.current_length) {
          // dest on shortest path with this node as predecessor
          galois::add(src_data.num_successors, (unsigned int)1);
          galois::atomicAdd(dst_data.num_predecessors, (unsigned int)1);

          bitset_num_successors.set(src);
          bitset_num_predecessors.set(dst);
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
    // DO NOT DO A BITSET RESET HERE BECAUSE IT WILL BE REUSED BY THE NEXT STEP
    // (updates to trim and pred are on the same nodes)
    auto& nodesWithEdges = _graph.allNodesWithEdgesRange();

    #ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) {
        //std::string impl_str("CUDA_DO_ALL_IMPL_NumShortestPathsChanges");
        std::string impl_str(
          //_graph.get_run_identifier("CUDA_DO_ALL_IMPL_NumShortestPathsChanges")
          "CUDA_DO_ALL_IMPL_NumShortestPathsChanges"
        );
        galois::StatTimer StatTimer_cuda(impl_str.c_str());
        StatTimer_cuda.start();
        // TODO local infinity needed
        NumShortestPathsChanges_cuda(
          *nodesWithEdges.begin(),
          *nodesWithEdges.end(),
          cuda_ctx
        );
        StatTimer_cuda.stop();
      } else if (personality == CPU)
    #endif
    galois::do_all(
      nodesWithEdges.begin(), nodesWithEdges.end(), 
      NumShortestPathsChanges{infinity, &_graph}, 
      galois::loopname("NumShortestPathsChanges"), 
      //galois::loopname(_graph.get_run_identifier("NumShortestPathsChanges").c_str()), 
      galois::timeit(),
      galois::no_stats()
    );

    // predecessors does not require syncing as syncing trim accomplishes the
    // same effect; as a result, flags are synced as well on sources
    // additionally, all sources will have trim from last sync, meaning all
    // sources will reset trim to 0 
    // Since we only read trim at source, this is sufficient to "sync"
  }

  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);
    
    if (src_data.current_length != local_infinity) {
      // decrement predecessor by trim then reset
      if (src_data.trim > 0) {
        // TODO use a Galois assert here? this is extremely important
        if (src_data.trim > src_data.num_predecessors) {
          printf("src is %lu trim is %u, pred is %u\n", graph->L2G(src),
                 src_data.trim.load(), src_data.num_predecessors.load());
          assert(src_data.trim <= src_data.num_predecessors); 
        }

        src_data.num_predecessors = src_data.num_predecessors - src_data.trim;
        src_data.trim = 0;

        // if I hit 0 predecessors after trim, set the flag to true (i.e. says
        // I need to propogate my value)
        if (src_data.num_predecessors == 0) {
          assert(!src_data.propogation_flag);
          src_data.propogation_flag = true;
        }
      }

      // increment num_shortest_paths by to_add then reset
      if (src_data.to_add > 0) {
        src_data.num_shortest_paths += src_data.to_add;
        src_data.to_add = 0;

        // this bitset is used in the NumShortestPaths go method to 
        // sync to destinations
        bitset_num_shortest_paths.set(src);
      }
    }

  }
};

/* Calculate the number of shortest paths for each node */
struct NumShortestPaths {
  const uint32_t &local_infinity;
  const uint64_t local_current_src_node;

  Graph* graph;
  galois::DGAccumulator<uint32_t>& DGAccumulator_accum;

  NumShortestPaths(const uint32_t &_local_infinity,
                   const uint64_t &_local_current_src_node,
                   Graph* _graph, galois::DGAccumulator<uint32_t>& dga) : 
     local_infinity(_local_infinity), 
     local_current_src_node(_local_current_src_node), 
     graph(_graph), DGAccumulator_accum(dga) {}

  void static go(Graph& _graph, galois::DGAccumulator<uint32_t>& dga) {
    uint32_t iterations = 0;
    uint32_t accum_result;

    auto& nodesWithEdges = _graph.allNodesWithEdgesRange();

    do {
      _graph.set_num_iter(iterations);
      dga.reset();

    #ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) {
        //std::string impl_str("CUDA_DO_ALL_IMPL_NumShortestPaths");
        std::string impl_str(
          //_graph.get_run_identifier("CUDA_DO_ALL_IMPL_NumShortestPaths")
          "CUDA_DO_ALL_IMPL_NumShortestPaths"
        );
        galois::StatTimer StatTimer_cuda(impl_str.c_str());
        StatTimer_cuda.start();
        int __retval = 0;
        NumShortestPaths_cuda(*nodesWithEdges.begin(), *nodesWithEdges.end(),
                              __retval, infinity, cuda_ctx);
        dga += __retval;
        StatTimer_cuda.stop();
      } else if (personality == CPU)
    #endif
      { 
        galois::do_all_local(
          nodesWithEdges,
          NumShortestPaths(infinity, current_src_node, &_graph, dga), 
          galois::loopname("NumShortestPaths"),
          //galois::loopname(_graph.get_run_identifier("NumShortestPaths").c_str()),
          galois::timeit(),
          galois::no_stats()
        );
      }

      // sync to_adds and trim on source
      _graph.sync<writeDestination, readSource, Reduce_add_trim, 
                  Broadcast_trim, Bitset_trim>("NumShortestPaths");
      _graph.sync<writeDestination, readSource, Reduce_add_to_add, 
                  Broadcast_to_add, Bitset_to_add>("NumShortestPaths");

      // do predecessor decrementing using trim + dependency changes with
      // to_add
      NumShortestPathsChanges::go(_graph);

      iterations++;

      accum_result = dga.reduce();

      // sync num_short_paths on dest (will be sync'd on source
      // already, i.e. all sources should already have the correct value)
      if (!accum_result) {
        _graph.sync<writeSource, readDestination, Reduce_set_num_shortest_paths, 
                    Broadcast_num_shortest_paths, 
                    Bitset_num_shortest_paths>("NumShortestPaths");

      }
    } while (accum_result);
  }

  /* If a source has no more predecessors, then its shortest path value is
   * complete.
   *
   * Propogate the shortest path value through all outgoing edges where this
   * source is a predecessor in the DAG, then
   * set a flag saying that we should not propogate it any more (otherwise you
   * send extra).
   *
   * Additionally, decrement the predecessor field on the destination nodes of
   * the outgoing edges.
   */
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    if (src_data.current_length != local_infinity) {
      // can do a num succ check for optimization
      //if (src_data.propogation_flag && src_data.num_successors > 0) {
      if (src_data.propogation_flag) {
        for (auto current_edge = graph->edge_begin(src), 
                  end_edge = graph->edge_end(src); 
             current_edge != end_edge; 
             ++current_edge) {
          GNode dst = graph->getEdgeDst(current_edge);

          // ignore self loops
          if (dst == src) {
            continue;
          }
          
          // ignore "shortest path" to source
          if (graph->L2G(dst) == local_current_src_node) {
            continue;
          }

          auto& dst_data = graph->getData(dst);

          #ifndef __USE_BFS__
          // SSSP
          uint32_t edge_weight = graph->getEdgeData(current_edge) + 1;
          #else
          // BFS
          uint32_t edge_weight = 1;
          #endif

          uint32_t paths_to_add = src_data.num_shortest_paths;

          assert(paths_to_add >= 1);

          if ((src_data.current_length + edge_weight) == dst_data.current_length) {
            // need to add my num_short_paths to dest
            galois::atomicAdd(dst_data.to_add, paths_to_add);
            // increment dst trim so it can decrement predecessor
            galois::atomicAdd(dst_data.trim, (unsigned int)1);

            bitset_to_add.set(dst);
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

/** 
 * Make sure all flags are false except for nodes with 0 successors and sync 
 * flag
 */
struct PropogationFlagUpdate {
  const uint32_t &local_infinity;
  Graph* graph;

  PropogationFlagUpdate(const uint32_t &_local_infinity, Graph* _graph) : 
    local_infinity(_local_infinity), graph(_graph) { }

  void static go(Graph& _graph) {
    auto& nodesWithEdges = _graph.allNodesWithEdgesRange();

    // TODO gpu code

    galois::do_all(
      nodesWithEdges.begin(), nodesWithEdges.end(),
      PropogationFlagUpdate(infinity, &_graph), 
      galois::loopname("PropogationFlagUpdate"),
      galois::timeit(),
      galois::no_stats()
    );

    // note that only nodes with succ == 0 will have their flags sync'd
    // by this call (through bitset; only set for those cases); the others
    // do not need to be sync'd as they will (or should) all be false already
    _graph.sync<writeSource, readDestination, Reduce_set_propogation_flag, 
                Broadcast_propogation_flag, 
                Bitset_propogation_flag>("PropogationFlagUpdate");
  }

  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    // this shouldn't print 
    if (src_data.num_predecessors != 0) {
      printf("[%d] WARNING node %lu with length %u, short paths %u\n", graph->id,
             graph->L2G(src), src_data.current_length.load(), src_data.num_shortest_paths);
    }

    assert(src_data.num_predecessors == 0);

    if (src_data.current_length != local_infinity) {
      if (src_data.num_successors == 0) {
        src_data.propogation_flag = true;
        bitset_propogation_flag.set(src);
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
    auto& nodesWithEdges = _graph.allNodesWithEdgesRange();

    #ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) {
        //std::string impl_str("CUDA_DO_ALL_IMPL_DependencyPropChanges");
        std::string impl_str(
          //_graph.get_run_identifier("CUDA_DO_ALL_IMPL_DependencyPropChanges")
          "CUDA_DO_ALL_IMPL_DependencyPropChanges"
        );
        galois::StatTimer StatTimer_cuda(impl_str.c_str());
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
      nodesWithEdges.begin(), nodesWithEdges.end(),
      DependencyPropChanges{infinity, &_graph}, 
      galois::loopname("DependencyPropChanges"),
      //galois::loopname(_graph.get_run_identifier("DependencyPropChanges").c_str()),
      galois::timeit(),
      galois::no_stats()
    );

    // need reduce set for flag
    _graph.sync<writeSource, readDestination, Reduce_set_propogation_flag, 
                Broadcast_propogation_flag,
                //Bitset_propogation_flag>("DependencyPropChanges_prop_flag");
                Bitset_propogation_flag>("DependencyPropChanges");
  }

  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    if (src_data.current_length != local_infinity) {
      // increment dependency using to_add_float then reset
      if (src_data.to_add_float > 0.0) {
        src_data.dependency += src_data.to_add_float;
        src_data.to_add_float = 0.0;

        // used in DependencyPropogation's go method
        bitset_dependency.set(src);
      }

      if (src_data.num_successors == 0 && src_data.propogation_flag) {
        // has had dependency back-propogated; reset the flag
        assert(src_data.trim == 0);
        src_data.propogation_flag = false;
        bitset_propogation_flag.set(src);
      } else if (src_data.trim > 0) {
        // decrement successor by trim then reset
        assert(src_data.trim <= src_data.num_successors);

        src_data.num_successors = src_data.num_successors - src_data.trim;
        src_data.trim = 0;

        if (src_data.num_successors == 0) {
          assert(!src_data.propogation_flag);
          src_data.propogation_flag = true;
          bitset_propogation_flag.set(src);
        }
      }
    }
  }
};

/* Do dependency propogation which is required for betweeness centraility
 * calculation */
struct DependencyPropogation {
  const uint32_t &local_infinity;
  const uint64_t &local_current_src_node;
  Graph* graph;
  galois::DGAccumulator<uint32_t>& DGAccumulator_accum;

  DependencyPropogation(const uint32_t &_local_infinity,
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

      auto& nodesWithEdges = _graph.allNodesWithEdgesRange();

    #ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) {
        //std::string impl_str("CUDA_DO_ALL_IMPL_DependencyPropogation");
        std::string impl_str(
          //_graph.get_run_identifier("CUDA_DO_ALL_IMPL_DependencyPropogation")
          "CUDA_DO_ALL_IMPL_DependencyPropogation"
        );
        galois::StatTimer StatTimer_cuda(impl_str.c_str());
        StatTimer_cuda.start();
        int __retval = 0;
        DependencyPropogation_cuda(
          *nodesWithEdges.begin(), *nodesWithEdges.end(),
          __retval, infinity, current_src_node, cuda_ctx
        );
        dga += __retval;
        StatTimer_cuda.stop();
      } else if (personality == CPU)
    #endif
    {
      galois::do_all_local(
        nodesWithEdges,
        DependencyPropogation(infinity, current_src_node, &_graph, dga), 
        galois::loopname("DependencyPropogation"),
        //galois::loopname(_graph.get_run_identifier("DependencyPropogation").c_str()),
        galois::timeit(),
        galois::no_stats()
      );
    }
                    

      _graph.sync<writeSource, readSource, Reduce_add_trim, 
                  Broadcast_trim, Bitset_trim>("DependencyPropogation");
      _graph.sync<writeSource, readSource, Reduce_add_to_add_float, 
                  Broadcast_to_add_float, 
                  Bitset_to_add_float>("DependencyPropogation");

      // use trim + to add to do appropriate changes
      DependencyPropChanges::go(_graph);

      iterations++;
      accum_result = dga.reduce();

      // while the loop still goes on...
      if (accum_result) {
        // sync dependency on dest; source should all have same dep
        _graph.sync<writeSource, readDestination, Reduce_set_dependency,
                    Broadcast_dependency, 
                    Bitset_dependency>("DependencyPropogation");
      } 
    } while (accum_result);
  }

  /* Summary:
   * if we have outgoing edges...
   * for each node, check if dest of edge has no successors + check if on 
   * shortest path with src as predeccesor
   *
   * if yes, then decrement src successors by 1 + grab dest delta + dest num 
   * shortest * paths and use it to increment src own delta
   **/
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    // IGNORE THE SOURCE NODE OF THIS CURRENT ITERATION OF SSSP
    // + do not redo computation if src has no successors left
    if (src_data.current_length != local_infinity) {
      if (src_data.num_successors > 0) {
        if (graph->getGID(src) != local_current_src_node) {
          for (auto current_edge = graph->edge_begin(src), 
                    end_edge = graph->edge_end(src); 
               current_edge != end_edge; 
               ++current_edge) {
            GNode dst = graph->getEdgeDst(current_edge);

            // ignore self loops
            if (dst == src) {
              continue;
            }

            if (graph->L2G(dst) == local_current_src_node) {
              continue;
            }

            auto& dst_data = graph->getData(dst);

            #ifndef __USE_BFS__
            // SSSP
            uint32_t edge_weight = graph->getEdgeData(current_edge) + 1;
            #else
            // BFS
            uint32_t edge_weight = 1;
            #endif

            // only operate if a dst flag is set (i.e. no more succ, finalized
            // dependency to take)
            if (dst_data.propogation_flag) {
              // dest on shortest path with this node as predecessor
              if ((src_data.current_length + edge_weight) == dst_data.current_length) {
                // increment my trim for later use to decrement successor
                galois::atomicAdd(src_data.trim, (unsigned int)1);

                assert(src_data.num_shortest_paths != 0);
                assert(dst_data.num_shortest_paths != 0);

                // update my to_add_float (which is later used to update dependency)
                galois::add(src_data.to_add_float, 
                (((float)src_data.num_shortest_paths / 
                      (float)dst_data.num_shortest_paths) * 
                 (float)(1.0 + dst_data.dependency)));

                bitset_trim.set(src);
                bitset_to_add_float.set(src);

                DGAccumulator_accum += 1;
              }
            }
          }
        } else {
          // this is source of this iteration's of sssp/bfs; reset num succ to 0
          // so loop isn't entered again
          src_data.num_successors = 0;
        }
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
        loop_end = _graph.totalNodes;
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
      if (_graph.id == 0) {
        if (i % 5000 == 0) {
          std::cout << "SSSP source node " << i << "\n";
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

      PropogationFlagUpdate::go(_graph);

      // do between-cent calculations for this iteration 
      DependencyPropogation::go(_graph, dga);
      //galois::gDebug("DepPropogation done");

      _graph.set_num_iter(0);

      auto& nodesWithEdges = _graph.allNodesWithEdgesRange();

      // finally, since dependencies are finalized for this round at this 
      // point, add them to the betweeness centrality measure on each node
    #ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) {
        //std::string impl_str(_graph.get_run_identifier("CUDA_DO_ALL_IMPL_BC"));
        std::string impl_str("CUDA_DO_ALL_IMPL_BC");
        galois::StatTimer StatTimer_cuda(impl_str.c_str());
        StatTimer_cuda.start();
        BC_cuda(*nodesWithEdges.begin(), *nodesWithEdges.end(), cuda_ctx);
        StatTimer_cuda.stop();
      } else if (personality == CPU)
    #endif
      galois::do_all(
        nodesWithEdges.begin(), nodesWithEdges.end(), 
        BC(&_graph), 
        galois::loopname("BC"),
        //galois::loopname(_graph.get_run_identifier("BC").c_str()),
        galois::timeit(),
        galois::no_stats()
      );
    }
  }

  /* adds dependency measure to BC measure (dependencies should be finalized,
   * i.e. no unprocessed successors on the node) */
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    if (src_data.dependency > 0) {
      galois::add(src_data.betweeness_centrality, src_data.dependency);
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

    galois::do_all(_graph.begin(), _graph.end(), 
                   Sanity(
                     &_graph,
                     DGA_max,
                     DGA_min,
                     DGA_sum
                   ), 
                   galois::loopname("Sanity"),
                   galois::no_stats());

    DGA_max = current_max;
    DGA_min = current_min;

    float max_bc = DGA_max.reduce_max();
    float min_bc = DGA_min.reduce_min();
    double bc_sum = DGA_sum.reduce();

    // Only node 0 will print data
    if (_graph.id == 0) {
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

int main(int argc, char** argv) {
  galois::DistMemSys G;
  DistBenchStart(argc, argv, name, desc, url);

  auto& net = galois::runtime::getSystemNetworkInterface();
  if (net.ID == 0) {
    galois::runtime::reportParam("BC", "Max Iterations", 
                                (unsigned long)maxIterations);
  }

  galois::StatTimer StatTimer_total("TIMER_TOTAL");

  StatTimer_total.start();

  #ifndef __USE_BFS__

  #ifdef __GALOIS_HET_CUDA__
  Graph* h_graph = distGraphInitialization<NodeData, unsigned int>(&cuda_ctx);
  #else
  Graph* h_graph = distGraphInitialization<NodeData, unsigned int>();
  #endif

  #else

  #ifdef __GALOIS_HET_CUDA__
  Graph* h_graph = distGraphInitialization<NodeData, void>(&cuda_ctx);
  #else
  Graph* h_graph = distGraphInitialization<NodeData, void>();
  #endif

  #endif

  // random num generate for sources
  std::minstd_rand0 r_generator;
  r_generator.seed(100);
  std::uniform_int_distribution<uint64_t> r_dist(0, h_graph->totalNodes - 1);

  if (numberOfSources != 0) {
    // uncomment this to have srcnodeid included as well
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

  bitset_to_add.resize(h_graph->get_local_total_nodes());
  bitset_to_add_float.resize(h_graph->get_local_total_nodes());
  bitset_num_shortest_paths.resize(h_graph->get_local_total_nodes());
  bitset_num_successors.resize(h_graph->get_local_total_nodes());
  bitset_num_predecessors.resize(h_graph->get_local_total_nodes());
  bitset_trim.resize(h_graph->get_local_total_nodes());
  bitset_current_length.resize(h_graph->get_local_total_nodes());
  bitset_propogation_flag.resize(h_graph->get_local_total_nodes());
  bitset_dependency.resize(h_graph->get_local_total_nodes());

  std::cout << "[" << net.ID << "] InitializeGraph::go called\n";

  galois::StatTimer StatTimer_graph_init("TIMER_GRAPH_INIT");
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
    std::cout << "[" << net.ID << "] BC::go run " << run << " called\n";
    std::string timer_str("TIMER_" + std::to_string(run));
    galois::StatTimer StatTimer_main(timer_str.c_str());

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
      (*h_graph).reset_num_iter(run + 1);

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
      for (auto ii = (*h_graph).begin(); ii != (*h_graph).end(); ++ii) {
        if ((*h_graph).isOwned((*h_graph).getGID(*ii))) {
          // outputs betweenness centrality
          sprintf(v_out, "%lu %.9f\n", (*h_graph).getGID(*ii),
                  (*h_graph).getData(*ii).betweeness_centrality);
          galois::runtime::printOutput(v_out);
        }
      }
    #ifdef __GALOIS_HET_CUDA__
    } else if (personality == GPU_CUDA) {
      for (auto ii = (*h_graph).begin(); ii != (*h_graph).end(); ++ii) {
        if ((*h_graph).isOwned((*h_graph).getGID(*ii))) 
          sprintf(v_out, "%lu %.9f\n", (*h_graph).getGID(*ii),
                  get_node_betweeness_centrality_cuda(cuda_ctx, *ii));

          galois::runtime::printOutput(v_out);
          memset(v_out, '\0', 40);
      }
    }
    #endif
    free(v_out);
  }

  return 0;
}
