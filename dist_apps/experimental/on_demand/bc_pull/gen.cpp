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
 * Compute Betweeness-Centrality on distributed Galois using, at the moment,
 * BFS (NOT SSSP) for distances
 *
 * @author Loc Hoang <l_hoang@utexas.edu>
 */

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

#include "galois/runtime/dGraph_edgeCut.h"
#include "galois/runtime/dGraph_cartesianCut.h"
#include "galois/runtime/dGraph_hybridCut.h"

#include "galois/DistAccumulator.h"
#include "galois/runtime/Tracer.h"

#include "galois/runtime/dGraphLoader.h"

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
static cll::opt<bool> verify("verify", 
                             cll::desc("Verify ranks by printing to "
                                       "'page_ranks.#hid.csv' file"),
                             cll::init(false));
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

const uint32_t infinity = std::numeric_limits<uint32_t>::max() / 4;
static uint64_t current_src_node = 0;

/******************************************************************************/
/* Graph structure declarations */
/******************************************************************************/

// NOTE: types assume that these values will not reach uint64_t: it may
// need to be changed for very large graphs
struct NodeData {
  // SSSP vars
  uint32_t current_length;

  uint32_t num_shortest_paths;
  std::atomic<uint32_t> num_successors;
  uint32_t num_predecessors;
  std::atomic<float> dependency;
  float betweeness_centrality;

  // used to determine if data has been propogated yet
  uint8_t propogation_flag;

  uint32_t trim;
  uint32_t to_add;

  std::atomic<uint32_t> trim2;
  std::atomic<float> to_add_float;

  uint8_t num_short_paths_flag;
  uint8_t dep_prop_flag;
};

static std::set<uint64_t> random_sources = std::set<uint64_t>();

typedef hGraph<NodeData, unsigned int> Graph;
typedef typename Graph::GraphNode GNode;

#if __OPT_VERSION__ >= 3
galois::DynamicBitSet bitset_to_add;
galois::DynamicBitSet bitset_to_add_float;
galois::DynamicBitSet bitset_num_successors;
galois::DynamicBitSet bitset_num_predecessors;
galois::DynamicBitSet bitset_trim;
galois::DynamicBitSet bitset_trim2;
galois::DynamicBitSet bitset_current_length;
#endif

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

    galois::do_all(
      allNodes.begin(), allNodes.end(), 
      InitializeGraph{&_graph}, 
      galois::no_stats(),
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

  /* Reset necessary graph metadata for next iteration of SSSP/BFS */
  void static go(Graph& _graph) {
    auto& allNodes = _graph.allNodesRange();

    galois::do_all(
      allNodes.begin(), allNodes.end(), 
      InitializeIteration{infinity, current_src_node, &_graph},
      galois::no_stats(),
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

    src_data.num_short_paths_flag = false;
    src_data.dep_prop_flag = false;
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

    auto& nodesWithEdges = _graph.allNodesWithEdgesRange();

    do {
      _graph.set_num_iter(iterations);
      dga.reset();

      #if __OPT_VERSION__ == 5
      _graph.sync_on_demand<readDestination, Reduce_min_current_length,
                            Broadcast_current_length,
                            Bitset_current_length>(Flags_current_length, "SSSP");
      #endif

      galois::do_all(
        nodesWithEdges,
        SSSP(&_graph, dga), 
        galois::no_stats(),
        galois::loopname("SSSP"), 
        galois::steal());

      #if __OPT_VERSION__ == 5
      Flags_current_length.set_write_src();
      #endif

      iterations++;

      accum_result = dga.reduce();

      #if __OPT_VERSION__ == 1
      _graph.sync<writeAny, readAny, Reduce_min_current_length,
                  Broadcast_current_length>("SSSP");
      #elif __OPT_VERSION__ == 2
      _graph.sync<writeAny, readAny, Reduce_min_current_length,
                  Broadcast_current_length>("SSSP");
      #elif __OPT_VERSION__ == 3
      _graph.sync<writeAny, readAny, Reduce_min_current_length,
                  Broadcast_current_length, Bitset_current_length>("SSSP");
      #elif __OPT_VERSION__ == 4
      _graph.sync<writeSource, readAny, Reduce_min_current_length,
                  Broadcast_current_length, Bitset_current_length>("SSSP");
      #endif
    } while (accum_result);
  }

  /* Does SSSP (actually BFS at the moment) */
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    for (auto current_edge = graph->edge_begin(src), 
              end_edge = graph->edge_end(src); 
         current_edge != end_edge; 
         ++current_edge) {
      GNode dst = graph->getEdgeDst(current_edge);
      auto& dst_data = graph->getData(dst);

      //uint32_t new_dist = 1 + dst_data.current_length;
      uint32_t new_dist = graph->getEdgeData(current_edge) + 
                              dst_data.current_length + 1;

      uint32_t old = galois::min(src_data.current_length, new_dist);

      if (old > new_dist) {
        #if __OPT_VERSION__ >= 3
        bitset_current_length.set(src);
        #endif
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
    auto& nodesWithEdges = _graph.allNodesWithEdgesRange();

    #if __OPT_VERSION__ == 5
    _graph.sync_on_demand<readAny, Reduce_min_current_length,
                          Broadcast_current_length,
                          Bitset_current_length>(Flags_current_length, 
                                                 "PredAndSucc");
    #endif

    galois::do_all(
      nodesWithEdges,
      PredAndSucc(infinity, &_graph), 
      galois::no_stats(),
      galois::loopname("PredAndSucc"),
      galois::steal());

    #if __OPT_VERSION__ == 5
    Flags_num_predecessors.set_write_src();
    Flags_num_successors.set_write_dst();
    #endif

    #if __OPT_VERSION__ == 1
    _graph.sync<writeAny, readAny, Reduce_add_num_predecessors,
                Broadcast_num_predecessors>("PredAndSucc");
    _graph.sync<writeAny, readAny, Reduce_add_num_successors,
                Broadcast_num_successors>("PredAndSucc");
    #elif __OPT_VERSION__ == 2
    _graph.sync<writeAny, readAny, Reduce_add_num_predecessors,
                Broadcast_num_predecessors>("PredAndSucc");
    _graph.sync<writeAny, readAny, Reduce_add_num_successors,
                Broadcast_num_successors>("PredAndSucc");
    #elif __OPT_VERSION__ == 3
    _graph.sync<writeAny, readAny, Reduce_add_num_predecessors,
                Broadcast_num_predecessors,
                Bitset_num_predecessors>("PredAndSucc");
    _graph.sync<writeAny, readAny, Reduce_add_num_successors,
                Broadcast_num_successors, Bitset_num_successors>("PredAndSucc");
    #elif __OPT_VERSION__ == 4
    _graph.sync<writeSource, readAny, Reduce_add_num_predecessors,
                Broadcast_num_predecessors,
                Bitset_num_predecessors>("PredAndSucc");
    _graph.sync<writeDestination, readAny, Reduce_add_num_successors,
                Broadcast_num_successors, Bitset_num_successors>("PredAndSucc");
    #endif
  }

  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    if (src_data.current_length != local_infinity) {
      for (auto current_edge = graph->edge_begin(src),  
                end_edge = graph->edge_end(src); 
           current_edge != end_edge; 
           ++current_edge) {
        GNode dst = graph->getEdgeDst(current_edge);
        auto& dst_data = graph->getData(dst);

        //uint32_t edge_weight = 1;
        uint32_t edge_weight = graph->getEdgeData(current_edge) + 1;

        if ((dst_data.current_length + edge_weight) == src_data.current_length) {
          // dest on shortest path with this node as successor
          galois::add(src_data.num_predecessors, (unsigned int)1);
          galois::atomicAdd(dst_data.num_successors, (unsigned int)1);

          #if __OPT_VERSION__ >= 3
          bitset_num_successors.set(dst);
          bitset_num_predecessors.set(src);
          #endif
        }
      }
    }
  }
};

struct NSPTrim {
  Graph* graph;

  NSPTrim(Graph* _graph) : graph(_graph) { }

  void static go(Graph& _graph) {
    auto& allNodes = _graph.allNodesRange();

    #if __OPT_VERSION__ == 5
    _graph.sync_on_demand<readAny, Reduce_add_trim, Broadcast_trim,
                          Bitset_trim>(Flags_trim, "NSPTrim");
    #endif

    galois::do_all(
      allNodes.begin(), allNodes.end(), 
      NSPTrim{&_graph}, 
      galois::no_stats(),
      galois::loopname("NSPTrim"));
  }

  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    if (src_data.trim > 0) {
      src_data.num_predecessors -= src_data.trim;
      src_data.trim = 0;
    }
  }
};

struct NSPAdd {
  Graph* graph;

  NSPAdd(Graph* _graph) : graph(_graph) { }

  void static go(Graph& _graph) {
    auto& allNodes = _graph.allNodesRange();

    #if __OPT_VERSION__ == 5
    _graph.sync_on_demand<readAny, Reduce_add_to_add, Broadcast_to_add,
                          Bitset_to_add>(Flags_to_add, "NSPAdd");
    #endif

    galois::do_all(
      allNodes.begin(), allNodes.end(), 
      NSPAdd{&_graph}, 
      galois::no_stats(),
      galois::loopname("NSPAdd"));
  }

  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    if (src_data.to_add > 0) {
      src_data.num_shortest_paths += src_data.to_add;
      src_data.to_add = 0;
    }
  }
};


struct NumShortestPathsChanges {
  const uint32_t &local_infinity;
  Graph* graph;

  NumShortestPathsChanges(const uint32_t &_local_infinity, Graph* _graph) : 
      local_infinity(_local_infinity), graph(_graph) {}

  void static go(Graph& _graph) {
    auto& allNodes = _graph.allNodesRange();

    #if __OPT_VERSION__ == 5
    _graph.sync_on_demand<readAny, Reduce_min_current_length,
                          Broadcast_current_length, 
                          Bitset_current_length>(Flags_current_length,
                                                 "NumShortestPathsChanges");
    _graph.sync_on_demand<readAny, Reduce_add_num_predecessors,
                          Broadcast_num_predecessors, 
                          Bitset_num_predecessors>(Flags_num_predecessors,
                                                   "NumShortestPathsChanges");
    #endif

    galois::do_all(
      allNodes.begin(), allNodes.end(), 
      NumShortestPathsChanges{infinity, &_graph}, 
      galois::no_stats(),
      galois::loopname("NumShortestPathsChanges"), 
      galois::no_stats()
    );
  }

  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    if (src_data.current_length != local_infinity) {
      if (src_data.num_predecessors == 0 && src_data.propogation_flag) {
        // has had short path taken; reset the flag
        src_data.propogation_flag = false;

        if (!src_data.num_short_paths_flag) {
          src_data.num_short_paths_flag = true;
        }
      } else {
        if (src_data.num_predecessors == 0 && !src_data.num_short_paths_flag) {
          assert(!src_data.propogation_flag);
          src_data.propogation_flag = true;
          src_data.num_short_paths_flag = true;
        }
      }
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

    auto& nodesWithEdges = _graph.allNodesWithEdgesRange();

    #if __OPT_VERSION__ == 5
    _graph.sync_on_demand<readAny, Reduce_min_current_length,
                          Broadcast_current_length, 
                          Bitset_current_length>(Flags_current_length,
                                                 "NumShortestPaths");
    _graph.sync_on_demand<readSource, Reduce_add_num_predecessors,
                          Broadcast_num_predecessors, 
                          Bitset_num_predecessors>(Flags_num_predecessors,
                                                   "NumShortestPaths");
    #endif

    do {
      _graph.set_num_iter(iterations);
      dga.reset();

      galois::do_all(
        nodesWithEdges,
        NumShortestPaths(infinity, &_graph, dga), 
        galois::no_stats(),
        galois::loopname("NumShortestPaths"),
        galois::steal(),
        galois::no_stats()
      );

      #if __OPT_VERSION__ == 5
      Flags_trim.set_write_src();
      Flags_to_add.set_write_src();
      #endif

      #if __OPT_VERSION__ == 1
      _graph.sync<writeAny, readAny, Reduce_add_trim,
                  Broadcast_trim>("NumShortestPaths");
      #elif __OPT_VERSION__ == 2
      _graph.sync<writeAny, readAny, Reduce_add_trim,
                  Broadcast_trim>("NumShortestPaths");
      #elif __OPT_VERSION__ == 3
      _graph.sync<writeAny, readAny, Reduce_add_trim,
                  Broadcast_trim, Bitset_trim>("NumShortestPaths");
      #elif __OPT_VERSION__ == 4
      _graph.sync<writeSource, readAny, Reduce_add_trim,
                  Broadcast_trim, Bitset_trim>("NumShortestPaths");
      #endif
      NSPTrim::go(_graph);

      #if __OPT_VERSION__ == 1
      _graph.sync<writeAny, readAny, Reduce_add_to_add,
                  Broadcast_to_add>("NumShortestPaths");
      #elif __OPT_VERSION__ == 2
      _graph.sync<writeAny, readAny, Reduce_add_to_add,
                  Broadcast_to_add>("NumShortestPaths");
      #elif __OPT_VERSION__ == 3
      _graph.sync<writeAny, readAny, Reduce_add_to_add,
                  Broadcast_to_add, Bitset_to_add>("NumShortestPaths");
      #elif __OPT_VERSION__ == 4
      _graph.sync<writeSource, readAny, Reduce_add_to_add,
                  Broadcast_to_add, Bitset_to_add>("NumShortestPaths");
      #endif
      NSPAdd::go(_graph);

      // this deals with flag
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
        for (auto current_edge = graph->edge_begin(src), 
                  end_edge = graph->edge_end(src); 
             current_edge != end_edge; 
             ++current_edge) {
          GNode dst = graph->getEdgeDst(current_edge);
          auto& dst_data = graph->getData(dst);

          //uint32_t edge_weight = 1;
          uint32_t edge_weight = graph->getEdgeData(current_edge) + 1;

          // only operate if a flag is set (i.e. no more pred, finalized
          // short paths to take)
          if (dst_data.propogation_flag) {
            // dest on shortest path with this node as successor
            if ((dst_data.current_length + edge_weight) == src_data.current_length) {
              galois::add(src_data.trim, (uint32_t)1);
              galois::add(src_data.to_add, 
                          (uint32_t)dst_data.num_shortest_paths);

              #if __OPT_VERSION__ >= 3
              bitset_trim.set(src);
              bitset_to_add.set(src);
              #endif

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
    auto& allNodes = _graph.allNodesRange();

    #if __OPT_VERSION__ == 5
    _graph.sync_on_demand<readAny, Reduce_add_num_successors,
                          Broadcast_num_successors, 
                          Bitset_num_successors>(Flags_num_successors,
                                                 "FlagPrep");
    _graph.sync_on_demand<readAny, Reduce_min_current_length,
                          Broadcast_current_length, 
                          Bitset_current_length>(Flags_current_length,
                                                 "FlagPrep");
    #endif

    galois::do_all(
      allNodes.begin(), allNodes.end(), 
      FlagPrep{infinity, &_graph}, 
      galois::no_stats(),
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
        // sanity check assert; doesn't actually need a read sync
        assert(src_data.propogation_flag == false);
      }
    }
  }
};

struct DPTrim {
  Graph* graph;

  DPTrim(Graph* _graph) : graph(_graph) { }

  void static go(Graph& _graph) {
    auto& allNodes = _graph.allNodesRange();

    #if __OPT_VERSION__ == 5
    _graph.sync_on_demand<readAny, Reduce_add_trim2, Broadcast_trim2,
                          Bitset_trim2>(Flags_trim2, "DPTrim");
    #endif

    galois::do_all(
      allNodes.begin(), allNodes.end(), 
      DPTrim{&_graph}, 
      galois::no_stats(),
      galois::loopname("DPTrim"), 
      galois::no_stats()
    );
  }

  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    if (src_data.trim2 > 0) {
      src_data.num_successors -= src_data.trim2;
      src_data.trim2 = 0;
    }
  }
};

struct DPAdd {
  Graph* graph;

  DPAdd(Graph* _graph) : graph(_graph) { }

  void static go(Graph& _graph) {
    auto& allNodes = _graph.allNodesRange();

    #if __OPT_VERSION__ == 5
    _graph.sync_on_demand<readAny, Reduce_add_to_add_float, Broadcast_to_add_float,
                          Bitset_to_add_float>(Flags_to_add_float, "DPAdd");
    #endif

    galois::do_all(
      allNodes.begin(), allNodes.end(), 
      DPAdd{&_graph}, 
      galois::no_stats(),
      galois::loopname("DPAdd"), 
      galois::no_stats()
    );
  }

  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    if (src_data.to_add_float > 0) {
      src_data.dependency = src_data.dependency + src_data.to_add_float;
      src_data.to_add_float = 0;
    }
  }
};

struct DependencyPropChanges {
  const uint32_t &local_infinity;
  Graph* graph;

  DependencyPropChanges(const uint32_t &_local_infinity,
               Graph* _graph) : local_infinity(_local_infinity), graph(_graph){}

  void static go(Graph& _graph) {
    auto& nodesWithEdges = _graph.allNodesWithEdgesRange();

    #if __OPT_VERSION__ == 5
    _graph.sync_on_demand<readSource, Reduce_min_current_length,
                          Broadcast_current_length, 
                          Bitset_current_length>(Flags_current_length,
                                                 "DependencyPropChanges");
    _graph.sync_on_demand<readSource, Reduce_add_num_successors,
                          Broadcast_num_successors, 
                          Bitset_num_successors>(Flags_num_successors,
                                                 "DependencyPropChanges");
    #endif

    galois::do_all(
      nodesWithEdges.begin(), nodesWithEdges.end(),
      DependencyPropChanges{infinity, &_graph}, 
      galois::no_stats(),
      galois::loopname("DependencyPropChanges"),
      galois::no_stats()
    );
  }

  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    if (src_data.current_length != local_infinity) {
      if (src_data.num_successors == 0 && !src_data.dep_prop_flag) {
        src_data.propogation_flag = true;
        src_data.dep_prop_flag = true;
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

      #if __OPT_VERSION__ == 5
      _graph.sync_on_demand<readAny, Reduce_min_current_length, 
                            Broadcast_current_length, 
                            Bitset_current_length>(Flags_current_length,
                                                   "DependencyPropogation");
      #endif 

      galois::do_all(
        nodesWithEdges,
        DependencyPropogation(infinity, current_src_node, &_graph, dga), 
        galois::no_stats(),
        galois::loopname("DependencyPropogation"),
        galois::steal(),
        galois::no_stats()
      );

      #if __OPT_VERSION__ == 5
      Flags_trim2.set_write_dst();
      Flags_to_add_float.set_write_dst();
      #endif

      #if __OPT_VERSION__ == 1
      _graph.sync<writeAny, readAny, Reduce_add_trim2,
                  Broadcast_trim2>("DependencyPropogation");
      #elif __OPT_VERSION__ == 2
      _graph.sync<writeAny, readAny, Reduce_add_trim2,
                  Broadcast_trim2>("DependencyPropogation");
      #elif __OPT_VERSION__ == 3
      _graph.sync<writeAny, readAny, Reduce_add_trim2,
                  Broadcast_trim2, Bitset_trim2>("DependencyPropogation");
      #elif __OPT_VERSION__ == 4
      _graph.sync<writeDestination, readAny, Reduce_add_trim2,
                  Broadcast_trim2, Bitset_trim2>("DependencyPropogation");
      #endif

      DPTrim::go(_graph);

      #if __OPT_VERSION__ == 1
      _graph.sync<writeAny, readAny, Reduce_add_to_add_float,
                  Broadcast_to_add_float>("DependencyPropogation");
      #elif __OPT_VERSION__ == 2
      _graph.sync<writeAny, readAny, Reduce_add_to_add_float,
                  Broadcast_to_add_float>("DependencyPropogation");
      #elif __OPT_VERSION__ == 3
      _graph.sync<writeAny, readAny, Reduce_add_to_add_float,
                  Broadcast_to_add_float, 
                  Bitset_to_add_float>("DependencyPropogation");
      #elif __OPT_VERSION__ == 4
      _graph.sync<writeDestination, readAny, Reduce_add_to_add_float,
                  Broadcast_to_add_float, 
                  Bitset_to_add_float>("DependencyPropogation");
      #endif

      DPAdd::go(_graph);

      // flag changing (has to be done between BSP rounds so values
      // are propogated more than once)
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
  
        for (auto current_edge = graph->edge_begin(src), 
                  end_edge = graph->edge_end(src); 
             current_edge != end_edge; 
             ++current_edge) {
          GNode dst = graph->getEdgeDst(current_edge);
  
          // ignore current source node of bc iteration
          if (graph->getGID(dst) == local_current_src_node) {
            continue;
          }
  
          auto& dst_data = graph->getData(dst);

          //uint32_t edge_weight = 1;
          uint32_t edge_weight = graph->getEdgeData(current_edge) + 1;
          uint32_t dep = src_data.dependency;

          // I am successor to destination
          if ((dst_data.current_length + edge_weight) == src_data.current_length) {
            galois::atomicAdd(dst_data.trim2, (uint32_t)1);
            galois::atomicAdd(dst_data.to_add_float,
                              (((float)dst_data.num_shortest_paths / 
                                  (float)src_data.num_shortest_paths) * 
                              (float)(1.0 + dep)));

            #if __OPT_VERSION__ >= 3
            bitset_trim2.set(dst);
            bitset_to_add_float.set(dst);
            #endif

            DGAccumulator_accum += 1;
          }
        }

        // reset flag so that it doesn't propogate its info more than once
        src_data.propogation_flag = false;
        if (!src_data.dep_prop_flag) {
          src_data.dep_prop_flag = true;
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
      if (galois::runtime::getSystemNetworkInterface().ID == 0) {
        if (i % 5000 == 0) {
          std::cout << "SSSP source node #" << i << "\n";
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
      DependencyPropogation::go(_graph, dga);
      //galois::gDebug("DepPropogation done");

      _graph.set_num_iter(0);

      auto& nodesWithEdges = _graph.allNodesWithEdgesRange();

      // finally, since dependencies are finalized for this round at this 
      // point, add them to the betweeness centrality measure on each node

      galois::do_all(
        nodesWithEdges.begin(), 
        nodesWithEdges.end(), 
        BC(&_graph), 
        galois::no_stats(),
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

    galois::do_all(_graph.allNodesRange().begin(), _graph.allNodesRange().end(), 
                   Sanity(
                     &_graph,
                     DGA_max,
                     DGA_min,
                     DGA_sum
                   ), 
                   galois::no_stats(),
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

int main(int argc, char** argv) {
  try {
    galois::DistMemSys G;
    DistBenchStart(argc, argv, name, desc, url);

    {
    auto& net = galois::runtime::getSystemNetworkInterface();
    if (net.ID == 0) {
      galois::runtime::reportParam("BC", "Max Iterations", 
                                  (unsigned long)maxIterations);
      #if __OPT_VERSION__ == 1
      printf("Version 1 of optimization\n");
      #elif __OPT_VERSION__ == 2
      printf("Version 2 of optimization\n");
      #elif __OPT_VERSION__ == 3
      printf("Version 3 of optimization\n");
      #elif __OPT_VERSION__ == 4
      printf("Version 4 of optimization\n");
      #elif __OPT_VERSION__ == 5
      printf("Version 5 of optimization\n");
      #endif
    }

    galois::StatTimer StatTimer_graph_init("TIMER_GRAPH_INIT"),
                      StatTimer_total("TIMER_TOTAL"),
                      StatTimer_hg_init("TIMER_HG_INIT");

    StatTimer_total.start();

    std::vector<unsigned> scalefactor;

    StatTimer_hg_init.start();

    Graph* h_graph = nullptr;
    h_graph = constructGraph<NodeData, unsigned int, false>(scalefactor);
    // uses bfs
    //h_graph = constructGraph<NodeData, void, false>(scalefactor);

    // random num generate for sources
    std::minstd_rand0 r_generator;
    r_generator.seed(100);
    std::uniform_int_distribution<uint64_t> r_dist(0, h_graph->size() - 1);

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

    #if __OPT_VERSION__ >= 3
    bitset_to_add.resize(h_graph->size());
    bitset_to_add_float.resize(h_graph->size());
    bitset_num_successors.resize(h_graph->size());
    bitset_num_predecessors.resize(h_graph->size());
    bitset_trim.resize(h_graph->size());
    bitset_trim2.resize(h_graph->size());
    bitset_current_length.resize(h_graph->size());
    #endif

    StatTimer_hg_init.stop();

    std::cout << "[" << net.ID << "] InitializeGraph::go called\n";

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

      // TODO sanity check setup
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
      #if __OPT_VERSION__ >= 3
      // TODO GPU code
      //#ifdef __GALOIS_HET_CUDA__
      //  if (personality == GPU_CUDA) { 
      //    bitset_to_add_reset_cuda(cuda_ctx);
      //    bitset_to_add_float_reset_cuda(cuda_ctx);
      //    bitset_num_successors_reset_cuda(cuda_ctx);
      //    bitset_num_predecessors_reset_cuda(cuda_ctx);
      //    bitset_trim_reset_cuda(cuda_ctx);
      //    bitset_trim2.reset();
      //    bitset_current_length_reset_cuda(cuda_ctx);
      //    bitset_old_length_reset_cuda(cuda_ctx);
      //  } else
      //#endif
        {
        bitset_to_add.reset();
        bitset_to_add_float.reset();
        bitset_num_successors.reset();
        bitset_num_predecessors.reset();
        bitset_trim.reset();
        bitset_trim2.reset();
        bitset_current_length.reset();
        }
      #endif

        #if __OPT_VERSION__ == 5
        Flags_current_length.clear_all();
        Flags_num_successors.clear_all();
        Flags_num_predecessors.clear_all();
        Flags_trim.clear_all();
        Flags_trim2.clear_all();
        Flags_to_add.clear_all();
        Flags_to_add_float.clear_all();
        #endif

        InitializeGraph::go((*h_graph));
        galois::runtime::getHostBarrier().wait();
      }
    }

    StatTimer_total.stop();

    // Verify, i.e. print out graph data for examination
    if (verify) {
      char *v_out = (char*)malloc(40);
        for (auto ii = (*h_graph).begin(); ii != (*h_graph).end(); ++ii) {
          if ((*h_graph).isOwned((*h_graph).getGID(*ii))) {
            // outputs betweenness centrality
            sprintf(v_out, "%lu %.9f\n", (*h_graph).getGID(*ii),
                    (*h_graph).getData(*ii).betweeness_centrality);
            galois::runtime::printOutput(v_out);
          }
        }
      free(v_out);
    }
    }
    galois::runtime::getHostBarrier().wait();

    return 0;
  } catch(const char* c) {
    std::cout << "Error: " << c << "\n";
    return 1;
  }
}
