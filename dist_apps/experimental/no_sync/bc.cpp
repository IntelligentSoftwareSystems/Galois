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
#include "galois/DistGalois.h"
#include "galois/gstl.h"
#include "DistBenchStart.h"
#include "galois/runtime/CompilerHelperFunctions.h"

#include "galois/runtime/dGraph_edgeCut.h"
#include "galois/runtime/dGraph_cartesianCut.h"
#include "galois/runtime/dGraph_hybridCut.h"

#include "galois/DReducible.h"
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
  std::atomic<uint32_t> current_length;

  uint32_t old_length;

  // Betweeness centrality vars
  uint32_t num_shortest_paths;
  uint32_t num_successors;
  std::atomic<uint32_t> num_predecessors;

  float dependency;
  float betweeness_centrality;

  // used to determine if data has been propogated yet
  uint8_t propogation_flag;
};

static std::set<uint64_t> random_sources = std::set<uint64_t>();

// no edge data = bfs not sssp
typedef hGraph<NodeData, void> Graph;
typedef typename Graph::GraphNode GNode;

// second type (unsigned int) is for edge weights
// uncomment this along with graph load below if you want to use sssp
//typedef hGraph<NodeData, unsigned int> Graph;


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
      src_data.old_length = local_infinity;
      src_data.propogation_flag = false;
    } else {
      src_data.current_length = 0;
      src_data.old_length = 0; 
      src_data.num_shortest_paths = 1;
      src_data.propogation_flag = true;
    }

    // this is read on dst too, and since this operator loops over all
    // nodes, we reset there as well
    src_data.dependency = 0;

    // both of these end up being 0 on both source and dest over course
    // of computation
    assert(src_data.num_predecessors == 0);
    assert(src_data.num_successors == 0);

    if (!is_source) {
      assert(!src_data.propogation_flag);
    }
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
      boost::make_counting_iterator(__begin), 
      boost::make_counting_iterator(__end), 
      FirstIterationSSSP(&_graph),
      galois::no_stats(),
      galois::loopname("FirstIterationSSSP"));
  }

  /* Does SSSP, push/filter based (actually it does BFS...) */
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);
    for (auto current_edge = graph->edge_begin(src), 
              end_edge = graph->edge_end(src); 
         current_edge != end_edge; 
         ++current_edge) {
      GNode dst = graph->getEdgeDst(current_edge);
      auto& dst_data = graph->getData(dst);

      // BFS 
      uint32_t new_dist = 1 + src_data.current_length;

      galois::atomicMin(dst_data.current_length, new_dist);
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

    uint32_t iterations = 1;
    uint32_t accum_result;

    auto& nodesWithEdges = _graph.allNodesWithEdgesRange();

    do {
      _graph.set_num_iter(iterations);
      dga.reset();

      galois::do_all(
        nodesWithEdges,
        SSSP(&_graph, dga), 
        galois::no_stats(),
        galois::loopname("SSSP"), 
        galois::steal());

      iterations++;

      accum_result = dga.reduce();
    } while (accum_result);
  }

  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    if (src_data.old_length > src_data.current_length) {
      src_data.old_length = src_data.current_length;

      for (auto current_edge = graph->edge_begin(src), 
                end_edge = graph->edge_end(src); 
           current_edge != end_edge; 
           ++current_edge) {

        GNode dst = graph->getEdgeDst(current_edge);
        auto& dst_data = graph->getData(dst);

        // BFS
        uint32_t new_dist = 1 + src_data.current_length;
        uint32_t old = galois::atomicMin(dst_data.current_length, new_dist);

        if (old > new_dist) {
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
    
    {
    galois::do_all(
      nodesWithEdges,
      PredAndSucc(infinity, &_graph), 
      galois::no_stats(),
      galois::loopname("PredAndSucc"),
      galois::steal());
    }
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
        auto& dst_data = graph->getData(dst);

        // BFS
        uint32_t edge_weight = 1;

        if ((src_data.current_length + edge_weight) == dst_data.current_length) {
          galois::add(src_data.num_successors, (unsigned int)1);
          galois::atomicAdd(dst_data.num_predecessors, (unsigned int)1);
        }
      }
    }
  }
};

struct NumShortestPathsChanges {
  Graph* graph;

  NumShortestPathsChanges(Graph* _graph) : graph(_graph){}

  void static go(Graph& _graph) {
    auto& nodesWithEdges = _graph.allNodesWithEdgesRange();

    galois::do_all(
      nodesWithEdges.begin(), nodesWithEdges.end(), 
      NumShortestPathsChanges{&_graph}, 
      galois::no_stats(),
      galois::loopname("NumShortestPathsChanges"));
  }

  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    // decrement predecessor by trim then reset
    if (src_data.trim > 0) {
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

    do {
      _graph.set_num_iter(iterations);
      dga.reset();

      { 
        galois::do_all(
          nodesWithEdges,
          NumShortestPaths(infinity, &_graph, dga), 
          galois::no_stats(),
          galois::loopname("NumShortestPaths"),
          //galois::loopname(_graph.get_run_identifier("NumShortestPaths").c_str()),
          galois::steal());
      }

      // do predecessor decrementing using trim + dependency changes with
      // to_add
      NumShortestPathsChanges::go(_graph);

      iterations++;

      accum_result = dga.reduce();

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
      if (src_data.propogation_flag && src_data.num_successors > 0) {
        for (auto current_edge = graph->edge_begin(src), 
                  end_edge = graph->edge_end(src); 
             current_edge != end_edge; 
             ++current_edge) {
          GNode dst = graph->getEdgeDst(current_edge);
          auto& dst_data = graph->getData(dst);

          // SSSP
          //uint32_t edge_weight = graph->getEdgeData(current_edge);

          // BFS
          uint32_t edge_weight = 1;

          uint32_t paths_to_add = src_data.num_shortest_paths;

          if ((src_data.current_length + edge_weight) == dst_data.current_length) {
            // need to add my num_short_paths to dest
            galois::atomicAdd(dst_data.to_add, paths_to_add);
            // increment dst trim so it can decrement predecessor
            galois::atomicAdd(dst_data.trim, (unsigned int)1);


            DGAccumulator_accum += 1;
          }
        }

        // set flag so that it doesn't propogate its info more than once
        src_data.propogation_flag = false;
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

    galois::do_all(
      nodesWithEdges.begin(), nodesWithEdges.end(),
      DependencyPropChanges{infinity, &_graph}, 
      galois::no_stats(),
      galois::loopname("DependencyPropChanges"));
  }

  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    if (src_data.current_length != local_infinity) {
      // increment dependency using to_add_float then reset
      if (src_data.to_add_float > 0.0) {
        src_data.dependency += src_data.to_add_float;
        src_data.to_add_float = 0.0;

      }

      if (src_data.num_successors == 0 && src_data.propogation_flag) {
        // has had dependency back-propogated; reset the flag
        assert(src_data.trim == 0);
        src_data.propogation_flag = false;
        // reset here so I don't have to do it later... (the sync will happen
        // later if it needs to)
        src_data.num_shortest_paths = 0;
      } else if (src_data.trim > 0) {
        // decrement successor by trim then reset
        assert(src_data.trim <= src_data.num_successors);

        src_data.num_successors = src_data.num_successors - src_data.trim;
        src_data.trim = 0;

        if (src_data.num_successors == 0) {
          assert(!src_data.propogation_flag);
          src_data.propogation_flag = true;
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

    {
      galois::do_all(
        nodesWithEdges,
        DependencyPropogation(infinity, current_src_node, &_graph, dga), 
        galois::no_stats(),
        galois::loopname("DependencyPropogation"),
        //galois::loopname(_graph.get_run_identifier("DependencyPropogation").c_str()),
        galois::steal());
    }
                    

      // use trim + to add to do appropriate changes
      DependencyPropChanges::go(_graph);

      iterations++;
      accum_result = dga.reduce();
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
            auto& dst_data = graph->getData(dst);

            // SSSP
            //unsigned int edge_weight = graph->getEdgeData(current_edge);
            // BFS
            uint32_t edge_weight = 1;

            // only operate if a dst flag is set (i.e. no more succ, finalized
            // dependency to take)
            if (dst_data.propogation_flag) {
              // dest on shortest path with this node as predecessor
              if ((src_data.current_length + edge_weight) == dst_data.current_length) {
                // increment my trim for later use to decrement successor
                galois::atomicAdd(src_data.trim, (unsigned int)1);

                // update my to_add_float (which is later used to update dependency)
                galois::add(src_data.to_add_float, 
                (((float)src_data.num_shortest_paths / 
                      (float)dst_data.num_shortest_paths) * 
                 (float)(1.0 + dst_data.dependency)));


                DGAccumulator_accum += 1;
              }
            }
          }
        } else {
          // this is source of this iteration's of sssp/bfs; reset num succ to 0
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

      galois::gDebug("Current source node for BC is ", current_src_node);

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

      // do between-cent calculations for this iteration 
      DependencyPropogation::go(_graph, dga);
      //galois::gDebug("DepPropogation done");

      _graph.set_num_iter(0);

      auto& nodesWithEdges = _graph.allNodesWithEdgesRange();

      // finally, since dependencies are finalized for this round at this 
      // point, add them to the betweeness centrality measure on each node
      // TODO all nodes here? would remove unnecessary dep sync later, 
      // but will cause destinations (which don't need to increment bc)
      // to do extra work on each host
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

    galois::add(src_data.betweeness_centrality, src_data.dependency);
    // done with it, reset
    //src_data.dependency = 0;
  }
};
 
/******************************************************************************/
/* Main method for running */
/******************************************************************************/

int main(int argc, char** argv) {
  try {
    galois::DistMemSys G(getStatsFile());
    DistBenchStart(argc, argv, name, desc, url);

    {
    auto& net = galois::runtime::getSystemNetworkInterface();
    if (net.ID == 0) {
      galois::runtime::reportStat("(NULL)", "Max Iterations", 
                                  (unsigned long)maxIterations, 0);
    }

    galois::StatTimer StatTimer_graph_init("TIMER_GRAPH_INIT"),
                      StatTimer_total("TIMER_TOTAL"),
                      StatTimer_hg_init("TIMER_HG_INIT");

    StatTimer_total.start();

    std::vector<unsigned> scalefactor;

    StatTimer_hg_init.start();

    Graph* h_graph = nullptr;
    // uses sssp
    //h_graph = constructGraph<NodeData, unsigned int>(scalefactor);
    // uses bfs
    h_graph = constructGraph<NodeData, void>(scalefactor);

    // random num generate for sources
    std::minstd_rand0 r_generator;
    r_generator.seed(100);
    std::uniform_int_distribution<uint64_t> r_dist(0, h_graph->size() - 1);

    if (numberOfSources != 0) {
      random_sources.insert(startSource);

      while (random_sources.size() < numberOfSources) {
        random_sources.insert(r_dist(r_generator));
      }
    }


    StatTimer_hg_init.stop();

    std::cout << "[" << net.ID << "] InitializeGraph::go called\n";

    StatTimer_graph_init.start();
      InitializeGraph::go((*h_graph));
    StatTimer_graph_init.stop();

    // shared DG accumulator among all steps
    galois::DGAccumulator<uint32_t> dga;

    for (auto run = 0; run < numRuns; ++run) {
      std::cout << "[" << net.ID << "] BC::go run " << run << " called\n";
      std::string timer_str("TIMER_" + std::to_string(run));
      galois::StatTimer StatTimer_main(timer_str.c_str());

      StatTimer_main.start();
        BC::go(*h_graph, dga);
      StatTimer_main.stop();

      // TODO sanity check setup

      // re-init graph for next run
      if ((run + 1) != numRuns) {
        galois::runtime::getHostBarrier().wait();
        (*h_graph).reset_num_iter(run + 1);


        InitializeGraph::go((*h_graph));
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
