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

#include "Galois/DistGalois.h"
#include "Galois/gstl.h"
#include "DistBenchStart.h"
#include "Galois/Runtime/CompilerHelperFunctions.h"

#include "Galois/Runtime/dGraph_edgeCut.h"
#include "Galois/Runtime/dGraph_cartesianCut.h"
#include "Galois/Runtime/dGraph_hybridCut.h"

#include "Galois/DistAccumulator.h"
#include "Galois/Runtime/Tracer.h"

#include "Galois/Runtime/dGraphLoader.h"

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

    Galois::do_all(
      allNodes.begin(), allNodes.end(), 
      InitializeGraph{&_graph}, 
      Galois::loopname("InitializeGraph"), 
      Galois::timeit(),
      Galois::no_stats()
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

    Galois::do_all(
      allNodes.begin(), allNodes.end(), 
      InitializeIteration{infinity, current_src_node, &_graph},
      Galois::loopname("InitializeIteration"), 
      //Galois::loopname(_graph.get_run_identifier("InitializeIteration").c_str()), 
      Galois::timeit(),
      Galois::no_stats()
    );
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

    assert(src_data.num_predecessors == 0);
    assert(src_data.num_successors == 0);
  }
};

/* Sub struct for running SSSP (beyond 1st iteration) */
struct SSSP {
  Graph* graph;
  Galois::DGAccumulator<uint32_t>& DGAccumulator_accum;

  SSSP(Graph* _graph, Galois::DGAccumulator<uint32_t>& dga) : 
    graph(_graph), DGAccumulator_accum(dga) { }

  void static go(Graph& _graph, Galois::DGAccumulator<uint32_t>& dga) {
    uint32_t iterations = 0;
    uint32_t accum_result;

    auto& nodesWithEdges = _graph.allNodesWithEdgesRange();

    do {
      _graph.set_num_iter(iterations);
      dga.reset();

      Galois::do_all_local(
        nodesWithEdges,
        SSSP(&_graph, dga), 
        Galois::loopname("SSSP"), 
        Galois::do_all_steal<true>(),
        Galois::timeit(),
        Galois::no_stats()
      );

      iterations++;

      accum_result = dga.reduce();

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

      uint32_t new_dist = 1 + dst_data.current_length;
      uint32_t old = Galois::min(src_data.current_length, new_dist);

      if (old > new_dist) {
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
    
    Galois::do_all_local(
      nodesWithEdges,
      PredAndSucc(infinity, &_graph), 
      Galois::loopname("PredAndSucc"),
      Galois::do_all_steal<true>(),
      Galois::timeit(),
      Galois::no_stats()
    );
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

        uint32_t edge_weight = 1;

        if ((dst_data.current_length + edge_weight) == src_data.current_length) {
          // dest on shortest path with this node as successor
          Galois::add(src_data.num_predecessors, (unsigned int)1);
          Galois::atomicAdd(dst_data.num_successors, (unsigned int)1);
        }
      }
    }
  }
};

struct NumShortestPathsChanges {
  Graph* graph;

  NumShortestPathsChanges(Graph* _graph) : graph(_graph){}

  void static go(Graph& _graph) {
    auto& allNodes = _graph.allNodesRange();

    Galois::do_all(
      allNodes.begin(), allNodes.end(), 
      NumShortestPathsChanges{&_graph}, 
      Galois::loopname("NumShortestPathsChanges"), 
      Galois::timeit(),
      Galois::no_stats()
    );
  }

  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    if (src_data.current_length != local_infinity) {
      if (src_data.num_predecessors == 0 && src_data.propogation_flag) {
        if (src_data.num_successors != 0) {
          // has had short path taken; reset the flag;
          // ...unless you are a leaf node,then keep flag on for 
          // next operator
          assert(src_data.trim == 0);
          src_data.propogation_flag = false;
        }
      } else {
        // TODO this is a big problem......; flag will get turned on again
        // after 1 round of being off unless I use another variable...
        // if I'm at 0 predecessors, set the flag to true (i.e. says
        // I'm ready to have my value pulled)
        if (src_data.num_predecessors == 0) {
          assert(!src_data.propogation_flag);
          src_data.propogation_flag = true;
        }
      }
    }
  }
};

/* Calculate the number of shortest paths for each node */
struct NumShortestPaths {
  const uint32_t &local_infinity;
  Graph* graph;
  Galois::DGAccumulator<uint32_t>& DGAccumulator_accum;

  NumShortestPaths(const uint32_t &_local_infinity,
                   Graph* _graph, Galois::DGAccumulator<uint32_t>& dga) : 
     local_infinity(_local_infinity), graph(_graph), DGAccumulator_accum(dga) {}

  void static go(Graph& _graph, Galois::DGAccumulator<uint32_t>& dga) {
    uint32_t iterations = 0;
    uint32_t accum_result;

    auto& nodesWithEdges = _graph.allNodesWithEdgesRange();

    do {
      _graph.set_num_iter(iterations);
      dga.reset();

      Galois::do_all_local(
        nodesWithEdges,
        NumShortestPaths(infinity, &_graph, dga), 
        Galois::loopname("NumShortestPaths"),
        Galois::do_all_steal<true>(),
        Galois::timeit(),
        Galois::no_stats()
      );

      // this deals with flag; in compiler version it should deal with trim/
      // to_add as well...
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

          uint32_t edge_weight = 1;

          // only operate if a dst flag is set (i.e. no more pred, finalized
          // short paths to take)
          if (dst_data.propogation_flag) {
            // dest on shortest path with this node as successor
            if ((dst_data.current_length + edge_weight) == src_data.current_length) {
              src_data.num_predecessors -= 1;
              src_data.num_shortest_paths += dst_data.num_shortest_paths;

              DGAccumulator_accum += 1;
            }
          }
        }
      }
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

    Galois::do_all(
      nodesWithEdges.begin(), nodesWithEdges.end(),
      DependencyPropChanges{infinity, &_graph}, 
      Galois::loopname("DependencyPropChanges"),
      Galois::timeit(),
      Galois::no_stats()
    );
  }

  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    if (src_data.current_length != local_infinity) {
      // TODO this causes flag to get turned on again unless I can
      // track if succ changed or something between rounds
      if (src_data.num_successors == 0) {
        src_data.propogation_flag = true;
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
  Galois::DGAccumulator<uint32_t>& DGAccumulator_accum;

  DependencyPropogation(const uint32_t &_local_infinity,
                        const uint64_t &_local_current_src_node,
                        Graph* _graph, Galois::DGAccumulator<uint32_t>& dga) : 
      local_infinity(_local_infinity),
      local_current_src_node(_local_current_src_node),
      graph(_graph),
      DGAccumulator_accum(dga) {}

  /* Look at all nodes to do propogation until no more work is done */
  void static go(Graph& _graph, Galois::DGAccumulator<uint32_t>& dga) {
    uint32_t iterations = 0;
    uint32_t accum_result;

    do {
      _graph.set_num_iter(iterations);
      dga.reset();

      auto& nodesWithEdges = _graph.allNodesWithEdgesRange();

      Galois::do_all_local(
        nodesWithEdges,
        DependencyPropogation(infinity, current_src_node, &_graph, dga), 
        Galois::loopname("DependencyPropogation"),
        Galois::do_all_steal<true>(),
        Galois::timeit(),
        Galois::no_stats()
      );

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

          uint32_t edge_weight = 1;
          uint32_t dep = src_data.dependency;

          // I am successor to destination
          if ((dst_data.current_length + edge_weight) == src_data.current_length) {
            dst_data.num_successors -= 1;
            dst_data.dependency += (((float)dst_data.num_shortest_paths / 
                                     (float)src_data.num_shortest_paths) * 
                                   (float)(1.0 + dep))

            DGAccumulator_accum += 1;
          }
        }

        // reset flag so that it doesn't propogate its info more than once
        src_data.propogation_flag = false;
      }
    }
  }
};

struct BC {
  Graph* graph;

  BC(Graph* _graph) : graph(_graph){}

  void static go(Graph& _graph, Galois::DGAccumulator<uint32_t>& dga){
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
        Galois::gDebug("This is single source node BC");
        current_src_node = startSource;
      } else if (use_random) {
        // number of sources non-zero, so use random sources
        current_src_node = *random_sources_iterator;
        random_sources_iterator++;
      } else {
        // all sources
        current_src_node = i;
      }

      Galois::gDebug("Current source node for BC is ", current_src_node);

      #ifndef NDEBUG
      if (_graph.id == 0) {
        if (i % 5000 == 0) {
          std::cout << "SSSP source node #" << i << "\n";
        }
      }
      #endif

      _graph.set_num_iter(0);

      // reset the graph aside from the between-cent measure
      InitializeIteration::go(_graph);
      //Galois::gDebug("Init done");

      // get SSSP on the current graph
      SSSP::go(_graph, dga);
      //Galois::gDebug("SSSP done");

      _graph.set_num_iter(0);

      // calculate the succ/pred for all nodes in the SSSP DAG
      PredAndSucc::go(_graph);
      //Galois::gDebug("PredAndSucc done");

      // calculate the number of shortest paths for each node
      NumShortestPaths::go(_graph, dga);
      //Galois::gDebug("NumShortestPaths done");

      _graph.set_num_iter(0);

      // do between-cent calculations for this iteration 
      DependencyPropogation::go(_graph, dga);
      //Galois::gDebug("DepPropogation done");

      _graph.set_num_iter(0);

      auto& allNodes = _graph.allNodesRange();

      // finally, since dependencies are finalized for this round at this 
      // point, add them to the betweeness centrality measure on each node

      // TODO all nodes here? would remove unnecessary dep sync later, 
      // but will cause destinations (which don't need to increment bc)
      // to do extra work on each host
      Galois::do_all(
        allNodes.begin(), 
        allNodes.end(), 
        BC(&_graph), 
        Galois::loopname("BC"),
        //Galois::loopname(_graph.get_run_identifier("BC").c_str()),
        Galois::timeit(),
        Galois::no_stats()
      );
    }
  }

  /* adds dependency measure to BC measure (dependencies should be finalized,
   * i.e. no unprocessed successors on the node) */
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    if (src_data.dependency > 0) {
      Galois::add(src_data.betweeness_centrality, src_data.dependency);
      src_data.dependency = 0;
    }

  }
};
 
/******************************************************************************/
/* Main method for running */
/******************************************************************************/

int main(int argc, char** argv) {
  try {
    Galois::DistMemSys G(getStatsFile());
    DistBenchStart(argc, argv, name, desc, url);

    {
    auto& net = Galois::Runtime::getSystemNetworkInterface();
    if (net.ID == 0) {
      Galois::Runtime::reportStat("(NULL)", "Max Iterations", 
                                  (unsigned long)maxIterations, 0);
    }

    Galois::StatTimer StatTimer_graph_init("TIMER_GRAPH_INIT"),
                      StatTimer_total("TIMER_TOTAL"),
                      StatTimer_hg_init("TIMER_HG_INIT");





    StatTimer_total.start();

    std::vector<unsigned> scalefactor;
  #ifdef __GALOIS_HET_CUDA__
    const unsigned my_host_id = Galois::Runtime::getHostID();
    int gpu_device = gpudevice;

    if (num_nodes == -1) num_nodes = net.Num;
    assert((net.Num % num_nodes) == 0);

    // Parse arg string when running on multiple hosts and update/override 
    // personality with corresponding value.
    if (personality_set.length() == Galois::Runtime::NetworkInterface::Num) {
      switch (personality_set.c_str()[my_host_id]) {
        case 'g':
          personality = GPU_CUDA;
          break;
        case 'o':
          assert(0); // o currently not supported
          personality = GPU_OPENCL;
          break;
        case 'c':
        default:
          personality = CPU;
          break;
      }

      if ((personality == GPU_CUDA) && (gpu_device == -1)) {
        gpu_device = get_gpu_device_id(personality_set, num_nodes);
      }

      if ((scalecpu > 1) || (scalegpu > 1)) {
        for (unsigned i = 0; i < net.Num; ++i) {
          if (personality_set.c_str()[i % num_nodes] == 'c') 
            scalefactor.push_back(scalecpu);
          else
            scalefactor.push_back(scalegpu);
        }
      }
    }
  #endif

    StatTimer_hg_init.start();

    Graph* h_graph = nullptr;
    // uses bfs
    h_graph = constructGraph<NodeData, void, false>(scalefactor);

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

    #ifndef NDEBUG
    int counter = 0;
    for (auto i = random_sources.begin(); i != random_sources.end(); i++) {
      printf("Source #%d: %lu\n", counter, *i);
      counter++;
    }
    #endif

  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      cuda_ctx = get_CUDA_context(my_host_id);
      if (!init_CUDA_context(cuda_ctx, gpu_device))
        return -1;
      MarshalGraph m = (*h_graph).getMarshalGraph(my_host_id);
      load_graph_CUDA(cuda_ctx, m, net.Num);
    } else if (personality == GPU_OPENCL) {
      //Galois::OpenCL::cl_env.init(cldevice.Value);
    }
  #endif


    StatTimer_hg_init.stop();

    std::cout << "[" << net.ID << "] InitializeGraph::go called\n";

    StatTimer_graph_init.start();
      InitializeGraph::go((*h_graph));
    StatTimer_graph_init.stop();

    // shared DG accumulator among all steps
    Galois::DGAccumulator<uint32_t> dga;

    for (auto run = 0; run < numRuns; ++run) {
      std::cout << "[" << net.ID << "] BC::go run " << run << " called\n";
      std::string timer_str("TIMER_" + std::to_string(run));
      Galois::StatTimer StatTimer_main(timer_str.c_str());

      StatTimer_main.start();
        BC::go(*h_graph, dga);
      StatTimer_main.stop();

      // TODO sanity check setup

      // re-init graph for next run
      if ((run + 1) != numRuns) {
        Galois::Runtime::getHostBarrier().wait();
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
            Galois::Runtime::printOutput(v_out);
          }
        }
      free(v_out);
    }
    }
    Galois::Runtime::getHostBarrier().wait();

    return 0;
  } catch(const char* c) {
    std::cout << "Error: " << c << "\n";
    return 1;
  }
}
