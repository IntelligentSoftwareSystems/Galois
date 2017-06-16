/** -*- C++ -*-
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
 * Compute Betweeness-Centrality on distributed Galois 
 *
 * @author Loc Hoang <l_hoang@utexas.edu>
 */


#include <iostream>
#include <limits>
#include "Galois/Galois.h"
#include "Galois/gstl.h"
#include "Lonestar/BoilerPlate.h"
#include "Galois/Runtime/CompilerHelperFunctions.h"

#include "Galois/Runtime/dGraph_edgeCut.h"
#include "Galois/Runtime/dGraph_vertexCut.h"

#include "Galois/DistAccumulator.h"
#include "Galois/Runtime/Tracer.h"

#ifdef __GALOIS_HET_CUDA__
#include "Galois/Runtime/Cuda/cuda_device.h"
#include "gen_cuda.h"
struct CUDA_Context *cuda_ctx;

enum Personality {
   CPU, GPU_CUDA, GPU_OPENCL
};
std::string personality_str(Personality p) {
   switch (p) {
   case CPU:
      return "CPU";
   case GPU_CUDA:
      return "GPU_CUDA";
   case GPU_OPENCL:
      return "GPU_OPENCL";
   }
   assert(false && "Invalid personality");
   return "";
}
#endif

static const char* const name = "Betweeness Centrality - "
                                "Distributed Heterogeneous.";
static const char* const desc = "Betweeness Centrality on Distributed Galois.";
static const char* const url = 0;

/******************************************************************************/
/* Declaration of command line arguments */
/******************************************************************************/
namespace cll = llvm::cl;
static cll::opt<std::string> inputFile(cll::Positional,
                                       cll::desc("<input file>"),
                                       cll::Required);
static cll::opt<std::string> partFolder("partFolder",
                                        cll::desc("path to partitionFolder"),
                                        cll::init(""));
static cll::opt<unsigned int> maxIterations("maxIterations", 
                               cll::desc("Maximum iterations: Default 10000"), 
                               cll::init(10000));
static cll::opt<bool> verify("verify", 
                             cll::desc("Verify ranks by printing to "
                                       "'page_ranks.#hid.csv' file"),
                             cll::init(false));

static cll::opt<bool> enableVCut("enableVertexCut", 
                                 cll::desc("Use vertex cut for graph " 
                                           "partitioning."), 
                                 cll::init(false));
#ifdef __GALOIS_HET_CUDA__
// If running on both CPUs and GPUs, below is included
static cll::opt<int> gpudevice("gpu", 
                      cll::desc("Select GPU to run on, default is "
                                "to choose automatically"), cll::init(-1));
static cll::opt<Personality> personality("personality", 
                 cll::desc("Personality"),
                 cll::values(clEnumValN(CPU, "cpu", "Galois CPU"),
                             clEnumValN(GPU_CUDA, "gpu/cuda", "GPU/CUDA"),
                             clEnumValN(GPU_OPENCL, "gpu/opencl", "GPU/OpenCL"),
                             clEnumValEnd),
                 cll::init(CPU));
static cll::opt<std::string> personality_set("pset", 
                              cll::desc("String specifying personality for "
                                        "each host. 'c'=CPU,'g'=GPU/CUDA and "
                                        "'o'=GPU/OpenCL"),
                              cll::init(""));
static cll::opt<unsigned> scalegpu("scalegpu", 
                           cll::desc("Scale GPU workload w.r.t. CPU, default "
                                     "is proportionally equal workload to CPU "
                                     "and GPU (1)"), 
                           cll::init(1));
static cll::opt<unsigned> scalecpu("scalecpu", 
                           cll::desc("Scale CPU workload w.r.t. GPU, "
                                     "default is proportionally equal "
                                     "workload to CPU and GPU (1)"), 
                           cll::init(1));
static cll::opt<int> num_nodes("num_nodes", 
                      cll::desc("Num of physical nodes with devices (default "
                                "= num of hosts): detect GPU to use for each "
                                "host automatically"), 
                      cll::init(-1));
#endif

const unsigned int infinity = std::numeric_limits<unsigned int>::max() / 4;
static unsigned int src_node = 0;

/******************************************************************************/
/* Graph structure declarations */
/******************************************************************************/

struct NodeData {

  // SSSP vars
  std::atomic<unsigned int> current_length;
  unsigned int old_length;

  // Betweeness centrality vars
  std::atomic<unsigned int> num_shortest_paths;
  std::atomic<unsigned int> num_successors;
  std::atomic<float> dependency;
  std::atomic<unsigned int> trim;

  std::atomic<unsigned int> betweeness_centrality;
};

// second type (unsigned int) for edge weights
typedef hGraph<NodeData, unsigned int> Graph;
typedef hGraph_edgeCut<NodeData, unsigned int> Graph_edgeCut;
typedef hGraph_vertexCut<NodeData, unsigned int> Graph_vertexCut;

typedef typename Graph::GraphNode GNode;

/******************************************************************************/
/* Functors for running the algorithm */
/******************************************************************************/
struct InitializeGraph {
  Graph *graph;

  InitializeGraph(Graph* _graph) : graph(_graph){}

  /* Initialize the entire graph node-by-node */
  void static go(Graph& _graph) {
    Galois::do_all(_graph.begin(), _graph.end(), InitializeGraph{&_graph}, 
                   Galois::loopname("InitializeGraph"), 
                   Galois::numrun(_graph.get_run_identifier()));
  }

  /* Functor passed into the Galois operator to carry out initialization */
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);
    src_data.betweeness_centrality = 0;
  }
};

/* This is used to reset node data when switching to a difference source */
struct InitializeIteration {
  Graph *graph;

  InitializeIteration(Graph* _graph) : graph(_graph){}

  /* Reset graph metadata node-by-node */
  void static go(Graph& _graph) {
    Galois::do_all(_graph.begin(), _graph.end(), InitializeGraph{&_graph}, 
                   Galois::loopname("InitializeIteration"), 
                   Galois::numrun(_graph.get_run_identifier()));
  }

  /* Functor passed into the Galois operator to carry out reset of node data
   * (aside from betweeness centrality measure */
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    src_data.current_length = (graph->getGID(src) == src_node) ? 0 : infinity;
    src_data.old_length = (graph->getGID(src) == src_node) ? 0 : infinity;

    src_data.trim = 0;

    src_data.num_shortest_paths = 0; // TODO verify if diff. for source
    src_data.num_successors = 0;
    src_data.dependency = 0;
  }
};

/* Sub struct for running SSSP */
struct SSSP {
  Graph* graph;

  SSSP(Graph* _graph) : graph(_graph){}

  void static go(Graph& _graph){
    Galois::for_each(src_node, SSSP(&_graph), Galois::loopname("SSSP"));
  }

  /* Does SSSP, push based */
  void operator()(GNode src, Galois::UserContext<GNode>& ctx) const {
    NodeData& src_data = graph->getData(src);
    src_data.old_length = src_data.current_length;

    for (auto current_edge = graph->edge_begin(src), 
              end_edge = graph->edge_end(src); 
         current_edge != end_edge; 
         ++current_edge) {
      GNode dst = graph->getEdgeDst(current_edge);
      auto& dst_data = graph->getData(dst);

      unsigned int new_dist = graph->getEdgeData(current_edge) + 
                              src_data.current_length;

      Galois::atomicMin(dst_data.current_length, new_dist);

      // TODO get rid of worklist?
      if (dst_data.old_length > dst_data.current_length){
        ctx.push(graph->getGID(dst));
      }
    }
  }
};

/* Struct to get number of shortest paths as well as successors on the
 * SSSP DAG */
struct ShortestPathsAndSuccessors {
  Graph* graph;

  ShortestPathsAndSuccessors(Graph* _graph) : graph(_graph){}

  void static go(Graph& _graph){
    // Loop over all nodes in graph
    Galois::do_all(_graph.begin(), _graph.end(), 
                   ShortestPathsAndSuccessors(&_graph), 
                   Galois::loopname("ShortestPathsAndSuccessors"));
  }

  /* Summary:
   * Look at outgoing edges; see if dest is on a shortest path from src node.
   * If it is, increment the number of successors on src by 1 and
   * increment # of shortest paths on dest by 1 
   * (pull + push)
   * Read from dest
   * Writes to src and dest
   */
  // successor (on src) and # shortest paths (on dest)
  //
  // for successor, dest to src
  // for # short paths, src to dest
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    for (auto current_edge = graph->edge_begin(src), 
              end_edge = graph->edge_end(src); 
         current_edge != end_edge; 
         ++current_edge) {
      GNode dst = graph->getEdgeDst(current_edge);
      auto& dst_data = graph->getData(dst);
      unsigned int edge_weight = graph->getEdgeData(current_edge);

      if ((src_data.current_length + edge_weight) == dst_data.current_length) {
        // dest on shortest path with this node as predecessor
        Galois::atomicAdd(src_data.num_successors, (unsigned int)1);
        Galois::atomicAdd(dst_data.num_shortest_paths, (unsigned int)1);
      }
    }
  }
};

/* Uses an incremented trim value to decrement the successor: the trim value
 * has to be synchronized across ALL nodes (including slaves) */
struct SuccessorDecrement {
  Graph* graph;

  SuccessorDecrement(Graph* _graph) : graph(_graph){}

  void static go(Graph& _graph) {
    Galois::do_all(_graph.begin(), _graph.end(), SuccessorDecrement{&_graph}, 
                   Galois::loopname("SuccessorDecrement"), 
                   Galois::numrun(_graph.get_run_identifier()));
  }

  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    // decrement successor by trim then reset
    // NOTE: trim needs to be synchronized in a previous operator
    if (src_data.trim > 0) {
      src_data.num_successors -= src_data.trim;
      src_data.trim = 0;

      // multiply dependency by # of shortest paths to finalize if no more 
      // successors
      if (src_data.num_successors == 0) {
        src_data.dependency = src_data.dependency * src_data.num_shortest_paths;
      }
    }
  }
};

/* Do dependency propogation which is required for betweeness centraility
 * calculation */
struct DependencyPropogation {
  Graph* graph;
  static Galois::DGAccumulator<int> work_accumulator;

  DependencyPropogation(Graph* _graph) : graph(_graph){}

  /* Look at all nodes to do propogation until no more work is done */
  void static go(Graph& _graph) {
    unsigned int iterations = 0;
    
    do {
      _graph.set_num_iter(iterations);
      work_accumulator.reset();

      Galois::do_all(_graph.begin(), _graph.end(), 
                     DependencyPropogation(&_graph), 
                     Galois::loopname("DependencyPropogation"));
      // do successor decrementing using trim
      SuccessorDecrement::go(_graph);

      iterations++;
    } while ((iterations < maxIterations) && work_accumulator.reduce());
  }

  /* Summary:
   * TOP based, but can filter if successors = 0; can do trim based decrement
   * like kcore
   * if we have outgoing edges...
   * for each node, check if dest of edge has no successors + check if on 
   * shortest path with src as predeccesor
   *
   * if yes, then decrement src successors by 1 + grab dest delta + dest num 
   * shortest * paths and use it to increment src own delta (1 / dest short 
   * paths * (1 + delta of dest)
   *
   * sync details: push src delta changes, src sucessor changes (via trim) 
   * to ALL COPIES (not just master)
   *
   * dest to src flow for successors
   * dest to src flow for delta
   *
   * This is pull based, no? */
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    for (auto current_edge = graph->edge_begin(src), 
              end_edge = graph->edge_end(src); 
         current_edge != end_edge; 
         ++current_edge) {
      GNode dst = graph->getEdgeDst(current_edge);
      auto& dst_data = graph->getData(dst);
      unsigned int edge_weight = graph->getEdgeData(current_edge);

      // only operate if a dst has no more successors (i.e. delta finalized
      // for this round)
      if (dst_data.num_successors == 0) {
        // dest on shortest path with this node as predecessor
        if ((src_data.current_length + edge_weight) == dst_data.current_length) {

          // increment my trim for later use to decrement successor
          Galois::atomicAdd(src_data.trim, (unsigned int)1);

          // update my dependency
          src_data.dependency = src_data.dependency +
              ((1 + dst_data.dependency) / dst_data.num_shortest_paths);
        }
      }
    }
  }
};
Galois::DGAccumulator<int> DependencyPropogation::work_accumulator;

// TODO change this
struct BC {
  Graph* graph;

  BC(Graph* _graph) : graph(_graph){}

  void static go(Graph& _graph){
    // NOTE: kill this loop if you just want to run BC with 1 source node for
    // SSSP
    // TODO: find a way to loop over all nodes for SSSP; do_all maybe, but not
    // in parallel?
    //for (auto i = _graph.begin(); i != _graph.end(); i++) {
      // change source nodes for this iteration of SSSP
      //src_node = _graph.getGID(i);

      // reset the graph aside from the between-cent measure
      InitializeIteration::go(_graph);

      // get SSSP on the current graph (sync should be handled in it)
      SSSP::go(_graph);

      // calculate the num of shortest paths for each node after SSSP
      // convergence + count the num of successors on SSSP DAG for each node
      ShortestPathsAndSuccessors::go(_graph);

      // do between-cent calculations for this iteration 
      DependencyPropogation::go(_graph);

      // finally, since dependencies are finalized for this round at this 
      // point, add them to the betweeness centrality measure on each node
      Galois::do_all(_graph.begin(), _graph.end(), BC(&_graph), 
                     Galois::loopname("BC"));
    //}
  }

  /* adds dependency measure to BC measure (dependencies should be finalized,
   * i.e. no unprocessed successors on the node) */
  void operator()(GNode src) const {
    NodeData& src_node = graph->getData(src);

    src_node.betweeness_centrality += src_node.dependency;
  }
};
 
/******************************************************************************/
/* Main method for running */
/******************************************************************************/

int main(int argc, char** argv) {
  try {
    LonestarStart(argc, argv, name, desc, url);
    Galois::Runtime::reportStat("(NULL)", "Max Iterations", 
                                (unsigned long)maxIterations, 0);
    Galois::StatManager statManager;


    Galois::StatTimer StatTimer_graph_init("TIMER_GRAPH_INIT"),
                      StatTimer_total("TIMER_TOTAL"),
                      StatTimer_hg_init("TIMER_HG_INIT");
    StatTimer_total.start();

    std::vector<unsigned> scalefactor;
#ifdef __GALOIS_HET_CUDA__
    const unsigned my_host_id = Galois::Runtime::getHostID();
    int gpu_device = gpudevice;
    // Parse arg string when running on multiple hosts and update/override 
    // personality with corresponding value.
    if (personality_set.length() == Galois::Runtime::NetworkInterface::Num) {
      switch (personality_set.c_str()[my_host_id]) {
        case 'g':
          personality = GPU_CUDA;
          break;
        case 'o':
          assert(0); // o currently not supported (apparently)
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

      for (unsigned i = 0; i < personality_set.length(); ++i) {
        if (personality_set.c_str()[i] == 'c') 
          scalefactor.push_back(scalecpu);
        else
          scalefactor.push_back(scalegpu);
      }
    }
#endif
    auto& net = Galois::Runtime::getSystemNetworkInterface();

    StatTimer_hg_init.start();

    Graph* h_graph;
    if (enableVCut) {
      h_graph = new Graph_vertexCut(inputFile, partFolder, net.ID, net.Num,
                               scalefactor);
    } else {
      h_graph = new Graph_edgeCut(inputFile, partFolder, net.ID, net.Num,
                             scalefactor);
    }

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

    for (auto run = 0; run < numRuns; ++run) {
      std::cout << "[" << net.ID << "] BC::go run " << run << " called\n";
      std::string timer_str("TIMER_" + std::to_string(run));
      Galois::StatTimer StatTimer_main(timer_str.c_str());

      StatTimer_main.start();
        BC::go((*h_graph));
      StatTimer_main.stop();

      // re-init graph for next run
      if ((run + 1) != numRuns) {
        Galois::Runtime::getHostBarrier().wait();
        (*h_graph).reset_num_iter(run+1);
        InitializeGraph::go((*h_graph));
      }
    }

    StatTimer_total.stop();

    // Verify, i.e. print out graph data for examination
    if (verify) {
#ifdef __GALOIS_HET_CUDA__
      if (personality == CPU) { 
#endif
        for (auto ii = (*h_graph).begin(); ii != (*h_graph).end(); ++ii) {
          if ((*h_graph).isOwned((*h_graph).getGID(*ii))) 
            Galois::Runtime::printOutput("% %\n", (*h_graph).getGID(*ii), 
                                 (*h_graph).getData(*ii).betweeness_centrality);
        }
#ifdef __GALOIS_HET_CUDA__
      } else if (personality == GPU_CUDA) {
        // TODO not changed yet
        for(auto ii = (*h_graph).begin(); ii != (*h_graph).end(); ++ii) {
          if ((*h_graph).isOwned((*h_graph).getGID(*ii))) 
            Galois::Runtime::printOutput("% %\n", (*h_graph).getGID(*ii), 
                                     get_node_dist_current_cuda(cuda_ctx, *ii));
        }
      }
#endif
    }
    return 0;
  } catch(const char* c) {
    std::cerr << "Error: " << c << "\n";
    return 1;
  }
}
