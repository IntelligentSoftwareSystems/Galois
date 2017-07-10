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
 * Compute Betweeness-Centrality on distributed Galois 
 *
 * @author Loc Hoang <l_hoang@utexas.edu>
 */

// TODO reason about incoming edge cut syncs

#include <iostream>
#include <limits>
#include "Galois/Galois.h"
#include "Galois/gstl.h"
#include "Lonestar/BoilerPlate.h"
#include "Galois/Runtime/CompilerHelperFunctions.h"

#include "Galois/Runtime/dGraph_edgeCut.h"
#include "Galois/Runtime/dGraph_cartesianCut.h"
#include "Galois/Runtime/dGraph_hybridCut.h"

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

enum VertexCut {
  PL_VCUT, CART_VCUT
};

static const char* const name = "Betweeness Centrality (OEC) - "
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
static cll::opt<bool> transpose("transpose", 
                                cll::desc("transpose the graph in memory after "
                                          "partitioning"),
                                cll::init(false));
static cll::opt<bool> verify("verify", 
                             cll::desc("Verify ranks by printing to "
                                       "'page_ranks.#hid.csv' file"),
                             cll::init(false));
static cll::opt<bool> enableVCut("enableVertexCut", 
                                 cll::desc("Use vertex cut for graph " 
                                           "partitioning."), 
                                 cll::init(false));
static cll::opt<unsigned int> VCutThreshold("VCutThreshold", 
                                            cll::desc("Threshold for high "
                                                      "degree edges."),
                                            cll::init(100));
static cll::opt<VertexCut> vertexcut("vertexcut", 
                                     cll::desc("Type of vertex cut."),
                                     cll::values(clEnumValN(PL_VCUT, "pl_vcut",
                                                        "Powerlyra Vertex Cut"),
                                                 clEnumValN(CART_VCUT, 
                                                   "cart_vcut", 
                                                   "Cartesian Vertex Cut"),
                                                 clEnumValEnd),
                                     cll::init(PL_VCUT));
static cll::opt<bool> singleSourceBC("singleSource", 
                                cll::desc("Use for single source BC"),
                                cll::init(false));
static cll::opt<unsigned int> singleSourceNode("srcNodeId", 
                                cll::desc("Source node used for single source "
                                          "betweeness-centraility"),
                                cll::init(0));

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
static unsigned int current_src_node = 0;

/******************************************************************************/
/* Graph structure declarations */
/******************************************************************************/

struct NodeData {
  // SSSP vars
  std::atomic<unsigned int> current_length;

  unsigned int old_length;

  // Betweeness centrality vars
  std::atomic<unsigned int> num_shortest_paths;
  unsigned int num_successors;
  std::atomic<unsigned int> num_predecessors;
  std::atomic<unsigned int> trim;
  std::atomic<unsigned int> to_add;
  std::atomic<float> to_add_float;

  float dependency;
  float betweeness_centrality;

  // used to determine if data has been propogated yet
  unsigned char propogation_flag;
};

// second type (unsigned int) is for edge weights
typedef hGraph<NodeData, unsigned int> Graph;
typedef hGraph_edgeCut<NodeData, unsigned int> Graph_edgeCut;
typedef hGraph_vertexCut<NodeData, unsigned int> Graph_vertexCut;
typedef hGraph_cartesianCut<NodeData, unsigned int> Graph_cartesianCut;

typedef typename Graph::GraphNode GNode;

// bitsets for tracking updates
Galois::DynamicBitSet bitset_to_add;
Galois::DynamicBitSet bitset_to_add_float;
Galois::DynamicBitSet bitset_num_shortest_paths;
Galois::DynamicBitSet bitset_num_successors;
Galois::DynamicBitSet bitset_num_predecessors;
Galois::DynamicBitSet bitset_trim;
Galois::DynamicBitSet bitset_current_length;
Galois::DynamicBitSet bitset_propogation_flag;
Galois::DynamicBitSet bitset_dependency;

// sync structures
#include "gen_sync.hh"

/******************************************************************************/
/* Functors for running the algorithm */
/******************************************************************************/
struct InitializeGraph {
  Graph *graph;

  InitializeGraph(Graph* _graph) : graph(_graph){}

  /* Initialize the entire graph node-by-node */
  void static go(Graph& _graph) {
    #ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) {
        std::string impl_str("CUDA_DO_ALL_IMPL_InitializeGraph_" + 
                             (_graph.get_run_identifier()));
        Galois::StatTimer StatTimer_cuda(impl_str.c_str());
        StatTimer_cuda.start();
        InitializeGraph_all_cuda(cuda_ctx);
        StatTimer_cuda.stop();
      } else if (personality == CPU)
    #endif

    Galois::do_all(_graph.begin(), _graph.end(), InitializeGraph{&_graph}, 
                   Galois::loopname("InitializeGraph"), 
                   Galois::numrun(_graph.get_run_identifier()));
    // doesn't need to be sync'd on destinations: we never read it on 
    // destination nor write to it
  }

  /* Functor passed into the Galois operator to carry out initialization;
   * Reset bc measure */
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);
    src_data.betweeness_centrality = 0;
  }
};

/* This is used to reset node data when switching to a difference source */
struct InitializeIteration {
  const unsigned int &local_infinity;
  const unsigned int &local_current_src_node;
  Graph *graph;

  InitializeIteration(const unsigned int &_local_infinity,
                      const unsigned int &_local_current_src_node,
                      Graph* _graph) : 
                       local_infinity(_local_infinity),
                       local_current_src_node(_local_current_src_node),
                       graph(_graph){}

  /* Reset graph metadata node-by-node */
  void static go(Graph& _graph) {
    #ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) {
        std::string impl_str("CUDA_DO_ALL_IMPL_InitializeIteration_" + 
                             (_graph.get_run_identifier()));
        Galois::StatTimer StatTimer_cuda(impl_str.c_str());
        StatTimer_cuda.start();
        InitializeIteration_all_cuda(infinity, current_src_node, cuda_ctx);
        StatTimer_cuda.stop();
      } else if (personality == CPU)
    #endif
    Galois::do_all(_graph.begin(), _graph.end(), 
                   InitializeIteration{infinity, current_src_node, &_graph},
                   Galois::loopname("InitializeIteration"), 
                   Galois::numrun(_graph.get_run_identifier()));

    // broadcast ALL reset values (inefficient, but this is initialization)
    // note no bitset is used here
    _graph.sync<writeSource, readDestination, Reduce_set_current_length, 
                Broadcast_current_length>("InitializeIteration");
    _graph.sync<writeSource, readDestination, Reduce_set_propogation_flag, 
                Broadcast_propogation_flag>("InitializeIteration");
    _graph.sync<writeSource, readDestination, Reduce_set_num_successors, 
                Broadcast_num_successors>("InitializeIteration");
    _graph.sync<writeSource, readDestination, Reduce_set_old_length,
                Broadcast_old_length>("InitializeIteration");
    _graph.sync<writeSource, readDestination, Reduce_set_num_predecessors, 
                Broadcast_num_predecessors>("InitializeIteration");
    _graph.sync<writeSource, readDestination, Reduce_set_trim,
                Broadcast_trim>("InitializeIteration");
    _graph.sync<writeSource, readDestination, Reduce_set_num_shortest_paths,
                Broadcast_num_shortest_paths>("InitializeIteration");
    _graph.sync<writeSource, readDestination, Reduce_set_dependency,
                Broadcast_dependency>("InitializeIteration");
    _graph.sync<writeSource, readDestination, Reduce_set_to_add,
                Broadcast_to_add>("InitializeIteration");
    _graph.sync<writeSource, readDestination, Reduce_set_to_add_float,
                Broadcast_to_add_float>("InitializeIteration");

    // manual reset of all bit fields
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

    //if (personality == GPU_CUDA) {
    //  char test[100];
    //  for (auto ii = (_graph).begin(); ii != (_graph).ghost_end(); ++ii) {
    //    if ((_graph).isOwned((_graph).getGID(*ii))) {
    //      sprintf(test, "bnsp %lu %u\n", (_graph).getGID(*ii),
    //              get_node_num_shortest_paths_cuda(cuda_ctx, *ii));
    //      Galois::Runtime::printOutput(test);
    //    }
    //  }
    //} else {
    //  char test[100];
    //  for (auto ii = (_graph).begin(); ii != (_graph).ghost_end(); ++ii) {
    //    if ((_graph).isOwned((_graph).getGID(*ii))) {
    //      sprintf(test, "bnsp %lu %u\n", (_graph).getGID(*ii),
    //              _graph.getData(*ii).num_shortest_paths.load());
    //      Galois::Runtime::printOutput(test);
    //    }
    //  }
    //}

  }

  /* Functor passed into the Galois operator to carry out reset of node data
   * (aside from betweeness centrality measure */
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    src_data.current_length = (graph->getGID(src) == local_current_src_node) ? 0 : local_infinity;
    src_data.old_length = (graph->getGID(src) == local_current_src_node) ? 0 : local_infinity;

    src_data.to_add = 0;
    src_data.to_add_float = 0;

    // set to true = "I have propogated" i.e. don't propogate anymore
    // set to false = "I have not propogated and/or I cannot propogate"
    src_data.propogation_flag = false; 

    // set num to 1 on source so that it can propogate it across nodes later
    // note source will not have sigma accessed anyways (at least it shouldn't)
    src_data.num_shortest_paths = (graph->getGID(src) == local_current_src_node) ? 1 : 0;

    src_data.num_successors = 0;
    src_data.num_predecessors = 0;
    src_data.dependency = 0;
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
        std::string impl_str("CUDA_DO_ALL_IMPL_FirstIterationSSSP_" + 
                             (_graph.get_run_identifier()));
        Galois::StatTimer StatTimer_cuda(impl_str.c_str());
        StatTimer_cuda.start();
        FirstIterationSSSP_cuda(__begin, __end, cuda_ctx);
        StatTimer_cuda.stop();
      } else if (personality == CPU)
    #endif
    Galois::do_all(boost::make_counting_iterator(__begin), 
                   boost::make_counting_iterator(__end), 
                   FirstIterationSSSP(&_graph),
                   Galois::loopname("FirstIterationSSSP"));

    //// Next op will read src, current length
    //if (_graph.isLocal(21848)) {
    //  printf("[%u]is 21848 set; first sync? %d, owned %d\n", _graph.id, 
    //         bitset_current_length.test(_graph.getLID(21848)),
    //         _graph.isOwned(21848));
    //}

    _graph.sync<writeDestination, readSource, Reduce_min_current_length, 
                Broadcast_current_length, Bitset_current_length>(
                "FirstIterationSSSP");
    // if this is a vertex cut then it would reset the flag for broadcast
    // dest

    //if (personality == GPU_CUDA) {
    // char test[100];
    //  for (auto ii = (_graph).begin(); ii != (_graph).ghost_end(); ++ii) {
    //      sprintf(test, "[%u]flen %lu %u\n", _graph.id, (_graph).getGID(*ii),
    //              get_node_current_length_cuda(cuda_ctx, *ii));
    //      Galois::Runtime::printOutput(test);
    //  }
    //} else {
    //  char test[100];
    //  for (auto ii = (_graph).begin(); ii != (_graph).ghost_end(); ++ii) {
    //      sprintf(test, "[%u]flen %lu %u\n", _graph.id, (_graph).getGID(*ii),
    //              _graph.getData(*ii).current_length.load());
    //      Galois::Runtime::printOutput(test);
    //  }
    //}
  }

  /* Does SSSP, push/filter based (actually it does BFS...) */
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    for (auto current_edge = graph->edge_begin(src), 
              end_edge = graph->edge_end(src); 
         current_edge != end_edge; 
         ++current_edge) {
      //if (graph->isLocal(src)) {
      //  if (graph->getGID(src) == 21848) {
      //    printf("[%u]edges?\n", graph->id);
      //  }
      //}

      GNode dst = graph->getEdgeDst(current_edge);
      auto& dst_data = graph->getData(dst);

      // For SSSP (uses the edge weight)
      //unsigned int new_dist = graph->getEdgeData(current_edge) + 
      //                        src_data.current_length;
      // BFS 
      unsigned int new_dist = 1 + src_data.current_length;
      Galois::atomicMin(dst_data.current_length, new_dist);

      bitset_current_length.set(dst);
    }
  }
};

/* Sub struct for running SSSP (beyond 1st iteration) */
struct SSSP {
  Graph* graph;
  static Galois::DGAccumulator<unsigned int> DGAccumulator_accum;
  SSSP(Graph* _graph) : graph(_graph){}

  void static go(Graph& _graph) {
    FirstIterationSSSP::go(_graph);

    // starts at 1 since FirstSSSP takes care of the first one
    unsigned int iterations = 1;

    unsigned int accum_result;

    do {
      //printf("[%u] begin iter %u\n", _graph.id, iterations);
      _graph.set_num_iter(iterations);
      DGAccumulator_accum.reset();

    #ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) {
        std::string impl_str("CUDA_DO_ALL_IMPL_SSSP_" + 
                             (_graph.get_run_identifier()));
        Galois::StatTimer StatTimer_cuda(impl_str.c_str());
        StatTimer_cuda.start();
        int __retval = 0;
        SSSP_all_cuda(__retval, cuda_ctx);
        //std::cout << __retval << "\n";
        DGAccumulator_accum += __retval;
        StatTimer_cuda.stop();
      } else if (personality == CPU)
    #endif
      //if (_graph.isLocal(21848)) {
      //  printf("[%u]is 21848 set; before operator? %d, owned %d\n", _graph.id, 
      //         bitset_current_length.test(_graph.getLID(21848)),
      //         _graph.isOwned(21848));
      //}
      Galois::do_all(_graph.begin(), _graph.end(), SSSP(&_graph), 
                     Galois::loopname("SSSP"));

      iterations++;

      accum_result = DGAccumulator_accum.reduce();

      if (accum_result) {
        //if (_graph.isLocal(21848)) {
        //  printf("[%u]is 21848 set; read source sync? %d\n", _graph.id, 
        //         bitset_current_length.test(_graph.getLID(21848)));
        //}
        _graph.sync<writeDestination, readSource, Reduce_min_current_length, 
                    Broadcast_current_length, Bitset_current_length>("SSSP");
        //_graph.sync<writeDestination, readSource, Reduce_min_current_length, 
        //            Broadcast_current_length>("SSSP");
      } else {
        // write destination, read any, fails.....

        //if (_graph.isLocal(21848)) {
        //  printf("[%u]is 21848 set; all sync? %d\n", _graph.id, 
        //         bitset_current_length.test(_graph.getLID(21848)));
        //}

        // sync src and dst
        if (_graph.is_vertex_cut()) {
          // no bitset used = sync all; at time of writing, vertex cut
          // syncs cause the bit to be reset prematurely, so using the bitset
          // will lead to incorrect results as it will not sync what is
          // necessary
          _graph.sync<writeDestination, readAny, Reduce_min_current_length, 
                       Broadcast_current_length>("SSSP");
          //_graph.sync<writeDestination, readAny, Reduce_min_current_length, 
          //            Broadcast_current_length, Bitset_current_length>("SSSP");
        } else {
          _graph.sync<writeDestination, readAny, Reduce_min_current_length, 
                      Broadcast_current_length, Bitset_current_length>("SSSP");
        }
      }
    } while (accum_result);

    //if (personality == GPU_CUDA) {
    // char test[100];
    //  for (auto ii = (_graph).begin(); ii != (_graph).ghost_end(); ++ii) {
    //      sprintf(test, "[%u]clen %lu %u\n", _graph.id, (_graph).getGID(*ii),
    //              get_node_current_length_cuda(cuda_ctx, *ii));
    //      Galois::Runtime::printOutput(test);
    //  }
    //} else {
    //  char test[100];
    //  for (auto ii = (_graph).begin(); ii != (_graph).ghost_end(); ++ii) {
    //      sprintf(test, "[%u]clen %lu %u\n", _graph.id, (_graph).getGID(*ii),
    //              _graph.getData(*ii).current_length.load());
    //      Galois::Runtime::printOutput(test);
    //  }
    //}
  }

  /* Does SSSP (actually BFS at the moment), push/filter based */
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

        //if (graph->getGID(dst) == 21848) {
        //  printf("[%u]source is %lu\n", graph->id, graph->getGID(src));
        //}

        // SSSP
        //unsigned int new_dist = graph->getEdgeData(current_edge) + 
        //                        src_data.current_length;

        // BFS
        unsigned int new_dist = 1 + src_data.current_length;
        unsigned int old = Galois::atomicMin(dst_data.current_length, new_dist);

        if (old > new_dist) {
          bitset_current_length.set(dst);
          DGAccumulator_accum += 1;
        }
      }
    }
  }
};
Galois::DGAccumulator<unsigned int> SSSP::DGAccumulator_accum;

/* Struct to get pred and succon the SSSP DAG */
struct PredAndSucc {
  Graph* graph;

  PredAndSucc(Graph* _graph) : graph(_graph){}

  void static go(Graph& _graph){
    // Loop over all nodes in graph iteratively
    #ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) {
        std::string impl_str("CUDA_DO_ALL_IMPL_PredAndSucc_" + 
                             (_graph.get_run_identifier()));
        Galois::StatTimer StatTimer_cuda(impl_str.c_str());
        StatTimer_cuda.start();
        PredAndSucc_all_cuda(cuda_ctx);
        StatTimer_cuda.stop();
      } else if (personality == CPU)
    #endif
    Galois::do_all(_graph.begin(), _graph.end(), PredAndSucc(&_graph), 
                   Galois::loopname("PredAndSucc"));

    // sync for use in NumShortPath calculation
    _graph.sync<writeDestination, readSource, Reduce_add_num_predecessors, 
                Broadcast_num_predecessors, Bitset_num_predecessors>("PredAndSucc");

    // sync now for later DependencyPropogation use (read src/dst)
    _graph.sync<writeSource, readAny, Reduce_add_num_successors, 
                Broadcast_num_successors, Bitset_num_successors>("PredAndSucc");

    //if (personality == GPU_CUDA) {
    //  char test[100];
    //  for (auto ii = (_graph).begin(); ii != (_graph).ghost_end(); ++ii) {
    //    if ((_graph).isOwned((_graph).getGID(*ii))) {
    //      sprintf(test, "pred %lu %u\n", (_graph).getGID(*ii),
    //              get_node_num_predecessors_cuda(cuda_ctx, *ii));
    //      Galois::Runtime::printOutput(test);
    //    }
    //  }
    //} else {
    //  char test[100];
    //  for (auto ii = (_graph).begin(); ii != (_graph).ghost_end(); ++ii) {
    //    if ((_graph).isOwned((_graph).getGID(*ii))) {
    //      sprintf(test, "pred %lu %u\n", (_graph).getGID(*ii),
    //              _graph.getData(*ii).num_predecessors.load());
    //      Galois::Runtime::printOutput(test);
    //    }
    //  }
    //}

    //if (personality == GPU_CUDA) {
    //  char test[100];
    //  for (auto ii = (_graph).begin(); ii != (_graph).ghost_end(); ++ii) {
    //    if ((_graph).isOwned((_graph).getGID(*ii))) {
    //      sprintf(test, "succ %lu %u\n", (_graph).getGID(*ii),
    //              get_node_num_successors_cuda(cuda_ctx, *ii));
    //      Galois::Runtime::printOutput(test);
    //    }
    //  }
    //} else {
    //  char test[100];
    //  for (auto ii = (_graph).begin(); ii != (_graph).ghost_end(); ++ii) {
    //    if ((_graph).isOwned((_graph).getGID(*ii))) {
    //      sprintf(test, "succ %lu %u\n", (_graph).getGID(*ii),
    //              _graph.getData(*ii).num_successors);
    //      Galois::Runtime::printOutput(test);
    //    }
    //  }
    //}
  }

  /* Summary:
   * Look at outgoing edges; see if dest is on a shortest path from src node.
   * If it is, increment the number of successors on src by 1 and
   * increment # of pred on dest by 1 
   */
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    for (auto current_edge = graph->edge_begin(src), 
              end_edge = graph->edge_end(src); 
         current_edge != end_edge; 
         ++current_edge) {
      GNode dst = graph->getEdgeDst(current_edge);
      auto& dst_data = graph->getData(dst);

      // SSSP
      //unsigned int edge_weight = graph->getEdgeData(current_edge);
      // BFS
      unsigned int edge_weight = 1;

      if ((src_data.current_length + edge_weight) == dst_data.current_length) {
        // dest on shortest path with this node as predecessor
        Galois::add(src_data.num_successors, (unsigned int)1);
        Galois::atomicAdd(dst_data.num_predecessors, (unsigned int)1);

        bitset_num_successors.set(src);
        bitset_num_predecessors.set(dst);
      }
    }
  }
};

/* Uses an incremented trim value to decrement the predecessor: the trim value
 * has to be synchronized across ALL nodes (including mirrors) */
struct PredecessorDecrement {
  Graph* graph;

  PredecessorDecrement(Graph* _graph) : graph(_graph){}

  void static go(Graph& _graph) {
    // DO NOT DO A BITSET RESET HERE BECAUSE IT WILL BE REUSED BY THE NEXT STEP
    // (updates to trim and pred are on the same nodes)

    #ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) {
        std::string impl_str("CUDA_DO_ALL_IMPL_PredecessorDecrement_" + 
                             (_graph.get_run_identifier()));
        Galois::StatTimer StatTimer_cuda(impl_str.c_str());
        StatTimer_cuda.start();
        PredecessorDecrement_all_cuda(cuda_ctx);
        StatTimer_cuda.stop();
      } else if (personality == CPU)
    #endif
    Galois::do_all(_graph.begin(), _graph.end(), PredecessorDecrement{&_graph}, 
                   Galois::loopname("PredecessorDecrement"), 
                   Galois::numrun(_graph.get_run_identifier()));

    // predecessors does not require syncing as syncing trim accomplishes the
    // same effect; as a result, flags are synced as well on sources
    // additionally, all sources will have trim from last sync, meaning all
    // sources will reset trim to 0 
    // Since we only read trim at source, this is sufficient to "sync"
  }

  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    // decrement predecessor by trim then reset
    if (src_data.trim > 0) {
      if (src_data.trim > src_data.num_predecessors) {
        std::cout << "ISSUE P: src " << src << " " << src_data.trim << " " << 
                                     src_data.num_predecessors << "\n";
        abort();                                    
      }

      src_data.num_predecessors = src_data.num_predecessors - src_data.trim;
      src_data.trim = 0;

      // if I hit 0 predecessors, set the flag to false (i.e. says
      // I need to propogate my value)
      // NOTE: actually, at the moment, it's set to false by default, but
      // no harm keeping this here
      if (src_data.num_predecessors == 0) {
        src_data.propogation_flag = false;
      }
    }
  }
};

/* Increment num_shortest_paths using the to_add variable which should be 
 * sync'd among source nodes */
struct PathsIncrement {
  Graph* graph;
  PathsIncrement(Graph* _graph) : graph(_graph) {}

  void static go(Graph& _graph) {
    #ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) {
        std::string impl_str("CUDA_DO_ALL_IMPL_PathsIncrement_" + 
                             (_graph.get_run_identifier()));
        Galois::StatTimer StatTimer_cuda(impl_str.c_str());
        StatTimer_cuda.start();
        PathsIncrement_all_cuda(cuda_ctx);
        StatTimer_cuda.stop();
      } else if (personality == CPU)
    #endif
    Galois::do_all(_graph.begin(), _graph.end(), PathsIncrement{&_graph}, 
                   Galois::loopname("PathsIncrement"), 
                   Galois::numrun(_graph.get_run_identifier()));
  }

  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    // increment num_shortest_paths by to_add then reset
    if (src_data.to_add > 0) {
      src_data.num_shortest_paths += src_data.to_add;
      src_data.to_add = 0;

      // this bitset is used in the NumShortestPaths go method to 
      // sync to destinations
      bitset_num_shortest_paths.set(src);
    }
  }
};


/* Calculate the number of shortest paths for each node */
struct NumShortestPaths {
  Graph* graph;
  static Galois::DGAccumulator<unsigned int> DGAccumulator_accum;

  NumShortestPaths(Graph* _graph) : graph(_graph){}

  void static go(Graph& _graph) {
    unsigned int iterations = 0;

    unsigned int accum_result;

    do {
      _graph.set_num_iter(iterations);
      DGAccumulator_accum.reset();

    #ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) {
        std::string impl_str("CUDA_DO_ALL_IMPL_NumShortestPaths_" + 
                             (_graph.get_run_identifier()));
        Galois::StatTimer StatTimer_cuda(impl_str.c_str());
        StatTimer_cuda.start();
        int __retval = 0;
        NumShortestPaths_all_cuda(__retval, cuda_ctx);
        DGAccumulator_accum += __retval;
        StatTimer_cuda.stop();
      } else if (personality == CPU)
    #endif
      Galois::do_all(_graph.begin(), _graph.end(), 
                     NumShortestPaths(&_graph), 
                     Galois::loopname("NumShortestPaths"));

      _graph.sync<writeDestination, readSource, Reduce_add_trim, 
                  Broadcast_trim, Bitset_trim>("NumShortestPaths");

    //if (personality == GPU_CUDA) {
    //  char test[100];
    //  for (auto ii = (_graph).begin(); ii != (_graph).ghost_end(); ++ii) {
    //    if ((_graph).isOwned((_graph).getGID(*ii))) {
    //      sprintf(test, "trimnsp%u %lu %u\n", iterations,
    //              (_graph).getGID(*ii),
    //              get_node_trim_cuda(cuda_ctx, *ii));
    //      Galois::Runtime::printOutput(test);
    //    }
    //  }
    //} else {
    //  char test[100];
    //  for (auto ii = (_graph).begin(); ii != (_graph).ghost_end(); ++ii) {
    //    if ((_graph).isOwned((_graph).getGID(*ii))) {
    //      sprintf(test, "trimnsp%u %lu %u\n", iterations,
    //              (_graph).getGID(*ii),
    //              _graph.getData(*ii).trim.load());
    //      Galois::Runtime::printOutput(test);
    //    }
    //  }
    //}

      // do predecessor decrementing using trim
      PredecessorDecrement::go(_graph);

      // sync to_adds on source
      _graph.sync<writeDestination, readSource, Reduce_add_to_add, 
                  Broadcast_to_add, Bitset_to_add>("NumShortestPaths");
      // do num_shortest_paths incrementing using to_add 
      PathsIncrement::go(_graph);

      iterations++;

      accum_result = DGAccumulator_accum.reduce();

      // sync num_short_paths on dest if necessary (will be sync'd on source
      // already, i.e. all sources should already have the correct value)
      if (!accum_result) {
        _graph.sync<writeSource, readDestination, Reduce_set_num_shortest_paths, 
                    Broadcast_num_shortest_paths, 
                    Bitset_num_shortest_paths>("NumShortestPaths");
      }

    //if (personality == GPU_CUDA) {
    //  char test[100];
    //  for (auto ii = (_graph).begin(); ii != (_graph).ghost_end(); ++ii) {
    //    if ((_graph).isOwned((_graph).getGID(*ii))) {
    //      sprintf(test, "nspinloop%u %lu %u\n", iterations, (_graph).getGID(*ii),
    //              get_node_num_shortest_paths_cuda(cuda_ctx, *ii));
    //      Galois::Runtime::printOutput(test);
    //    }
    //  }
    //} else {
    //  char test[100];
    //  for (auto ii = (_graph).begin(); ii != (_graph).ghost_end(); ++ii) {
    //    if ((_graph).isOwned((_graph).getGID(*ii))) {
    //      sprintf(test, "nspinloop%u %lu %u\n", iterations, (_graph).getGID(*ii),
    //              _graph.getData(*ii).num_shortest_paths.load());
    //      Galois::Runtime::printOutput(test);
    //    }
    //  }
    //}
    } while (accum_result);
    //_graph.sync<writeDestination, readDestination, Reduce_num_shortest_paths, 
    //            Broadcast_num_shortest_paths>("NumShortestPaths");

    //if (personality == GPU_CUDA) {
    //  char test[100];
    //  for (auto ii = (_graph).begin(); ii != (_graph).ghost_end(); ++ii) {
    //    if ((_graph).isOwned((_graph).getGID(*ii))) {
    //      sprintf(test, "nsp %lu %u\n", (_graph).getGID(*ii),
    //              get_node_num_shortest_paths_cuda(cuda_ctx, *ii));
    //      Galois::Runtime::printOutput(test);
    //    }
    //  }
    //} else {
    //  char test[100];
    //  for (auto ii = (_graph).begin(); ii != (_graph).ghost_end(); ++ii) {
    //    if ((_graph).isOwned((_graph).getGID(*ii))) {
    //      sprintf(test, "nsp %lu %u\n", (_graph).getGID(*ii),
    //              _graph.getData(*ii).num_shortest_paths.load());
    //      Galois::Runtime::printOutput(test);
    //    }
    //  }
    //}
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

    if (src_data.num_predecessors == 0 && !src_data.propogation_flag) {
      for (auto current_edge = graph->edge_begin(src), 
                end_edge = graph->edge_end(src); 
           current_edge != end_edge; 
           ++current_edge) {
        GNode dst = graph->getEdgeDst(current_edge);
        auto& dst_data = graph->getData(dst);

        // SSSP
        //unsigned int edge_weight = graph->getEdgeData(current_edge);

        // BFS
        unsigned int edge_weight = 1;

        unsigned int paths_to_add = src_data.num_shortest_paths;

        if ((src_data.current_length + edge_weight) == dst_data.current_length) {
          // need to add my num_short_paths to dest
          Galois::atomicAdd(dst_data.to_add, paths_to_add);
          // increment dst trim so it can decrement predecessor
          Galois::atomicAdd(dst_data.trim, (unsigned int)1);

          bitset_to_add.set(dst);
          bitset_trim.set(dst);

          DGAccumulator_accum += 1;
        }
      }

      // set flag so that it doesn't propogate its info more than once
      src_data.propogation_flag = true;
    }
  }
};
Galois::DGAccumulator<unsigned int> NumShortestPaths::DGAccumulator_accum;

/* Loop over all nodes and set the prop flag to false in prep for more
 * propogation later */
struct PropFlagReset {
  Graph* graph;

  PropFlagReset(Graph* _graph) : graph(_graph){}

  void static go(Graph& _graph) {
    #ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) {
        std::string impl_str("CUDA_DO_ALL_IMPL_PropFlagReset_" + 
                             (_graph.get_run_identifier()));
        Galois::StatTimer StatTimer_cuda(impl_str.c_str());
        StatTimer_cuda.start();
        PropFlagReset_all_cuda(cuda_ctx);
        StatTimer_cuda.stop();
      } else if (personality == CPU)
    #endif
    Galois::do_all(_graph.begin(), _graph.end(), PropFlagReset{&_graph}, 
                   Galois::loopname("PropFlagReset"), 
                   Galois::numrun(_graph.get_run_identifier()));

    // flag read later on both source and dest, but only need to sync
    // dst as source should all be set
    _graph.sync<writeSource, readDestination, Reduce_set_propogation_flag, 
                Broadcast_propogation_flag,
                Bitset_propogation_flag>("PropFlagReset");

    //if (personality == GPU_CUDA) {
    //  char test[100];
    //  for (auto ii = (_graph).begin(); ii != (_graph).ghost_end(); ++ii) {
    //    if ((_graph).isOwned((_graph).getGID(*ii))) {
    //      sprintf(test, "pflag %lu %u\n", (_graph).getGID(*ii),
    //              get_node_propogation_flag_cuda(cuda_ctx, *ii));
    //      Galois::Runtime::printOutput(test);
    //    }
    //  }
    //} else {
    //  char test[100];
    //  for (auto ii = (_graph).begin(); ii != (_graph).ghost_end(); ++ii) {
    //    if ((_graph).isOwned((_graph).getGID(*ii))) {
    //      sprintf(test, "pflag %lu %u\n", (_graph).getGID(*ii),
    //              _graph.getData(*ii).propogation_flag);
    //      Galois::Runtime::printOutput(test);
    //    }
    //  }
    //}
  }

  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);
    src_data.propogation_flag = false;
    bitset_propogation_flag.set(src);
  }
};

/* Uses an incremented trim value to decrement the successor: the trim value
 * has to be synchronized across ALL nodes (including mirrors) */
struct SuccessorDecrement {
  Graph* graph;

  SuccessorDecrement(Graph* _graph) : graph(_graph){}

  void static go(Graph& _graph) {
    #ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) {
        std::string impl_str("CUDA_DO_ALL_IMPL_SuccessorDecrement_" + 
                             (_graph.get_run_identifier()));
        Galois::StatTimer StatTimer_cuda(impl_str.c_str());
        StatTimer_cuda.start();
        SuccessorDecrement_all_cuda(cuda_ctx);
        StatTimer_cuda.stop();
      } else if (personality == CPU)
    #endif
    Galois::do_all(_graph.begin(), _graph.end(), SuccessorDecrement{&_graph}, 
                   Galois::loopname("SuccessorDecrement"), 
                   Galois::numrun(_graph.get_run_identifier()));

    // need reduce set for both flag and succ
    _graph.sync<writeSource, readDestination, Reduce_set_propogation_flag, 
                Broadcast_propogation_flag,
                Bitset_propogation_flag>("SuccessorDecrement");
    _graph.sync<writeSource, readDestination, Reduce_set_num_successors, 
                Broadcast_num_successors,
                Bitset_num_successors>("SuccessorDecrement");
  }

  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    if (src_data.num_successors == 0 && !src_data.propogation_flag) {
      assert(src_data.trim == 0);
      src_data.propogation_flag = true;
      bitset_propogation_flag.set(src);
    } else if (src_data.trim > 0) {
      // decrement successor by trim then reset
      assert(src_data.trim <= src_data.num_successors);

      src_data.num_successors = src_data.num_successors - src_data.trim;
      bitset_num_successors.set(src);

      src_data.trim = 0;

      // set prop flag to false so it can propogate the value (actually
      // it should already be false at this point from reset)
      if (src_data.num_successors == 0) {
        src_data.propogation_flag = false;
        bitset_propogation_flag.set(src);
      }
    }
  }
};

/* Use to_add to increment the dependency value */
struct DependencyIncrement {
  Graph* graph;
  DependencyIncrement(Graph* _graph) : graph(_graph) {}

  void static go(Graph& _graph) {
    #ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) {
        std::string impl_str("CUDA_DO_ALL_IMPL_DependencyIncrement_" + 
                             (_graph.get_run_identifier()));
        Galois::StatTimer StatTimer_cuda(impl_str.c_str());
        StatTimer_cuda.start();
        DependencyIncrement_all_cuda(cuda_ctx);
        StatTimer_cuda.stop();
      } else if (personality == CPU)
    #endif
    Galois::do_all(_graph.begin(), _graph.end(), DependencyIncrement{&_graph}, 
                   Galois::loopname("DependencyIncrement"), 
                   Galois::numrun(_graph.get_run_identifier()));
  }

  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    // increment dependency using to_add_float then reset
    if (src_data.to_add_float > 0.0) {
      src_data.dependency += src_data.to_add_float;
      src_data.to_add_float = 0.0;

      // used in DependencyPropogation's go method
      bitset_dependency.set(src);
    }
  }
};

/* Do dependency propogation which is required for betweeness centraility
 * calculation */
struct DependencyPropogation {
  const unsigned int &local_current_src_node;
  Graph* graph;
  static Galois::DGAccumulator<unsigned int> DGAccumulator_accum;

  DependencyPropogation(const unsigned int &_local_current_src_node,
                        Graph* _graph) : 
                          local_current_src_node(_local_current_src_node),
                          graph(_graph){}

  /* Look at all nodes to do propogation until no more work is done */
  void static go(Graph& _graph) {
    unsigned int iterations = 0;
    unsigned int accum_result;

    do {
      _graph.set_num_iter(iterations);
      DGAccumulator_accum.reset();

    #ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) {
        std::string impl_str("CUDA_DO_ALL_IMPL_DependencyPropogation_" + 
                             (_graph.get_run_identifier()));
        Galois::StatTimer StatTimer_cuda(impl_str.c_str());
        StatTimer_cuda.start();
        int __retval = 0;
        DependencyPropogation_all_cuda(__retval, current_src_node, cuda_ctx);
        DGAccumulator_accum += __retval;
        StatTimer_cuda.stop();
      } else if (personality == CPU)
    #endif
      Galois::do_all(_graph.begin(), _graph.end(), 
                     DependencyPropogation(current_src_node, &_graph), 
                     Galois::loopname("DependencyPropogation"));

    //if (personality == GPU_CUDA) {
    //  char test[100];
    //  for (auto ii = (_graph).begin(); ii != (_graph).ghost_end(); ++ii) {
    //    if ((_graph).isLocal((_graph).getGID(*ii))) {
    //      sprintf(test, "btrimdp%u %lu %u\n", iterations,
    //              (_graph).getGID(*ii),
    //              get_node_trim_cuda(cuda_ctx, *ii));
    //      Galois::Runtime::printOutput(test);
    //    }
    //  }
    //} else {
    //  char test[100];
    //  for (auto ii = (_graph).begin(); ii != (_graph).ghost_end(); ++ii) {
    //    if ((_graph).isLocal((_graph).getGID(*ii))) {
    //      sprintf(test, "btrimdp%u %lu %u\n", iterations,
    //              (_graph).getGID(*ii),
    //              _graph.getData(*ii).trim.load());
    //      Galois::Runtime::printOutput(test);
    //    }
    //  }
    //}

      _graph.sync<writeSource, readSource, Reduce_add_trim, 
                  Broadcast_trim, Bitset_trim>("DependencyPropogation");
      //_graph.sync<writeSource, readSource, Reduce_add_trim, 
      //            Broadcast_trim>("DependencyPropogation");

    //if (personality == GPU_CUDA) {
    //  char test[100];
    //  for (auto ii = (_graph).begin(); ii != (_graph).ghost_end(); ++ii) {
    //    if ((_graph).isLocal((_graph).getGID(*ii))) {
    //      sprintf(test, "aftertrimdp%u %lu %u\n", iterations,
    //              (_graph).getGID(*ii),
    //              get_node_trim_cuda(cuda_ctx, *ii));
    //      Galois::Runtime::printOutput(test);
    //    }
    //  }
    //} else {
    //  char test[100];
    //  for (auto ii = (_graph).begin(); ii != (_graph).ghost_end(); ++ii) {
    //    if ((_graph).isLocal((_graph).getGID(*ii))) {
    //      sprintf(test, "aftertrimdp%u %lu %u\n", iterations,
    //              (_graph).getGID(*ii),
    //              _graph.getData(*ii).trim.load());
    //      Galois::Runtime::printOutput(test);
    //    }
    //  }
    //}

      // do successor decrementing using sync'd trim
      SuccessorDecrement::go(_graph);

      _graph.sync<writeSource, readSource, Reduce_add_to_add_float, 
                  Broadcast_to_add_float, 
                  Bitset_to_add_float>("DependencyPropogation");
      //_graph.sync<writeSource, readSource, Reduce_add_to_add_float, 
      //            Broadcast_to_add_float>("DependencyPropogation");

      // with to_add_float sync'd, update dependency
      DependencyIncrement::go(_graph);

      iterations++;
      accum_result = DGAccumulator_accum.reduce();

      if (accum_result) {
        // sync dependency on dest; source should all have same dep
        _graph.sync<writeSource, readDestination, Reduce_set_dependency,
                    Broadcast_dependency, Bitset_dependency>("DependencyPropogation");
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
    if (src_data.num_successors > 0 && graph->getGID(src) != local_current_src_node) {
      for (auto current_edge = graph->edge_begin(src), 
                end_edge = graph->edge_end(src); 
           current_edge != end_edge; 
           ++current_edge) {

        GNode dst = graph->getEdgeDst(current_edge);
        auto& dst_data = graph->getData(dst);

        // SSSP
        //unsigned int edge_weight = graph->getEdgeData(current_edge);
        // BFS
        unsigned int edge_weight = 1;

        // only operate if a dst has no more successors (i.e. delta finalized
        // for this round) + if it hasn't propogated the value yet (i.e. because
        // successors become 0 in the last do_all round)
        if (dst_data.num_successors == 0 && !dst_data.propogation_flag) {
          // dest on shortest path with this node as predecessor
          if ((src_data.current_length + edge_weight) == 
               dst_data.current_length) {
            // increment my trim for later use to decrement successor
            Galois::atomicAdd(src_data.trim, (unsigned int)1);

            // update my to_add_float (which is later used to update dependency)
            Galois::add(src_data.to_add_float, 
                        (((float)src_data.num_shortest_paths / 
                          (float)dst_data.num_shortest_paths) * 
                          (float)(1.0 + dst_data.dependency)));

            bitset_trim.set(src);
            bitset_to_add_float.set(src);

            DGAccumulator_accum += 1;
          }
        }
      }
    }
  }
};
Galois::DGAccumulator<unsigned int> DependencyPropogation::DGAccumulator_accum;

struct BC {
  Graph* graph;

  BC(Graph* _graph) : graph(_graph){}

  void static go(Graph& _graph){
    uint64_t start_i;
    uint64_t end_i;

    if (singleSourceBC) {
      start_i = singleSourceNode;
      end_i = singleSourceNode + 1;
    } else {
      start_i = 0;
      end_i = _graph.totalNodes;
    }

    for (uint64_t i = start_i; i < end_i; i++) {
      current_src_node = i;

      std::cout << "SSSP source node " << i << "\n";

      // reset the graph aside from the between-cent measure
      InitializeIteration::go(_graph);
      //std::cout << "init done\n";

      // get SSSP on the current graph
      SSSP::go(_graph);
      //std::cout << "SSSP done\n";

      // calculate the succ/pred for all nodes in the SSSP DAG
      PredAndSucc::go(_graph);
      //std::cout << "PredSucc done\n";

      // calculate the number of shortest paths for each node
      NumShortestPaths::go(_graph);
      //std::cout << "NumShortPaths done\n";

      // RESET PROP FLAG
      PropFlagReset::go(_graph);
      //std::cout << "PropFlagReset done\n";

      // do between-cent calculations for this iteration 
      DependencyPropogation::go(_graph);
      //std::cout << "DepProp done\n";

      // finally, since dependencies are finalized for this round at this 
      // point, add them to the betweeness centrality measure on each node
    #ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) {
        std::string impl_str("CUDA_DO_ALL_IMPL_BC_" + 
                             (_graph.get_run_identifier()));
        Galois::StatTimer StatTimer_cuda(impl_str.c_str());
        StatTimer_cuda.start();
        BC_all_cuda(cuda_ctx);
        StatTimer_cuda.stop();
      } else if (personality == CPU)
    #endif
      Galois::do_all(_graph.begin(), _graph.end(), BC(&_graph), 
                     Galois::loopname("BC"));
      
      // all sources should have dependency value, meaning all sources will
      // update the BC value correctly; no sync required here 
    }
  }

  /* adds dependency measure to BC measure (dependencies should be finalized,
   * i.e. no unprocessed successors on the node) */
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    Galois::add(src_data.betweeness_centrality, src_data.dependency);
  }
};
 
/******************************************************************************/
/* Main method for running */
/******************************************************************************/

int main(int argc, char** argv) {
  try {
    LonestarStart(argc, argv, name, desc, url);
    Galois::StatManager statManager;

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

    if (enableVCut) {
      if (vertexcut == CART_VCUT)
        h_graph = new Graph_cartesianCut(inputFile, partFolder, net.ID, net.Num,
                                         scalefactor, transpose);
      else if (vertexcut == PL_VCUT)
        h_graph = new Graph_vertexCut(inputFile, partFolder, net.ID, net.Num, 
                                      scalefactor, transpose, VCutThreshold);
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

    bitset_to_add.resize(h_graph->get_local_total_nodes());
    bitset_to_add_float.resize(h_graph->get_local_total_nodes());
    bitset_num_shortest_paths.resize(h_graph->get_local_total_nodes());
    bitset_num_successors.resize(h_graph->get_local_total_nodes());
    bitset_num_predecessors.resize(h_graph->get_local_total_nodes());
    bitset_trim.resize(h_graph->get_local_total_nodes());
    bitset_current_length.resize(h_graph->get_local_total_nodes());
    bitset_propogation_flag.resize(h_graph->get_local_total_nodes());
    bitset_dependency.resize(h_graph->get_local_total_nodes());

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
      }
    }

    StatTimer_total.stop();

    // Verify, i.e. print out graph data for examination
    if (verify) {
      char *v_out = (char*)malloc(35);
#ifdef __GALOIS_HET_CUDA__
      if (personality == CPU) { 
#endif
        for (auto ii = (*h_graph).begin(); ii != (*h_graph).end(); ++ii) {
          if ((*h_graph).isOwned((*h_graph).getGID(*ii))) {
            // outputs betweenness centrality
            sprintf(v_out, "%lu %.9f\n", (*h_graph).getGID(*ii),
                    (*h_graph).getData(*ii).betweeness_centrality);
            Galois::Runtime::printOutput(v_out);
          }
        }
#ifdef __GALOIS_HET_CUDA__
      } else if (personality == GPU_CUDA) {
        for (auto ii = (*h_graph).begin(); ii != (*h_graph).end(); ++ii) {
          if ((*h_graph).isOwned((*h_graph).getGID(*ii))) 
            sprintf(v_out, "%lu %.9f\n", (*h_graph).getGID(*ii),
                    get_node_betweeness_centrality_cuda(cuda_ctx, *ii));

            Galois::Runtime::printOutput(v_out);
            memset(v_out, '\0', 35);
        }
      }
#endif
      free(v_out);
    }
    return 0;
  } catch(const char* c) {
    std::cout << "Error: " << c << "\n";
    return 1;
  }
}
