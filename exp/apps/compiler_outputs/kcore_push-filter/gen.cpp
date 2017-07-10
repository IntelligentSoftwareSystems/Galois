/**KCore -*- C++ -*-
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
 * Compute KCore on distributed Galois using top-filter.
 *
 * @author Loc Hoang <l_hoang@utexas.edu>
 */

/******************************************************************************/
/* This was manually genereted, not compiler generated */
/******************************************************************************/

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

static const char* const name = "KCore - Distributed Heterogeneous "
                                "with Worklist.";
static const char* const desc = "KCore on Distributed Galois.";
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
static cll::opt<bool> transpose("transpose", 
                                cll::desc("transpose the graph in memory after "
                                          "partitioning"),
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

// required k specification for k-core
static cll::opt<unsigned int> k_core_num("kcore",
                                     cll::desc("KCore value"),
                                     cll::Required);

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


/******************************************************************************/
/* Graph structure declarations + other inits */
/******************************************************************************/

struct NodeData {
  std::atomic<unsigned int> current_degree;
  std::atomic<unsigned int> trim;
  bool flag;
};

typedef hGraph<NodeData, void> Graph;
typedef hGraph_edgeCut<NodeData, void> Graph_edgeCut;
typedef hGraph_vertexCut<NodeData, void> Graph_vertexCut;
typedef hGraph_cartesianCut<NodeData, void> Graph_cartesianCut;

typedef typename Graph::GraphNode GNode;

// bitset for tracking updates
Galois::DynamicBitSet bitset_current_degree;
Galois::DynamicBitSet bitset_trim;

// add all sync/bitset structs (needs above declarations)
#include "gen_sync.hh"

/******************************************************************************/
/* Functors for running the algorithm */
/******************************************************************************/

/* Degree counting
 * Called by InitializeGraph1 */
struct InitializeGraph2 {
  Graph *graph;

  InitializeGraph2(Graph* _graph) : graph(_graph){}

  /* Initialize the entire graph node-by-node */
  void static go(Graph& _graph) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      std::string impl_str("CUDA_DO_ALL_IMPL_InitializeGraph2_" + 
                           (_graph.get_run_identifier()));
      Galois::StatTimer StatTimer_cuda(impl_str.c_str());
      StatTimer_cuda.start();
      InitializeGraph2_all_cuda(cuda_ctx);
      StatTimer_cuda.stop();
    } else if (personality == CPU)
  #endif
    Galois::do_all(_graph.begin(), _graph.end(), InitializeGraph2{&_graph}, 
                   Galois::loopname("InitializeGraph2"), 
                   Galois::numrun(_graph.get_run_identifier()));

    _graph.sync<writeDestination, readSource, Reduce_add_current_degree, 
      Broadcast_current_degree, Bitset_current_degree>("InitializeGraph2");
  }

  /* Calculate degree of nodes by checking how many nodes have it as a dest and
   * adding for every dest */
  void operator()(GNode src) const {
    for (auto current_edge = graph->edge_begin(src), 
              end_edge = graph->edge_end(src);
         current_edge != end_edge;
         current_edge++) {
      GNode dest_node = graph->getEdgeDst(current_edge);

      //if (graph->getGID(dest_node) == 350208) {
      //  std::cout << "[" << graph->id << "]" <<
      //            "src is " << graph->getGID(src) << "\n";
      //}

      NodeData& dest_data = graph->getData(dest_node);
      Galois::atomicAdd(dest_data.current_degree, (unsigned int)1);

      bitset_current_degree.set(dest_node);
    }
  }
};


/* Initialize: initial field setup */
struct InitializeGraph1 {
  Graph *graph;

  InitializeGraph1(Graph* _graph) : graph(_graph){}

  /* Initialize the entire graph node-by-node */
  void static go(Graph& _graph) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      std::string impl_str("CUDA_DO_ALL_IMPL_InitializeGraph1_" + 
                           (_graph.get_run_identifier()));
      Galois::StatTimer StatTimer_cuda(impl_str.c_str());
      StatTimer_cuda.start();
      InitializeGraph1_all_cuda(cuda_ctx);
      StatTimer_cuda.stop();
    } else if (personality == CPU)
  #endif
    Galois::do_all(_graph.begin(), _graph.end(), InitializeGraph1{&_graph}, 
                   Galois::loopname("InitializeGraph1"), 
                   Galois::numrun(_graph.get_run_identifier()));

    _graph.sync<writeSource, readDestination, Reduce_set_current_degree, 
      Broadcast_current_degree, Bitset_current_degree>("InitializeGraph1");

    // make sure everything is initialized
    InitializeGraph2::go(_graph);
  }

  /* Setup intial fields */
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);
    src_data.flag = true;
    src_data.trim = 0;
    src_data.current_degree = 0;

    bitset_current_degree.set(src);
  }
};


/* Use the trim value (i.e. number of incident nodes that have been removed)
 * to update degrees.
 * Called by KCoreStep1 */
struct KCoreStep2 {
  Graph* graph;

  KCoreStep2(Graph* _graph) : graph(_graph){}

  void static go(Graph& _graph){
    //if (personality == GPU_CUDA) {
    //  char test[100];
    //  for (auto ii = (_graph).begin(); ii != (_graph).ghost_end(); ++ii) {
    //    if ((_graph).isOwned((_graph).getGID(*ii))) {
    //      sprintf(test, "%utrim %lu %u\n", _graph.id, (_graph).getGID(*ii),
    //              get_node_trim_cuda(cuda_ctx, *ii));
    //      Galois::Runtime::printOutput(test);
    //    }
    //  }
    //} else {
    //  char test[100];
    //  for (auto ii = (_graph).begin(); ii != (_graph).ghost_end(); ++ii) {
    //    if ((_graph).isOwned((_graph).getGID(*ii))) {
    //      sprintf(test, "%utrim %lu %u\n", _graph.id, (_graph).getGID(*ii),
    //              _graph.getData(*ii).trim.load());
    //      Galois::Runtime::printOutput(test);
    //    }
    //  }
    //}

  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      std::string impl_str("CUDA_DO_ALL_IMPL_KCoreStep2_" + 
                           (_graph.get_run_identifier()));
      Galois::StatTimer StatTimer_cuda(impl_str.c_str());

      //for (auto ii = (_graph).begin(); ii != (_graph).end(); ++ii) {
      //  if ((_graph).isOwned((_graph).getGID(*ii))) 
      //    Galois::Runtime::printOutput("% % t: % %\n", (_graph).getGID(*ii), 
      //                             get_node_flag_cuda(cuda_ctx, *ii),
      //                             get_node_trim_cuda(cuda_ctx, *ii),
      //                             (get_node_current_degree_cuda(cuda_ctx, *ii)));
      //}

      StatTimer_cuda.start();
      KCoreStep2_all_cuda(cuda_ctx);
      StatTimer_cuda.stop();
    } else if (personality == CPU)
  #endif
    Galois::do_all(_graph.begin(), _graph.end(), KCoreStep2(&_graph), 
                   Galois::loopname("KCoreStep2"));
  }

  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);
    // note even dead nodes can have trim updated
    if (src_data.trim > 0) {
      src_data.current_degree = src_data.current_degree - src_data.trim;
      src_data.trim = 0;
    }
  }
};


/* Step that determines if a node is dead and updates its neighbors' trim
 * if it is */
struct KCoreStep1 {
  cll::opt<unsigned int>& local_k_core_num;
  static Galois::DGAccumulator<int> DGAccumulator_accum;
  Graph* graph;

  KCoreStep1(cll::opt<unsigned int>& _kcore, Graph* _graph) : 
    local_k_core_num(_kcore), graph(_graph){}

  void static go(Graph& _graph){
    unsigned iterations = 0;
    
    do {
      _graph.set_num_iter(iterations);
      DGAccumulator_accum.reset();

    #ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) {
        std::string impl_str("CUDA_DO_ALL_IMPL_KCoreStep1_" + 
                             (_graph.get_run_identifier()));
        Galois::StatTimer StatTimer_cuda(impl_str.c_str());
        StatTimer_cuda.start();
        int __retval = 0;
        KCoreStep1_all_cuda(__retval, k_core_num, cuda_ctx);
        DGAccumulator_accum += __retval;
        StatTimer_cuda.stop();
      } else if (personality == CPU)
    #endif
      Galois::do_all(_graph.begin(), _graph.end(), 
                     KCoreStep1(k_core_num, &_graph), 
                     Galois::loopname("KCoreStep1"));

    //if (personality == GPU_CUDA) {
    //  char test[100];
    //  for (auto ii = (_graph).begin(); ii != (_graph).ghost_end(); ++ii) {
    //    if ((_graph).isLocal((_graph).getGID(*ii))) {
    //      sprintf(test, "%ubeforesynctrim %lu %u\n", _graph.id, (_graph).getGID(*ii),
    //              get_node_trim_cuda(cuda_ctx, *ii));
    //      Galois::Runtime::printOutput(test);
    //    }
    //  }
    //} else {
    //  char test[100];
    //  for (auto ii = (_graph).begin(); ii != (_graph).ghost_end(); ++ii) {
    //    if ((_graph).isLocal((_graph).getGID(*ii))) {
    //      sprintf(test, "%ubeforesynctrim %lu %u\n", _graph.id, (_graph).getGID(*ii),
    //              _graph.getData(*ii).trim.load());
    //      Galois::Runtime::printOutput(test);
    //    }
    //  }
    //}

      
      // do the trim sync
      _graph.sync<writeDestination, readSource, Reduce_add_trim, Broadcast_trim, 
                  Bitset_trim>("KCoreStep1");
      //_graph.sync<writeDestination, readSource, Reduce_add_trim, 
      //            Broadcast_trim>("KCoreStep1");

      // handle trimming (locally)
      KCoreStep2::go(_graph);

      iterations++;
    } while ((iterations < maxIterations) && DGAccumulator_accum.reduce());
  }

  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    // only if node is alive we do things
    if (src_data.flag) {
      if (src_data.current_degree < local_k_core_num) {
        // set flag to 0 (false) and increment trim on outgoing neighbors
        src_data.flag = false;
        DGAccumulator_accum += 1;

        for (auto current_edge = graph->edge_begin(src), 
                  end_edge = graph->edge_end(src);
             current_edge != end_edge; 
             ++current_edge) {
           GNode dst = graph->getEdgeDst(current_edge);

           auto& dst_data = graph->getData(dst);

           Galois::atomicAdd(dst_data.trim, (unsigned int)1);
           bitset_trim.set(dst);
        }
      }
    }
  }
};
Galois::DGAccumulator<int> KCoreStep1::DGAccumulator_accum;

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

    bitset_current_degree.resize(h_graph->get_local_total_nodes());
    bitset_trim.resize(h_graph->get_local_total_nodes());

    StatTimer_hg_init.stop();

    std::cout << "[" << net.ID << "] InitializeGraph::go functions called\n";
    StatTimer_graph_init.start();
      InitializeGraph1::go((*h_graph));
    StatTimer_graph_init.stop();

    for (auto run = 0; run < numRuns; ++run) {
      std::cout << "[" << net.ID << "] KCoreStep1::go run " << run << " called\n";
      std::string timer_str("TIMER_" + std::to_string(run));
      Galois::StatTimer StatTimer_main(timer_str.c_str());

      StatTimer_main.start();
        KCoreStep1::go((*h_graph));
      StatTimer_main.stop();

      // re-init graph for next run
      if ((run + 1) != numRuns) {
        Galois::Runtime::getHostBarrier().wait();
        (*h_graph).reset_num_iter(run+1);

      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) { 
          bitset_current_degree_reset_cuda(cuda_ctx);
          bitset_trim_reset_cuda(cuda_ctx);
        } else
      #endif
        { bitset_current_degree.reset();
        bitset_trim.reset(); }

        InitializeGraph1::go((*h_graph));
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
            // prints the flag (alive/dead) and current (out) degree of a node
            Galois::Runtime::printOutput("% % %\n", (*h_graph).getGID(*ii), 
                                         (*h_graph).getData(*ii).flag,
                                         (*h_graph).getData(*ii).current_degree);

          if (!((*h_graph).getData(*ii).flag)) {
            assert((*h_graph).getData(*ii).current_degree < k_core_num);
          } else {
            assert((*h_graph).getData(*ii).current_degree >= k_core_num);
          }
        }
    #ifdef __GALOIS_HET_CUDA__
      } else if (personality == GPU_CUDA) {
        for(auto ii = (*h_graph).begin(); ii != (*h_graph).end(); ++ii) {
          if ((*h_graph).isOwned((*h_graph).getGID(*ii))) 
            Galois::Runtime::printOutput("% % %\n", (*h_graph).getGID(*ii), 
                                     get_node_flag_cuda(cuda_ctx, *ii),
                                     (get_node_current_degree_cuda(cuda_ctx, *ii)));
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
