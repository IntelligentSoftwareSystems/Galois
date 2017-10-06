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
#include "galois/runtime/CompilerHelperFunctions.h"

#include "galois/runtime/dGraph_edgeCut.h"
#include "galois/runtime/dGraph_cartesianCut.h"
#include "galois/runtime/dGraph_hybridCut.h"

#include "galois/DistAccumulator.h"
#include "galois/runtime/Tracer.h"

#include "galois/runtime/dGraphLoader.h"

#ifdef __GALOIS_HET_CUDA__
#include "galois/cuda/cuda_device.h"
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

static const char* const name = "KCore - Distributed Heterogeneous Pull Topological.";
static const char* const desc = "KCore on Distributed Galois.";
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
  uint32_t current_degree;
  uint32_t trim;
  uint8_t flag;
  uint8_t pull_flag;
};

typedef hGraph<NodeData, void> Graph;
typedef typename Graph::GraphNode GNode;

// bitset for tracking updates
galois::DynamicBitSet bitset_current_degree;
#if __OPT_VERSION__ >= 3
galois::DynamicBitSet bitset_trim;
#endif

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
    auto& nodesWithEdges = _graph.allNodesWithEdgesRange();

  #ifdef __GALOIS_HET_CUDA__
    // TODO calls all wrong
    if (personality == GPU_CUDA) {
      std::string impl_str("CUDA_DO_ALL_IMPL_InitializeGraph2_" + 
                           (_graph.get_run_identifier()));
      galois::StatTimer StatTimer_cuda(impl_str.c_str());
      StatTimer_cuda.start();
      InitializeGraph2_cuda(*nodesWithEdges.begin(), *nodesWithEdges.end(),
                            cuda_ctx);
      StatTimer_cuda.stop();
    } else if (personality == CPU)
  #endif
    galois::do_all_local(
      nodesWithEdges,
      DegreeCounting{ &_graph },
      galois::loopname(_graph.get_run_identifier("DegreeCounting").c_str()),
      galois::steal<true>(),
      galois::timeit(),
      galois::no_stats()
    );

    #if __OPT_VERSION__ == 5
    Flags_current_degree.set_write_src();
    // technically supposed to be elsewhere, but for timer's sake it's here
    _graph.sync_on_demand<readAny, Reduce_add_current_degree, 
                          Broadcast_current_degree, 
                          Bitset_current_degree>(Flags_current_degree,
                                                 "DegreeCounting");
    #endif

    #if __OPT_VERSION__ <= 4
    _graph.sync<writeSource, readAny, Reduce_add_current_degree, 
      Broadcast_current_degree, Bitset_current_degree>("DegreeCounting");
    #endif
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
    auto& allNodes = _graph.allNodesRange();

#ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      // TODO calls all wrong
      std::string impl_str("CUDA_DO_ALL_IMPL_InitializeGraph1_" + 
                           (_graph.get_run_identifier()));
      galois::StatTimer StatTimer_cuda(impl_str.c_str());
      StatTimer_cuda.start();
      InitializeGraph1_cuda(*(allNodes.begin()), *(allNodes.end()), 
                            cuda_ctx);
      StatTimer_cuda.stop();
    } else if (personality == CPU)
  #endif
     galois::do_all(
        allNodes.begin(), allNodes.end(),
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


/* Use the trim value (i.e. number of incident nodes that have been removed)
 * to update degrees.
 * Called by KCore */
struct DegreeUpdate {
  Graph* graph;

  DegreeUpdate(Graph* _graph) : graph(_graph){}

  void static go(Graph& _graph){
    auto& allNodes = _graph.allNodesRange();

    #if __OPT_VERSION__ == 5
    // current edgree here as well
    _graph.sync_on_demand<readAny, Reduce_add_trim, Broadcast_trim, 
                          Bitset_trim>(Flags_trim, "DegreeUpdate");
    #endif

  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      std::string impl_str("CUDA_DO_ALL_IMPL_KCoreStep2_" + 
                           (_graph.get_run_identifier()));
      galois::StatTimer StatTimer_cuda(impl_str.c_str());
      StatTimer_cuda.start();
      KCoreStep2_cuda(*allNodes.begin(), *allNodes.end(), cuda_ctx);
      StatTimer_cuda.stop();
    } else if (personality == CPU)
  #endif
     galois::do_all(
       allNodes.begin(), allNodes.end(),
       DegreeUpdate{ &_graph },
       galois::loopname(_graph.get_run_identifier("DegreeUpdate").c_str()),
       galois::timeit(),
       galois::no_stats()
     );
  }

  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    // we currently do not care about degree for dead nodes, 
    // so we ignore those (i.e. if flag isn't set, do nothing)
    if (src_data.flag) {
      if (src_data.trim > 0) {
        src_data.current_degree = src_data.current_degree - src_data.trim;
      }
    }

    src_data.trim = 0;
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
    auto& allNodes = _graph.allNodesRange();

    // current edgree here technically

    dga.reset();

    // TODO GPU code

    galois::do_all(
      allNodes.begin(), allNodes.end(),
      LiveUpdate{ k_core_num, &_graph, dga },
      galois::loopname(_graph.get_run_identifier("LiveUpdate").c_str()),
      galois::timeit(),
      galois::no_stats()
    );

    // TODO hand optimized can merge trim decrement into this operator.....

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
      // still alive
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
  }
};

/* Step that determines if a node is dead and updates its neighbors' trim
 * if it is */
struct KCore {
  Graph* graph;

  KCore(Graph* _graph) : graph(_graph) {}

  void static go(Graph& _graph, galois::DGAccumulator<unsigned int>& dga) {
    unsigned iterations = 0;
    
    auto& nodesWithEdges = _graph.allNodesWithEdgesRange();

    do {
      _graph.set_num_iter(iterations);

    #ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) {
        // TODO calls wrong
        std::string impl_str("CUDA_DO_ALL_IMPL_KCoreStep1_" + 
                             (_graph.get_run_identifier()));
        galois::StatTimer StatTimer_cuda(impl_str.c_str());
        StatTimer_cuda.start();
        int __retval = 0;
        // TODO kcore step 1 doesn't exist anymore
        //KCoreStep1_cuda(*nodesWithEdges.begin(), *nodesWithEdges.end(),
        //                __retval, k_core_num, cuda_ctx);
        dga += __retval;
        StatTimer_cuda.stop();
      } else if (personality == CPU)
    #endif

      galois::do_all_local(
        nodesWithEdges,
        KCore{ &_graph },
        galois::loopname(_graph.get_run_identifier("KCore").c_str()),
        galois::steal<true>(),
        galois::timeit(),
        galois::no_stats()
      );

      #if __OPT_VERSION__ == 5
      Flags_trim.set_write_src();
      #endif

      #if __OPT_VERSION__ == 1
      _graph.sync<writeAny, readAny, Reduce_add_trim, 
                  Broadcast_trim>("KCore");
      #elif __OPT_VERSION__ == 2
      _graph.sync<writeAny, readAny, Reduce_add_trim, 
                  Broadcast_trim>("KCore");
      #elif __OPT_VERSION__ == 3
      _graph.sync<writeAny, readAny, Reduce_add_trim, Broadcast_trim,
                  Bitset_trim>("KCore");
      #elif __OPT_VERSION__ == 4
      _graph.sync<writeSource, readAny, Reduce_add_trim, Broadcast_trim,
                  Bitset_trim>("KCore");
      #endif

      //_graph.sync<writeSource, readAny, Reduce_add_trim, Broadcast_trim, 
      //            Bitset_trim>("KCore");

      // handle trimming (locally on each node)
      DegreeUpdate::go(_graph);

      // update live/deadness
      LiveUpdate::go(_graph, dga);

      iterations++;
    } while ((iterations < maxIterations) && dga.reduce(_graph.get_run_identifier()));

    if (galois::runtime::getSystemNetworkInterface().ID == 0) {
      galois::runtime::reportStat("(NULL)", 
        "NUM_ITERATIONS_" + std::to_string(_graph.get_run_num()), 
        (unsigned long)iterations, 0);
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
 
           #if __OPT_VERSION__ >= 3
           bitset_trim.set(src);
           #endif
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

    galois::do_all(_graph.allNodesRange().begin(), _graph.allNodesRange().end(), 
                   GetAliveDead(&_graph, dga1, dga2), 
                   galois::loopname("GetAliveDead"),
                   galois::numrun(_graph.get_run_identifier()));

    uint32_t num_alive = dga1.reduce();
    uint32_t num_dead = dga2.reduce();

    // Only node 0 will print data
    if (galois::runtime::getSystemNetworkInterface().ID == 0) {
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

int main(int argc, char** argv) {
  try {
    galois::DistMemSys G(getStatsFile());
    DistBenchStart(argc, argv, name, desc, url);

    {
    auto& net = galois::runtime::getSystemNetworkInterface();
    if (net.ID == 0) {
      galois::runtime::reportStat("(NULL)", "Max Iterations", 
                                  (unsigned long)maxIterations, 0);
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
  #ifdef __GALOIS_HET_CUDA__
    const unsigned my_host_id = galois::runtime::getHostID();
    int gpu_device = gpudevice;

    if (num_nodes == -1) num_nodes = net.Num;
    assert((net.Num % num_nodes) == 0);

    // Parse arg string when running on multiple hosts and update/override 
    // personality with corresponding value.
    if (personality_set.length() == galois::runtime::NetworkInterface::Num) {
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

    if (inputFileSymmetric) {
      h_graph = constructSymmetricGraph<NodeData, void>(scalefactor);
    } else {
      GALOIS_DIE("must pass symmetricGraph flag with symmetric graph to "
                 "kcore");
    }

  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      cuda_ctx = get_CUDA_context(my_host_id);
      if (!init_CUDA_context(cuda_ctx, gpu_device))
        return -1;
      MarshalGraph m = (*h_graph).getMarshalGraph(my_host_id);
      load_graph_CUDA(cuda_ctx, m, net.Num);
    } else if (personality == GPU_OPENCL) {
      //galois::opencl::cl_env.init(cldevice.Value);
    }
  #endif

    bitset_current_degree.resize(h_graph->size());
    #if __OPT_VERSION__ >= 3
    bitset_trim.resize(h_graph->size());
    #endif

    StatTimer_hg_init.stop();

    std::cout << "[" << net.ID << "] InitializeGraph::go functions called\n";
    StatTimer_graph_init.start();
      InitializeGraph::go((*h_graph));
    StatTimer_graph_init.stop();
    galois::runtime::getHostBarrier().wait();

    galois::DGAccumulator<unsigned int> DGAccumulator_accum;
    galois::DGAccumulator<uint64_t> dga1;
    galois::DGAccumulator<uint64_t> dga2;

    for (auto run = 0; run < numRuns; ++run) {
      std::cout << "[" << net.ID << "] KCore::go run " << run << " called\n";
      std::string timer_str("TIMER_" + std::to_string(run));
      galois::StatTimer StatTimer_main(timer_str.c_str());

      StatTimer_main.start();
        KCore::go(*h_graph, DGAccumulator_accum);
      StatTimer_main.stop();

      // sanity check
      GetAliveDead::go(*h_graph, dga1, dga2);

      // re-init graph for next run
      if ((run + 1) != numRuns) {
        (*h_graph).set_num_run(run+1);

        #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) { 
          bitset_current_degree_reset_cuda(cuda_ctx);
          #if __OPT_VERSION__ >= 3
          bitset_trim_reset_cuda(cuda_ctx);
          #endif
        } else
        #endif
        { bitset_current_degree.reset();
        #if __OPT_VERSION__ >= 3
        bitset_trim.reset(); 
        #endif
        }

        #if __OPT_VERSION__ == 5
        Flags_current_degree.clear_all();
        Flags_trim.clear_all();
        #endif

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
          if ((*h_graph).isOwned((*h_graph).getGID(*ii))) 
            // prints the flag (alive/dead)
            galois::runtime::printOutput("% %\n", (*h_graph).getGID(*ii), 
                                         (bool)(*h_graph).getData(*ii).flag);


          // does a sanity check as well: 
          // degree higher than kcore if node is alive
          if (!((*h_graph).getData(*ii).flag)) {
            assert((*h_graph).getData(*ii).current_degree < k_core_num);
          } 
        }
    #ifdef __GALOIS_HET_CUDA__
      } else if (personality == GPU_CUDA) {
        for (auto ii = (*h_graph).begin(); ii != (*h_graph).end(); ++ii) {
          if ((*h_graph).isOwned((*h_graph).getGID(*ii))) 
            galois::runtime::printOutput("% %\n", (*h_graph).getGID(*ii), 
                                       (bool)get_node_flag_cuda(cuda_ctx, *ii));
                                     
        }
      }
    #endif
    }
    }
    galois::runtime::getHostBarrier().wait();
    G.printDistStats();
    galois::runtime::getHostBarrier().wait();


    return 0;
  } catch(const char* c) {
    std::cerr << "Error: " << c << "\n";
    return 1;
  }
}
