/** ConnectedComp -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
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
 * Compute ConnectedComp Pull on distributed Galois.
 *
 * @author Gurbinder Gill <gurbinder533@gmail.com>
 * @author Roshan Dathathri <roshan@cs.utexas.edu>
 * @author Loc Hoang <l_hoang@utexas.edu> (sanity check operators)
 */

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
   assert(false&& "Invalid personality");
   return "";
}
#endif

static const char* const name = "ConnectedComp Pull - Distributed Heterogeneous";
static const char* const desc = "ConnectedComp pull on Distributed Galois.";
static const char* const url = 0;

/******************************************************************************/
/* Declaration of command line arguments */
/******************************************************************************/

namespace cll = llvm::cl;
static cll::opt<unsigned int> maxIterations("maxIterations", 
                                            cll::desc("Maximum iterations: "
                                                      "Default 1000"), 
                                            cll::init(1000));

static cll::opt<unsigned long long> src_node("srcNodeId", 
                                             cll::desc("ID of the source node"), 
                                             cll::init(0));
static cll::opt<bool> verify("verify", 
                             cll::desc("Verify results by outputting results "
                                       "to file"), 
                             cll::init(false));

#ifdef __GALOIS_HET_CUDA__
static cll::opt<int> gpudevice("gpu", 
                                cll::desc("Select GPU to run on, "
                                          "default is to choose automatically"), 
                                cll::init(-1));
static cll::opt<Personality> personality("personality", cll::desc("Personality"),
      cll::values(clEnumValN(CPU, "cpu", "Galois CPU"), 
                  clEnumValN(GPU_CUDA, "gpu/cuda", "GPU/CUDA"), 
                  clEnumValN(GPU_OPENCL, "gpu/opencl", "GPU/OpenCL"), 
                  clEnumValEnd),
      cll::init(CPU));
static cll::opt<unsigned> scalegpu("scalegpu", 
      cll::desc("Scale GPU workload w.r.t. CPU, default is proportionally "
                "equal workload to CPU and GPU (1)"), 
      cll::init(1));
static cll::opt<unsigned> scalecpu("scalecpu", 
      cll::desc("Scale CPU workload w.r.t. GPU, default is proportionally "
                "equal workload to CPU and GPU (1)"), 
      cll::init(1));
static cll::opt<int> num_nodes("num_nodes", 
      cll::desc("Num of physical nodes with devices (default = num of hosts): " 
                "detect GPU to use for each host automatically"), 
      cll::init(-1));
static cll::opt<std::string> personality_set("pset", 
      cll::desc("String specifying personality for hosts on each physical "
                "node. 'c'=CPU,'g'=GPU/CUDA and 'o'=GPU/OpenCL"), 
      cll::init("c"));
#endif

/******************************************************************************/
/* Graph structure declarations + other initialization */
/******************************************************************************/

struct NodeData {
  unsigned long long comp_current;
};

#if __OPT_VERSION__ >= 3
galois::DynamicBitSet bitset_comp_current;
#endif

typedef hGraph<NodeData, void> Graph;
typedef typename Graph::GraphNode GNode;

#include "gen_sync.hh"

/******************************************************************************/
/* Algorithm structures */
/******************************************************************************/

struct InitializeGraph {
  Graph *graph;

  InitializeGraph(Graph* _graph) : graph(_graph){}

  void static go(Graph& _graph) {
    auto& allNodes = _graph.allNodesRange();
    #ifdef __GALOIS_HET_CUDA__
    	if (personality == GPU_CUDA) {
    		std::string impl_str("CUDA_DO_ALL_IMPL_InitializeGraph_" + 
                             (_graph.get_run_identifier()));
    		galois::StatTimer StatTimer_cuda(impl_str.c_str());
    		StatTimer_cuda.start();

        InitializeGraph_cuda(*(allNodes.begin()), *(allNodes.end()), 
                             cuda_ctx);

    		StatTimer_cuda.stop();
    	} else if (personality == CPU)
    #endif
    {
    galois::do_all_local(
      allNodes,
      InitializeGraph{&_graph}, 
      galois::loopname(_graph.get_run_identifier("InitializeGraph").c_str()),
      galois::steal<true>(),
      galois::timeit(),
      galois::no_stats()
    );
    }
  }

  void operator()(GNode src) const {
    NodeData& sdata = graph->getData(src);
    sdata.comp_current = graph->getGID(src);
  }
};

struct ConnectedComp {
  Graph* graph;
  galois::DGAccumulator<unsigned int>& DGAccumulator_accum;

  ConnectedComp(Graph* _graph, galois::DGAccumulator<unsigned int>& _dga) : 
    graph(_graph), DGAccumulator_accum(_dga) {}

  void static go(Graph& _graph, galois::DGAccumulator<unsigned int>& dga) {
    unsigned _num_iterations = 1;
    
    auto& nodesWithEdges = _graph.allNodesWithEdgesRange();
    do { 
      _graph.set_num_iter(_num_iterations);
      dga.reset();

      #if __OPT_VERSION__ == 5
      _graph.sync_on_demand<readDestination, Reduce_min_comp_current, 
                            Broadcast_comp_current, 
                            Bitset_comp_current>(Flags_comp_current, 
                                                 "ConnectedComp");
      #endif

    #ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) {
        std::string impl_str("CUDA_DO_ALL_IMPL_ConnectedComp_" + (_graph.get_run_identifier()));
        galois::StatTimer StatTimer_cuda(impl_str.c_str());
        StatTimer_cuda.start();
        int __retval = 0;
        ConnectedComp_cuda(*nodesWithEdges.begin(), *nodesWithEdges.end(),
                           __retval, cuda_ctx);
        dga += __retval;
        StatTimer_cuda.stop();
      } else if (personality == CPU)
    #endif
      galois::do_all_local(
        nodesWithEdges,
        ConnectedComp(&_graph, dga),
        galois::loopname(_graph.get_run_identifier("ConnectedComp").c_str()),
        galois::steal<true>(),
        galois::timeit(),
        galois::no_stats()
      );

      #if __OPT_VERSION__ == 5
      Flags_comp_current.set_write_src();
      #endif

      #if __OPT_VERSION__ == 1
      // naive sync of everything after operator 
      _graph.sync<writeAny, readAny, Reduce_min_comp_current,
                  Broadcast_comp_current>("ConnectedComp");
      #elif __OPT_VERSION__ == 2
      // sync of touched fields 
      _graph.sync<writeAny, readAny, Reduce_min_comp_current,
                  Broadcast_comp_current>("ConnectedComp");
      #elif __OPT_VERSION__ == 3
      // with bitset
      _graph.sync<writeAny, readAny, Reduce_min_comp_current,
                  Broadcast_comp_current, Bitset_comp_current>("ConnectedComp");
      #elif __OPT_VERSION__ == 4
      // write aware (not read aware, i.e. conservative)
      _graph.sync<writeSource, readAny, Reduce_min_comp_current,
                  Broadcast_comp_current, Bitset_comp_current>("ConnectedComp");
      #endif

      //_graph.sync<writeSource, readDestination, Reduce_min_comp_current, 
      //            Broadcast_comp_current, Bitset_comp_current>("ConnectedComp");
      
      galois::runtime::reportStat("(NULL)", 
        "NUM_WORK_ITEMS_" + (_graph.get_run_identifier()), 
        (unsigned long)dga.read_local(), 0);
      ++_num_iterations;
    } while((_num_iterations < maxIterations) && dga.reduce(_graph.get_run_identifier()));

    if (galois::runtime::getSystemNetworkInterface().ID == 0) {
      galois::runtime::reportStat("(NULL)", 
        "NUM_ITERATIONS_" + std::to_string(_graph.get_run_num()), 
        (unsigned long)_num_iterations, 0);
    }
  }

  void operator()(GNode src) const {
    NodeData& snode = graph->getData(src);

    for (auto jj = graph->edge_begin(src), ee = graph->edge_end(src); jj != ee; ++jj) {
      GNode dst = graph->getEdgeDst(jj);
      auto& dnode = graph->getData(dst);
      unsigned long long new_comp = dnode.comp_current;
      unsigned long long old_comp = galois::min(snode.comp_current, new_comp);
      if (old_comp > new_comp){
        #if __OPT_VERSION__ >= 3
        bitset_comp_current.set(src);
        #endif
        DGAccumulator_accum += 1;
      }
    }
  }
};

/******************************************************************************/
/* Sanity check operators */
/******************************************************************************/

/* Get/print the number of members in node 0's component */
struct SourceComponentSize {
  const unsigned long long src_comp;
  Graph* graph;

  galois::DGAccumulator<unsigned long long>& DGAccumulator_accum;

  SourceComponentSize(const unsigned long long _src_comp, Graph* _graph, 
                      galois::DGAccumulator<unsigned long long>& _dga) : 
    src_comp(_src_comp), graph(_graph), DGAccumulator_accum(_dga) {}

  void static go(Graph& _graph, galois::DGAccumulator<unsigned long long>& dga) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      // TODO currently no GPU support for sanity check operator
      fprintf(stderr, "Warning: No GPU support for sanity check; might get "
                      "wrong results.\n");
    }
  #endif

    dga.reset();

    if (_graph.isOwned(src_node)) {
      dga += _graph.getData(_graph.getLID(src_node)).comp_current;
    }

    uint64_t src_comp = dga.reduce();

    dga.reset();

    galois::do_all(_graph.allNodesRange().begin(), _graph.allNodesRange().end(), 
                   SourceComponentSize(src_comp, &_graph, dga), 
                   galois::loopname("SourceComponentSize"));

    uint64_t num_in_component = dga.reduce();

    // Only node 0 will print the number visited
    if (galois::runtime::getSystemNetworkInterface().ID == 0) {
      printf("Number of nodes in node %lu's component is %lu\n", (uint64_t)src_node, 
             num_in_component);
    }
  }

  /* Check if an owned node's component is the same as src's.
   * if yes, then increment an accumulator */
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    if (graph->isOwned(graph->getGID(src)) && 
        src_data.comp_current == src_comp) {
      DGAccumulator_accum += 1;
    }
  }
};

/******************************************************************************/
/* Main */
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
    galois::StatTimer StatTimer_init("TIMER_GRAPH_INIT"),
                      StatTimer_total("TIMER_TOTAL"),
                      StatTimer_hg_init("TIMER_HG_INIT");

    StatTimer_total.start();

    std::vector<unsigned> scalefactor;
#ifdef __GALOIS_HET_CUDA__
    const unsigned my_host_id = galois::runtime::getHostID();
    int gpu_device = gpudevice;
    //Parse arg string when running on multiple hosts and update/override personality
    //with corresponding value.
    if (num_nodes == -1) num_nodes = net.Num;
    assert((net.Num % num_nodes) == 0);
    if (personality_set.length() == (net.Num / num_nodes)) {
      switch (personality_set.c_str()[my_host_id % (net.Num / num_nodes)]) {
      case 'g':
        personality = GPU_CUDA;
        break;
      case 'o':
        assert(0);
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
        for (unsigned i=0; i<net.Num; ++i) {
          if (personality_set.c_str()[i % num_nodes] == 'c') 
            scalefactor.push_back(scalecpu);
          else
            scalefactor.push_back(scalegpu);
        }
      }
    }
#endif

    StatTimer_hg_init.start();
    Graph* hg = nullptr;

    // the symmetric flag needs to be explicitly set
    if (inputFileSymmetric) {
      hg = constructSymmetricGraph<NodeData, void>(scalefactor);
    } else {
      GALOIS_DIE("must pass symmetricGraph flag with symmetric graph to "
                 "connected-components");
    }

#ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      cuda_ctx = get_CUDA_context(my_host_id);
      if (!init_CUDA_context(cuda_ctx, gpu_device))
        return -1;
      MarshalGraph m = (*hg).getMarshalGraph(my_host_id);
      load_graph_CUDA(cuda_ctx, m, net.Num);
    } else if (personality == GPU_OPENCL) {
      //galois::opencl::cl_env.init(cldevice.Value);
    }
#endif
    #if __OPT_VERSION__ >= 3
    bitset_comp_current.resize(hg->size());
    #endif
    StatTimer_hg_init.stop();

    std::cout << "[" << net.ID << "] InitializeGraph::go called\n";
    StatTimer_init.start();
      InitializeGraph::go((*hg));
    StatTimer_init.stop();
    galois::runtime::getHostBarrier().wait();

    galois::DGAccumulator<unsigned int> DGAccumulator_accum;
    galois::DGAccumulator<unsigned long long> DGAccumulator_accum64;

    for(auto run = 0; run < numRuns; ++run){
      std::cout << "[" << net.ID << "] ConnectedComp::go run " << run << " called\n";
      std::string timer_str("TIMER_" + std::to_string(run));
      galois::StatTimer StatTimer_main(timer_str.c_str());

      StatTimer_main.start();
        ConnectedComp::go(*hg, DGAccumulator_accum);
      StatTimer_main.stop();

      SourceComponentSize::go(*hg, DGAccumulator_accum64);

      if((run + 1) != numRuns){
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) { 
          #if __OPT_VERSION__ >= 3
          bitset_comp_current_reset_cuda(cuda_ctx);
          #endif
        } else
      #endif
        #if __OPT_VERSION__ >= 3
        bitset_comp_current.reset();
        #endif

        #if __OPT_VERSION__ == 5
        Flags_comp_current.clear_all();
        #endif


        (*hg).set_num_run(run+1);
        InitializeGraph::go((*hg));
        galois::runtime::getHostBarrier().wait();
      }
    }

   StatTimer_total.stop();


    // Verify
    if (verify) {
#ifdef __GALOIS_HET_CUDA__
      if (personality == CPU) { 
#endif
        for(auto ii = (*hg).begin(); ii != (*hg).end(); ++ii) {
          if ((*hg).isOwned((*hg).getGID(*ii))) 
            galois::runtime::printOutput("% %\n", (*hg).getGID(*ii), 
              (*hg).getData(*ii).comp_current);
        }
#ifdef __GALOIS_HET_CUDA__
      } else if(personality == GPU_CUDA)  {
        for(auto ii = (*hg).begin(); ii != (*hg).end(); ++ii) {
          if ((*hg).isOwned((*hg).getGID(*ii))) 
            galois::runtime::printOutput("% %\n", (*hg).getGID(*ii), 
              get_node_comp_current_cuda(cuda_ctx, *ii));
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
