/** BFS -*- C++ -*-
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
 * Compute BFS on distributed Galois using worklist.
 *
 * @author Gurbinder Gill <gurbinder533@gmail.com>
 * @author Roshan Dathathri <roshan@cs.utexas.edu>
 * @author Loc Hoang <l_hoang@utexas.edu> (sanity check operators)
 */

#include <iostream>
#include <limits>
#include "galois/Galois.h"
#include "galois/gstl.h"
#include "Lonestar/BoilerPlate.h"
#include "galois/Runtime/CompilerHelperFunctions.h"

#include "galois/Runtime/dGraph_edgeCut.h"
#include "galois/Runtime/dGraph_cartesianCut.h"
#include "galois/Runtime/dGraph_hybridCut.h"

#include "galois/DistAccumulator.h"
#include "galois/Runtime/Tracer.h"

#include "galois/Runtime/dGraphLoader.h"

#ifdef __GALOIS_HET_CUDA__
#include "galois/Runtime/Cuda/cuda_device.h"
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

static const char* const name = "BFS - Distributed Heterogeneous with worklist.";
static const char* const desc = "BFS on Distributed Galois.";
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

const uint32_t infinity = std::numeric_limits<uint32_t>::max()/4;

struct NodeData {
  std::atomic<uint32_t> dist_current;
  uint32_t dist_old;
};

galois::DynamicBitSet bitset_dist_current;

typedef hGraph<NodeData, void> Graph;
typedef typename Graph::GraphNode GNode;

#include "gen_sync.hh"

/******************************************************************************/
/* Algorithm structures */
/******************************************************************************/

struct InitializeGraph {
  const uint32_t &local_infinity;
  cll::opt<unsigned long long> &local_src_node;
  Graph *graph;

  InitializeGraph(cll::opt<unsigned long long> &_src_node, 
                  const uint32_t &_infinity, Graph* _graph) : 
    local_infinity(_infinity), local_src_node(_src_node), graph(_graph){}

  void static go(Graph& _graph){
    auto& allNodes = _graph.allNodesRange();

    #ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) {
        std::string impl_str(_graph.get_run_identifier(
                               "CUDA_DO_ALL_IMPL_InitializeGraph_"));
        galois::StatTimer StatTimer_cuda(impl_str.c_str());
        StatTimer_cuda.start();
        InitializeGraph_cuda(*(allNodes.begin()), *(allNodes.end()),
                             infinity, src_node, cuda_ctx);
        StatTimer_cuda.stop();
      } else if (personality == CPU)
    #endif
    {

    galois::do_all_local(
      allNodes,
      InitializeGraph{src_node, infinity, &_graph}, 
      galois::loopname(_graph.get_run_identifier("InitializeGraph").c_str()),
      galois::do_all_steal<true>(),
      galois::timeit()
    );

    }
  }

  void operator()(GNode src) const {
    NodeData& sdata = graph->getData(src);
    sdata.dist_current = (graph->getGID(src) == local_src_node) ? 0 : 
                                                                  local_infinity;
    sdata.dist_old = (graph->getGID(src) == local_src_node) ? 0 : 
                                                              local_infinity;
  }
};

struct FirstItr_BFS{
  Graph * graph;

  FirstItr_BFS(Graph * _graph) : graph(_graph) {}

  void static go(Graph& _graph) {
    uint32_t __begin, __end;
    if (_graph.isLocal(src_node)) {
      __begin = _graph.getLID(src_node);
      __end = __begin + 1;
    } else {
      __begin = 0;
      __end = 0;
    }

    _graph.sync_on_demand<Flags_dist_current, readSource, Reduce_min_dist_current,
                          Broadcast_dist_current, Bitset_dist_current>("BFS");

  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      std::string impl_str(_graph.get_run_identifier(
                             "CUDA_DO_ALL_IMPL_BFS_"));
      galois::StatTimer StatTimer_cuda(impl_str.c_str());
      StatTimer_cuda.start();
      FirstItr_BFS_cuda(__begin, __end, cuda_ctx);
      StatTimer_cuda.stop();
    } else if (personality == CPU)
  #endif
    {
    // one node, doesn't matter which do_all you use, so regular one suffices
    galois::do_all(_graph.begin() + __begin, _graph.begin() + __end,
                FirstItr_BFS{&_graph}, 
                galois::loopname(_graph.get_run_identifier("BFS").c_str()),
                galois::timeit());
    }

    Flags_dist_current::set_write_dst();

    galois::runtime::reportStat("(NULL)", 
       _graph.get_run_identifier("NUM_WORK_ITEMS_"), __end - __begin, 0);
  }

  void operator()(GNode src) const {
    NodeData& snode = graph->getData(src);
    snode.dist_old = snode.dist_current;

    for (auto jj = graph->edge_begin(src), ee = graph->edge_end(src); 
         jj != ee; 
         ++jj) {
      GNode dst = graph->getEdgeDst(jj);
      auto& dnode = graph->getData(dst);
      uint32_t new_dist = 1 + snode.dist_current;
      uint32_t old_dist = galois::atomicMin(dnode.dist_current, new_dist);
      if (old_dist > new_dist) bitset_dist_current.set(dst);
    }
  }
};

struct BFS {
  Graph* graph;
  galois::DGAccumulator<unsigned int>& DGAccumulator_accum;

  BFS(Graph* _graph, galois::DGAccumulator<unsigned int>& _dga) : 
    graph(_graph), DGAccumulator_accum(_dga) {}

  void static go(Graph& _graph, galois::DGAccumulator<unsigned int>& dga) {
    using namespace galois::worklists;
    
    FirstItr_BFS::go(_graph);

    unsigned _num_iterations = 1;

    auto& nodesWithEdges = _graph.allNodesWithEdgesRange();

    
    do { 
      _graph.set_num_iter(_num_iterations);
      dga.reset();

      _graph.sync_on_demand<Flags_dist_current, readSource, Reduce_min_dist_current,
                            Broadcast_dist_current, Bitset_dist_current>("BFS");
    #ifdef __GALOIS_HET_CUDA__
      if (personality == GPU_CUDA) {
        std::string impl_str(_graph.get_run_identifier("CUDA_DO_ALL_IMPL_BFS"));
        galois::StatTimer StatTimer_cuda(impl_str.c_str());
        StatTimer_cuda.start();
        int __retval = 0;
        BFS_cuda(*nodesWithEdges.begin(), *nodesWithEdges.end(), 
                 __retval, cuda_ctx);
        dga += __retval;
        StatTimer_cuda.stop();
      } else if (personality == CPU)
    #endif
    {
      galois::do_all_local(
        nodesWithEdges,
        BFS(&_graph, dga),
        galois::loopname(_graph.get_run_identifier("BFS").c_str()),
        galois::do_all_steal<true>(),
        galois::timeit()
      );
    }

      Flags_dist_current::set_write_dst();

      //_graph.sync<writeDestination, readSource, Reduce_min_dist_current, 
      //            Broadcast_dist_current, Bitset_dist_current>("BFS");

      galois::runtime::reportStat("(NULL)", 
        _graph.get_run_identifier("NUM_WORK_ITEMS_"), 
        (unsigned long)dga.read_local(), 0);
      ++_num_iterations;
    } while ((_num_iterations < maxIterations) && dga.reduce());

    if (galois::runtime::getSystemNetworkInterface().ID == 0) {
      galois::runtime::reportStat("(NULL)", 
        "NUM_ITERATIONS_" + std::to_string(_graph.get_run_num()), 
        (unsigned long)_num_iterations, 0);
    }
  }

  void operator()(GNode src) const {
    NodeData& snode = graph->getData(src);

    if (snode.dist_old > snode.dist_current) {
      snode.dist_old = snode.dist_current;

      for (auto jj = graph->edge_begin(src), ee = graph->edge_end(src); 
           jj != ee;
           ++jj) {
        GNode dst = graph->getEdgeDst(jj);
        auto& dnode = graph->getData(dst);
        uint32_t new_dist = 1 + snode.dist_current;
        uint32_t old_dist = galois::atomicMin(dnode.dist_current, new_dist);
        if (old_dist > new_dist) bitset_dist_current.set(dst);
      }

      DGAccumulator_accum += 1;
    }
  }
};

/******************************************************************************/
/* Sanity check operators */
/******************************************************************************/

/* Prints total number of nodes visited + max distance */
struct BFSSanityCheck {
  const uint32_t &local_infinity;
  Graph* graph;

  static uint32_t current_max;

  galois::DGAccumulator<uint64_t>& DGAccumulator_sum;
  galois::DGAccumulator<uint32_t>& DGAccumulator_max;

  BFSSanityCheck(const uint32_t _infinity, Graph* _graph, 
                 galois::DGAccumulator<uint64_t>& dgas,
                 galois::DGAccumulator<uint32_t>& dgam) : 
    local_infinity(_infinity), graph(_graph), DGAccumulator_sum(dgas),
    DGAccumulator_max(dgam) {}

  void static go(Graph& _graph, galois::DGAccumulator<uint64_t>& dgas,
                 galois::DGAccumulator<uint32_t>& dgam) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      // TODO currently no GPU support for sanity check operator
      fprintf(stderr, "Warning: No GPU support for sanity check; might get "
                      "wrong results.\n");
    }
  #endif
    dgas.reset();
    dgam.reset();

    galois::do_all(_graph.begin(), _graph.end(), 
                   BFSSanityCheck(infinity, &_graph, dgas, dgam), 
                   galois::loopname("BFSSanityCheck"));

    uint64_t num_visited = dgas.reduce();

    dgam = current_max;
    uint32_t max_distance = dgam.reduce_max();

    // Only node 0 will print the info
    if (_graph.id == 0) {
      printf("Number of nodes visited is %lu\n", num_visited);
      printf("Max distance is %u\n", max_distance);
    }
  }

  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    if (graph->isOwned(graph->getGID(src)) && 
        src_data.dist_current < local_infinity) {
      DGAccumulator_sum += 1;

      if (current_max < src_data.dist_current) {
        current_max = src_data.dist_current;
      }
    }
  }
};
uint32_t BFSSanityCheck::current_max = 0;

/******************************************************************************/
/* Main */
/******************************************************************************/

int main(int argc, char** argv) {
  try {
    galois::System G;
    LonestarStart(argc, argv, name, desc, url);
    galois::StatManager statManager(statOutputFile);
    {
    auto& net = galois::runtime::getSystemNetworkInterface();
    if (net.ID == 0) {
      galois::runtime::reportStat("(NULL)", "Max Iterations", 
                                  (unsigned long)maxIterations, 0);
      galois::runtime::reportStat("(NULL)", "Source Node ID", 
                                  (unsigned long long)src_node, 0);
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

    hg = constructGraph<NodeData, void>(scalefactor);

#ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      cuda_ctx = get_CUDA_context(my_host_id);
      if (!init_CUDA_context(cuda_ctx, gpu_device))
        return -1;
      MarshalGraph m = (*hg).getMarshalGraph(my_host_id);
      load_graph_CUDA(cuda_ctx, m, net.Num);
    } else if (personality == GPU_OPENCL) {
      //galois::OpenCL::cl_env.init(cldevice.Value);
    }
#endif
    bitset_dist_current.resize(hg->get_local_total_nodes());
    StatTimer_hg_init.stop();

    std::cout << "[" << net.ID << "] InitializeGraph::go called\n";
    StatTimer_init.start();
      InitializeGraph::go((*hg));
    StatTimer_init.stop();

    // accumulators for use in operators
    galois::DGAccumulator<unsigned int> DGAccumulator_accum;
    galois::DGAccumulator<uint64_t> DGAccumulator_sum;
    galois::DGAccumulator<uint32_t> DGAccumulator_max;

    for(auto run = 0; run < numRuns; ++run){
      std::cout << "[" << net.ID << "] BFS::go run " << run << " called\n";
      std::string timer_str("TIMER_" + std::to_string(run));
      galois::StatTimer StatTimer_main(timer_str.c_str());

      StatTimer_main.start();
        BFS::go(*hg, DGAccumulator_accum);
      StatTimer_main.stop();

      // sanity check
      BFSSanityCheck::current_max = 0;
      BFSSanityCheck::go(*hg, DGAccumulator_sum, DGAccumulator_max);

      if((run + 1) != numRuns){
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) { 
          bitset_dist_current_reset_cuda(cuda_ctx);
        } else
      #endif
        bitset_dist_current.reset();

        //galois::runtime::getHostBarrier().wait();
        (*hg).reset_num_iter(run+1);
        InitializeGraph::go((*hg));
      }
    }

    StatTimer_total.stop();

    // Verify
    if(verify){
#ifdef __GALOIS_HET_CUDA__
      if (personality == CPU) { 
#endif
        for (auto ii = (*hg).masterNodesRange().begin(); 
                  ii != (*hg).masterNodesRange().end(); 
                  ++ii) {
            galois::runtime::printOutput("% %\n", (*hg).getGID(*ii), 
                                         (*hg).getData(*ii).dist_current);
        }
#ifdef __GALOIS_HET_CUDA__
      } else if (personality == GPU_CUDA)  {
        for (auto ii = (*hg).masterNodesRange().begin(); 
                  ii != (*hg).masterNodesRange().end(); 
                  ++ii) {
            galois::runtime::printOutput("% %\n", (*hg).getGID(*ii), 
                                     get_node_dist_current_cuda(cuda_ctx, *ii));
        }
      }
#endif
    }
    }
    galois::runtime::getHostBarrier().wait();
    statManager.reportStat();

    return 0;
  } catch(const char* c) {
    std::cerr << "Error: " << c << "\n";
      return 1;
  }
}
