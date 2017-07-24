/** Residual based Page Rank -*- C++ -*-
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
 * Compute pageRank using residual on distributed Galois.
 *
 * @author Gurbinder Gill <gurbinder533@gmail.com>
 * @author Loc Hoang <l_hoang@utexas.edu> (sanity check operators)
 */

#include <iostream>
#include <limits>
#include <algorithm>
#include <vector>
#include "Galois/Galois.h"
#include "Galois/DoAllWrap.h"
#include "Lonestar/BoilerPlate.h"
#include "Galois/gstl.h"

#include "Galois/Runtime/CompilerHelperFunctions.h"
#include "Galois/Runtime/Tracer.h"

#include "Galois/Runtime/dGraph_edgeCut.h"
#include "Galois/Runtime/dGraph_cartesianCut.h"
#include "Galois/Runtime/dGraph_hybridCut.h"

#include "Galois/DistAccumulator.h"

enum VertexCut {
  PL_VCUT, CART_VCUT
};

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
   assert(false&& "Invalid personality");
   return "";
}
#endif

/******************************************************************************/
/* Declaration of command line arguments */
/******************************************************************************/

static const char* const name = "PageRank - Compiler Generated Distributed Heterogeneous";
static const char* const desc = "Residual PageRank on Distributed Galois.";
static const char* const url = 0;

namespace cll = llvm::cl;
static cll::opt<std::string> inputFile(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<std::string> partFolder("partFolder", cll::desc("path to partitionFolder"), cll::init(""));
static cll::opt<bool> transpose("transpose", cll::desc("transpose the graph in memory after partitioning"), cll::init(false));
static cll::opt<float> tolerance("tolerance", cll::desc("tolerance"), cll::init(0.000001));
static cll::opt<unsigned int> maxIterations("maxIterations", cll::desc("Maximum iterations: Default 1000"), cll::init(1000));
static cll::opt<bool> verify("verify", cll::desc("Verify ranks by printing to 'page_ranks.#hid.csv' file"), cll::init(false));

static cll::opt<bool> enableVCut("enableVertexCut", cll::desc("Use vertex cut for graph partitioning."), cll::init(false));

static cll::opt<unsigned int> VCutThreshold("VCutThreshold", cll::desc("Threshold for high degree edges."), cll::init(100));
static cll::opt<VertexCut> vertexcut("vertexcut", cll::desc("Type of vertex cut."),
       cll::values(clEnumValN(PL_VCUT, "pl_vcut", "Powerlyra Vertex Cut"), clEnumValN(CART_VCUT , "cart_vcut", "Cartesian Vertex Cut"), clEnumValEnd),
       cll::init(PL_VCUT));

#ifdef __GALOIS_HET_CUDA__
static cll::opt<int> gpudevice("gpu", cll::desc("Select GPU to run on, default is to choose automatically"), cll::init(-1));
static cll::opt<Personality> personality("personality", cll::desc("Personality"),
      cll::values(clEnumValN(CPU, "cpu", "Galois CPU"), clEnumValN(GPU_CUDA, "gpu/cuda", "GPU/CUDA"), clEnumValN(GPU_OPENCL, "gpu/opencl", "GPU/OpenCL"), clEnumValEnd),
      cll::init(CPU));
static cll::opt<unsigned> scalegpu("scalegpu", cll::desc("Scale GPU workload w.r.t. CPU, default is proportionally equal workload to CPU and GPU (1)"), cll::init(1));
static cll::opt<unsigned> scalecpu("scalecpu", cll::desc("Scale CPU workload w.r.t. GPU, default is proportionally equal workload to CPU and GPU (1)"), cll::init(1));
static cll::opt<int> num_nodes("num_nodes", cll::desc("Num of physical nodes with devices (default = num of hosts): detect GPU to use for each host automatically"), cll::init(-1));
static cll::opt<std::string> personality_set("pset", cll::desc("String specifying personality for hosts on each physical node. 'c'=CPU,'g'=GPU/CUDA and 'o'=GPU/OpenCL"), cll::init("c"));
#endif

/******************************************************************************/
/* Graph structure declarations + other initialization */
/******************************************************************************/

static const float alpha = (1.0 - 0.85);
struct NodeData {
  float value;
  float delta;
  std::atomic<float> residual;
  std::atomic<uint32_t> nout;
};

Galois::DynamicBitSet bitset_residual;
Galois::DynamicBitSet bitset_nout;

typedef hGraph<NodeData, void> Graph;
typedef hGraph_edgeCut<NodeData, void> Graph_edgeCut;
typedef hGraph_vertexCut<NodeData, void> Graph_vertexCut;
typedef hGraph_cartesianCut<NodeData, void> Graph_cartesianCut;

typedef typename Graph::GraphNode GNode;

typedef GNode WorkItem;

#include "gen_sync.hh"

/******************************************************************************/
/* Algorithm structures */
/******************************************************************************/

struct ResetGraph {
  Graph* graph;

  ResetGraph(Graph* _graph) : graph(_graph){}
  void static go(Graph& _graph) {
    #ifdef __GALOIS_HET_CUDA__
    	if (personality == GPU_CUDA) {
    		std::string impl_str("CUDA_DO_ALL_IMPL_ResetGraph_" + (_graph.get_run_identifier()));
    		Galois::StatTimer StatTimer_cuda(impl_str.c_str());
    		StatTimer_cuda.start();
    		ResetGraph_all_cuda(cuda_ctx);
    		StatTimer_cuda.stop();
    	} else if (personality == CPU)
    #endif
    Galois::do_all(_graph.begin(), _graph.end(), 
                   ResetGraph{ &_graph }, 
                   Galois::loopname("ResetGraph"), 
                   Galois::numrun(_graph.get_run_identifier()));
  }

  void operator()(GNode src) const {
    NodeData& sdata = graph->getData(src);
    sdata.value = 0;
    sdata.nout = 0;
    sdata.residual = 0;
    sdata.delta = 0;
  }
};

struct InitializeGraphNout {
  Graph* graph;

  InitializeGraphNout(Graph* _graph) : graph(_graph){}
  void static go(Graph& _graph) {
    #ifdef __GALOIS_HET_CUDA__
    	if (personality == GPU_CUDA) {
    		std::string impl_str("CUDA_DO_ALL_IMPL_InitializeGraphNout_" + (_graph.get_run_identifier()));
    		Galois::StatTimer StatTimer_cuda(impl_str.c_str());
    		StatTimer_cuda.start();
    		InitializeGraphNout_all_cuda(cuda_ctx);
    		StatTimer_cuda.stop();
    	} else if (personality == CPU)
    #endif
    {
    Galois::do_all(_graph.begin(), _graph.end(), 
                   InitializeGraphNout{ &_graph }, 
                   Galois::loopname("InitializeGraphNout"), 
                   Galois::numrun(_graph.get_run_identifier()), 
                   Galois::write_set("reduce", "this->graph", 
                     "struct NodeData &", "struct PR_NodeData &" , "nout", 
                     "float" , "add",  "0"));
    }
    _graph.sync<writeSource, readSource, Reduce_add_nout, Broadcast_nout,
                Bitset_nout>("InitializeGraphNout");
  }

  void operator()(GNode src) const {
    NodeData& sdata = graph->getData(src);
    Galois::atomicAdd(sdata.nout, 
      (uint32_t) std::distance(graph->edge_begin(src), 
                                   graph->edge_end(src)));
    bitset_nout.set(src);
  }
};

struct InitializeGraph {
  const float &local_alpha;
  Graph* graph;

  InitializeGraph(const float &_alpha, Graph* _graph) : 
    local_alpha(_alpha), graph(_graph){}

  void static go(Graph& _graph) {
    #ifdef __GALOIS_HET_CUDA__
    	if (personality == GPU_CUDA) {
    		std::string impl_str("CUDA_DO_ALL_IMPL_InitializeGraph_" + 
          (_graph.get_run_identifier()));
    		Galois::StatTimer StatTimer_cuda(impl_str.c_str());
    		StatTimer_cuda.start();
    		InitializeGraph_all_cuda(alpha, cuda_ctx);
    		StatTimer_cuda.stop();
    	} else if (personality == CPU)
    #endif
    {
    Galois::do_all(_graph.begin(), _graph.end(), 
      InitializeGraph{ alpha, &_graph }, Galois::loopname("InitializeGraph"), 
      Galois::numrun(_graph.get_run_identifier()), 
      Galois::write_set("reduce", "this->graph", "struct NodeData &", 
        "struct PR_NodeData &" , "residual", "float" , "add",  "0"));
    }

    _graph.sync<writeDestination, readSource, Reduce_add_residual, 
                Broadcast_residual, Bitset_residual>("InitializeGraph");
  }

  void operator()(GNode src) const {
    NodeData& sdata = graph->getData(src);
    sdata.value = local_alpha;

    if (sdata.nout > 0) {
      float delta = sdata.value * (1 - local_alpha) / sdata.nout;
      for(auto nbr = graph->edge_begin(src), ee = graph->edge_end(src); 
          nbr != ee; 
          ++nbr){
        GNode dst = graph->getEdgeDst(nbr);
        NodeData& ddata = graph->getData(dst);
        Galois::atomicAdd(ddata.residual, delta);
        bitset_residual.set(dst);
      }
    }
  }
};

struct PageRankCopy {
  const float & local_alpha;
  cll::opt<float> & local_tolerance;
  Graph* graph;

  PageRankCopy(const float & _local_alpha, cll::opt<float> & _local_tolerance,
               Graph * _graph) : 
                 local_alpha(_local_alpha),
                 local_tolerance(_local_tolerance),
                 graph(_graph){}

  void static go(Graph& _graph) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      std::string impl_str("CUDA_DO_ALL_IMPL_PageRank_" + (_graph.get_run_identifier()));
      Galois::StatTimer StatTimer_cuda(impl_str.c_str());
      StatTimer_cuda.start();
      PageRankCopy_all_cuda(alpha, tolerance, cuda_ctx);
      StatTimer_cuda.stop();
    } else if (personality == CPU)
  #endif
    {
      //Galois::do_all_choice(Galois::Runtime::makeStandardRange(_graph.begin(), _graph.end()), 
      //               PageRankCopy{ alpha, tolerance, &_graph }, 
      //               std::make_tuple(Galois::loopname("PageRank"), 
      //               Galois::thread_range(_graph.get_thread_ranges()),
      //               Galois::numrun(_graph.get_run_identifier())));
      Galois::do_all_local(_graph.get_thread_ranges(),
                     PageRankCopy{ alpha, tolerance, &_graph }, 
                     Galois::loopname("PageRank"), 
                     Galois::numrun(_graph.get_run_identifier()));
    }
  }

  void operator()(WorkItem src) const {
    NodeData& sdata = graph->getData(src);
    if (sdata.residual > this->local_tolerance) {
      float residual_old = sdata.residual;
      sdata.residual = 0;
      sdata.value += residual_old;
      if (sdata.nout > 0) {
        sdata.delta = residual_old * (1 - local_alpha) / sdata.nout;
      }
    }
  }
};

struct FirstItr_PageRank{
  Graph * graph;
  FirstItr_PageRank(Graph * _graph):graph(_graph){}

  void static go(Graph& _graph) {
#ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      std::string impl_str("CUDA_DO_ALL_IMPL_PageRank_" + (_graph.get_run_identifier()));
      Galois::StatTimer StatTimer_cuda(impl_str.c_str());
      StatTimer_cuda.start();
      FirstItr_PageRank_all_cuda(cuda_ctx);
      StatTimer_cuda.stop();
    } else if (personality == CPU)
#endif
    {
      //Galois::do_all_choice(Galois::Runtime::makeStandardRange(_graph.begin(), _graph.end()), 
      //               FirstItr_PageRank{&_graph},
      //               std::make_tuple(Galois::loopname("PageRank"), 
      //               Galois::thread_range(_graph.get_thread_ranges()),
      //               Galois::numrun(_graph.get_run_identifier()), 
      //               Galois::write_set("reduce", "this->graph", 
      //                 "struct NodeData &", "struct PR_NodeData &" , "residual", 
      //                 "float" , "add",  "0")));
      Galois::do_all_local(_graph.get_thread_ranges(),
                     FirstItr_PageRank{&_graph},
                     Galois::loopname("PageRank"), 
                     Galois::numrun(_graph.get_run_identifier()), 
                     Galois::write_set("reduce", "this->graph", 
                       "struct NodeData &", "struct PR_NodeData &" , "residual", 
                       "float" , "add",  "0"));
    }
  _graph.sync<writeDestination, readSource, Reduce_add_residual, 
              Broadcast_residual, Bitset_residual>("PageRank");
  Galois::Runtime::reportStat("(NULL)", 
    "NUM_WORK_ITEMS_" + (_graph.get_run_identifier()), 
    _graph.end() - _graph.begin(), 0);
  }

  void operator()(WorkItem src) const {
    NodeData& sdata = graph->getData(src);

    if (sdata.delta > 0) {
      float delta = sdata.delta;
      sdata.delta = 0;
      for (auto nbr = graph->edge_begin(src), ee = graph->edge_end(src); 
           nbr != ee; 
           ++nbr){
        GNode dst = graph->getEdgeDst(nbr);
        NodeData& ddata = graph->getData(dst);
        Galois::atomicAdd(ddata.residual, delta);
        bitset_residual.set(dst);
      }
    }
  }
};

struct PageRank {
  Graph* graph;
  static Galois::DGAccumulator<int> DGAccumulator_accum;

  PageRank(Graph* _g): graph(_g){}

  void static go(Graph& _graph) {
    PageRankCopy::go(_graph);
    FirstItr_PageRank::go(_graph);
    
    unsigned _num_iterations = 1;
    
    do { 
      _graph.set_num_iter(_num_iterations);
      PageRankCopy::go(_graph);
      DGAccumulator_accum.reset();
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          std::string impl_str("CUDA_DO_ALL_IMPL_PageRank_" + (_graph.get_run_identifier()));
          Galois::StatTimer StatTimer_cuda(impl_str.c_str());
          StatTimer_cuda.start();
          int __retval = 0;
          PageRank_all_cuda(__retval, cuda_ctx);
          DGAccumulator_accum += __retval;
          StatTimer_cuda.stop();
        } else if (personality == CPU)
      #endif
        {
          //Galois::do_all_choice(Galois::Runtime::makeStandardRange(_graph.begin(), _graph.end()), 
          //               PageRank{ &_graph }, 
          //               std::make_tuple(Galois::loopname("PageRank"), 
          //               Galois::thread_range(_graph.get_thread_ranges()),
          //               Galois::write_set("reduce", "this->graph", 
          //                                 "struct NodeData &", 
          //                                 "struct PR_NodeData &" , "residual", 
          //                                 "float" , "add",  "0"), 
          //               Galois::numrun(_graph.get_run_identifier())));

          Galois::do_all_local(_graph.get_thread_ranges(),
                         PageRank{ &_graph }, 
                         Galois::loopname("PageRank"), 
                         Galois::write_set("reduce", "this->graph", 
                                           "struct NodeData &", 
                                           "struct PR_NodeData &" , "residual", 
                                           "float" , "add",  "0"), 
                         Galois::numrun(_graph.get_run_identifier()));

        }
      _graph.sync<writeDestination, readSource, Reduce_add_residual, 
                  Broadcast_residual, Bitset_residual>("PageRank");
      
      Galois::Runtime::reportStat("(NULL)", 
          "NUM_WORK_ITEMS_" + (_graph.get_run_identifier()), 
          (unsigned long)DGAccumulator_accum.read_local(), 0);

      ++_num_iterations;
    } while((_num_iterations < maxIterations) && DGAccumulator_accum.reduce());

    if (Galois::Runtime::getSystemNetworkInterface().ID == 0) {
      Galois::Runtime::reportStat("(NULL)", 
        "NUM_ITERATIONS_" + std::to_string(_graph.get_run_num()), 
        (unsigned long)_num_iterations, 0);
    }
  }

  void operator()(WorkItem src) const {
    NodeData& sdata = graph->getData(src);

    if (sdata.delta > 0) {
      float delta = sdata.delta;
      sdata.delta = 0;

      for(auto nbr = graph->edge_begin(src), ee = graph->edge_end(src); 
          nbr != ee; ++nbr) {
        GNode dst = graph->getEdgeDst(nbr);
        NodeData& ddata = graph->getData(dst);
        Galois::atomicAdd(ddata.residual, delta);
        bitset_residual.set(dst);
      }
      DGAccumulator_accum+= 1;
    }
  }
};
Galois::DGAccumulator<int> PageRank::DGAccumulator_accum;

/******************************************************************************/
/* Sanity check operators */
/******************************************************************************/

// gets rank max, min and sum across all nodes
struct PageRankSanity {
  cll::opt<float>& local_tolerance;
  Graph* graph;

  static float current_max;
  static float current_min;
  static float current_max_residual;
  static float current_min_residual;

  static Galois::DGAccumulator<float> DGAccumulator_max;
  static Galois::DGAccumulator<float> DGAccumulator_min;
  static Galois::DGAccumulator<float> DGAccumulator_sum;
  static Galois::DGAccumulator<float> DGAccumulator_sum_residual;
  static Galois::DGAccumulator<uint64_t> DGAccumulator_residual_over_tolerance;
  static Galois::DGAccumulator<float> DGAccumulator_max_residual;
  static Galois::DGAccumulator<float> DGAccumulator_min_residual;

  PageRankSanity(cll::opt<float>& _local_tolerance, Graph* _graph) : 
    local_tolerance(_local_tolerance), graph(_graph) {}

  void static go(Graph& _graph) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      // TODO currently no GPU support for sanity check operator
      printf("Warning: No GPU support for sanity check; might get "
             "wrong results.\n");
    }
  #endif

    DGAccumulator_max.reset();
    DGAccumulator_min.reset();
    DGAccumulator_sum.reset();
    DGAccumulator_sum_residual.reset();
    DGAccumulator_residual_over_tolerance.reset();
    DGAccumulator_max_residual.reset();
    DGAccumulator_min_residual.reset();

    Galois::do_all(_graph.begin(), _graph.end(), 
                   PageRankSanity(tolerance, &_graph), 
                   Galois::loopname("PageRankSanity"));

    DGAccumulator_max = current_max;
    DGAccumulator_min = current_min;
    DGAccumulator_max_residual = current_max_residual;
    DGAccumulator_min_residual = current_min_residual;

    float max_rank = DGAccumulator_max.reduce_max();
    float min_rank = DGAccumulator_min.reduce_min();
    float rank_sum = DGAccumulator_sum.reduce();
    float residual_sum = DGAccumulator_sum_residual.reduce();
    uint64_t over_tolerance = DGAccumulator_residual_over_tolerance.reduce();
    float max_residual = DGAccumulator_max_residual.reduce_max();
    float min_residual = DGAccumulator_min_residual.reduce_min();

    // Only node 0 will print data
    if (_graph.id == 0) {
      printf("Max rank is %f\n", max_rank);
      printf("Min rank is %f\n", min_rank);
      printf("Rank sum is %f\n", rank_sum);
      printf("Residual sum is %f\n", residual_sum);
      printf("# nodes with residual over tolerance is %lu\n", over_tolerance);
      printf("Max residual is %f\n", max_residual);
      printf("Min residual is %f\n", min_residual);
    }
  }
  
  /* Gets the max, min rank from all owned nodes and
   * also the sum of ranks */
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    if (graph->isOwned(graph->getGID(src))) {
      if (current_max < src_data.value) {
        current_max = src_data.value;
      }

      if (current_min > src_data.value) {
        current_min = src_data.value;
      }

      if (current_max_residual < src_data.residual) {
        current_max_residual = src_data.residual;
      }

      if (current_min_residual > src_data.residual) {
        current_min_residual = src_data.residual;
      }

      if (src_data.residual > local_tolerance) {
        DGAccumulator_residual_over_tolerance += 1;
      }

      DGAccumulator_sum += src_data.value;
      DGAccumulator_sum_residual += src_data.residual;
    }
  }
};

Galois::DGAccumulator<float> PageRankSanity::DGAccumulator_max;
Galois::DGAccumulator<float> PageRankSanity::DGAccumulator_min;
Galois::DGAccumulator<float> PageRankSanity::DGAccumulator_sum;
Galois::DGAccumulator<float> PageRankSanity::DGAccumulator_sum_residual;
Galois::DGAccumulator<uint64_t> PageRankSanity::DGAccumulator_residual_over_tolerance;
Galois::DGAccumulator<float> PageRankSanity::DGAccumulator_max_residual;
Galois::DGAccumulator<float> PageRankSanity::DGAccumulator_min_residual;

float PageRankSanity::current_max = 0;
float PageRankSanity::current_min = std::numeric_limits<float>::max() / 4;

float PageRankSanity::current_max_residual = 0;
float PageRankSanity::current_min_residual = std::numeric_limits<float>::max() / 4;


/******************************************************************************/
/* Main */
/******************************************************************************/

int main(int argc, char** argv) {
  try {
    LonestarStart(argc, argv, name, desc, url);
    auto& net = Galois::Runtime::getSystemNetworkInterface();
    Galois::StatManager statManager(statOutputFile);
    {
    if (net.ID == 0) {
      Galois::Runtime::reportStat("(NULL)", "Max Iterations", 
                                  (unsigned long)maxIterations, 0);
      std::ostringstream ss;
      ss << tolerance;
      Galois::Runtime::reportStat("(NULL)", "Tolerance", ss.str(), 0);
    }
    Galois::StatTimer StatTimer_init("TIMER_GRAPH_INIT"),
                      StatTimer_total("TIMER_TOTAL"),
                      StatTimer_hg_init("TIMER_HG_INIT");

    StatTimer_total.start();

    std::vector<unsigned> scalefactor;
#ifdef __GALOIS_HET_CUDA__
    const unsigned my_host_id = Galois::Runtime::getHostID();
    int gpu_device = gpudevice;
    //Parse arg string when running on multiple hosts and update/override personality
    //with corresponding value.
    if (num_nodes == -1) num_nodes = net.Num;
    assert((net.Num % num_nodes) == 0);
    if (personality_set.length() == (net.Num / num_nodes)) {
      switch (personality_set.c_str()[my_host_id % num_nodes]) {
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

    if (enableVCut){
      if (vertexcut == CART_VCUT)
        hg = new Graph_cartesianCut(inputFile, partFolder, net.ID, net.Num,
                                    scalefactor, transpose, Galois::doAllKind==Galois::DOALL_RANGE);
      else if (vertexcut == PL_VCUT)
        hg = new Graph_vertexCut(inputFile, partFolder, net.ID, net.Num, 
                                 scalefactor, transpose, VCutThreshold, false, Galois::doAllKind==Galois::DOALL_RANGE);
    } else {
      hg = new Graph_edgeCut(inputFile, partFolder, net.ID, net.Num, scalefactor,
                             transpose, Galois::doAllKind==Galois::DOALL_RANGE);
    }


#ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      cuda_ctx = get_CUDA_context(my_host_id);
      if (!init_CUDA_context(cuda_ctx, gpu_device))
        return -1;
      MarshalGraph m = (*hg).getMarshalGraph(my_host_id);
      load_graph_CUDA(cuda_ctx, m, net.Num);
    } else if (personality == GPU_OPENCL) {
      //Galois::OpenCL::cl_env.init(cldevice.Value);
    }
#endif
    bitset_residual.resize(hg->get_local_total_nodes());
    bitset_nout.resize(hg->get_local_total_nodes());
    StatTimer_hg_init.stop();

    std::cout << "[" << net.ID << "] InitializeGraph::go called\n";
    StatTimer_init.start();
      InitializeGraphNout::go((*hg));
      InitializeGraph::go((*hg));
    StatTimer_init.stop();


    for (auto run = 0; run < numRuns; ++run) {
      std::cout << "[" << net.ID << "] PageRank::go run " << run << " called\n";
      std::string timer_str("TIMER_" + std::to_string(run));
      Galois::StatTimer StatTimer_main(timer_str.c_str());

      StatTimer_main.start();
        PageRank::go((*hg));
      StatTimer_main.stop();

      // sanity check
      PageRankSanity::current_max = 0;
      PageRankSanity::current_min = std::numeric_limits<float>::max() / 4;

      PageRankSanity::current_max_residual = 0;
      PageRankSanity::current_min_residual = std::numeric_limits<float>::max() / 4;

      PageRankSanity::go(*hg);

      if((run + 1) != numRuns){
        //Galois::Runtime::getHostBarrier().wait();
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) { 
          bitset_residual_reset_cuda(cuda_ctx);
          bitset_nout_reset_cuda(cuda_ctx);
        } else
      #endif
        { bitset_residual.reset();
        bitset_nout.reset(); }

        (*hg).reset_num_iter(run+1);
        ResetGraph::go((*hg));
        InitializeGraphNout::go((*hg));
        InitializeGraph::go((*hg));
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
            Galois::Runtime::printOutput("% %\n", (*hg).getGID(*ii), 
              (*hg).getData(*ii).value);
        }
#ifdef __GALOIS_HET_CUDA__
      } else if (personality == GPU_CUDA)  {
        for (auto ii = (*hg).begin(); ii != (*hg).end(); ++ii) {
          if ((*hg).isOwned((*hg).getGID(*ii))) 
            Galois::Runtime::printOutput("% %\n", (*hg).getGID(*ii), 
              get_node_value_cuda(cuda_ctx, *ii));
        }
      }
#endif
    }

    }
    statManager.reportStat();

    return 0;
  } catch (const char* c) {
      std::cerr << "Error: " << c << "\n";
      return 1;
  }
}
