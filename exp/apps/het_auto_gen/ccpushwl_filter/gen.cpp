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
 * Compute ConnectedComp on distributed Galois.
 *
 * @author Gurbinder Gill <gurbinder533@gmail.com>
 */

#include <iostream>
#include <limits>
#include "Galois/Galois.h"
#include "Galois/gstl.h"
#include "Lonestar/BoilerPlate.h"
#include "Galois/Runtime/CompilerHelperFunctions.h"

#include "Galois/Dist/OfflineGraph.h"
#include "Galois/Dist/hGraph.h"
#include "Galois/DistAccumulator.h"
#include "Galois/Runtime/Tracer.h"

#ifdef __GALOIS_HET_CUDA__
#include "Galois/Cuda/cuda_mtypes.h"
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

static const char* const name = "ConnectedComp - Distributed Heterogeneous";
static const char* const desc = "ConnectedComp on Distributed Galois with 2 loop transform.";
static const char* const url = 0;

namespace cll = llvm::cl;
static cll::opt<std::string> inputFile(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<unsigned int> maxIterations("maxIterations", cll::desc("Maximum iterations: Default 1024"), cll::init(1024));
static cll::opt<bool> verify("verify", cll::desc("Verify ranks by printing to 'page_ranks.#hid.csv' file"), cll::init(false));
#ifdef __GALOIS_HET_CUDA__
static cll::opt<int> gpudevice("gpu", cll::desc("Select GPU to run on, default is to choose automatically"), cll::init(-1));
static cll::opt<Personality> personality("personality", cll::desc("Personality"),
      cll::values(clEnumValN(CPU, "cpu", "Galois CPU"), clEnumValN(GPU_CUDA, "gpu/cuda", "GPU/CUDA"), clEnumValN(GPU_OPENCL, "gpu/opencl", "GPU/OpenCL"), clEnumValEnd),
      cll::init(CPU));
static cll::opt<std::string> personality_set("pset", cll::desc("String specifying personality for each host. 'c'=CPU,'g'=GPU/CUDA and 'o'=GPU/OpenCL"), cll::init(""));
static cll::opt<unsigned> scalegpu("scalegpu", cll::desc("Scale GPU workload w.r.t. CPU, default is proportionally equal workload to CPU and GPU (1)"), cll::init(1));
static cll::opt<unsigned> scalecpu("scalecpu", cll::desc("Scale CPU workload w.r.t. GPU, default is proportionally equal workload to CPU and GPU (1)"), cll::init(1));
#endif

unsigned iteration;

struct NodeData {
  std::atomic<unsigned int> comp_current;
  unsigned int comp_old;
};

typedef hGraph<NodeData, void> Graph;
typedef typename Graph::GraphNode GNode;


struct InitializeGraph {
  unsigned long local_offset;
  Graph *graph;

  InitializeGraph(unsigned long _offset, Graph* _graph) : local_offset(_offset), graph(_graph){}
  void static go(Graph& _graph) {
    struct SyncerPull_0 {
      static unsigned int extract(uint32_t node_id, const struct NodeData & node) {
#ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) return get_node_comp_current_cuda(cuda_ctx, node_id);
        assert (personality == CPU);
#endif
        return node.comp_current;
      }
      static void setVal (uint32_t node_id, struct NodeData & node, unsigned int y) {
#ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) set_node_comp_current_cuda(cuda_ctx, node_id, y);
        else if (personality == CPU)
#endif
          node.comp_current = y;
      }
      typedef unsigned int ValTy;
    };
    struct SyncerPull_1 {
      static unsigned int extract(uint32_t node_id, const struct NodeData & node) {
#ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) return get_node_comp_old_cuda(cuda_ctx, node_id);
        assert (personality == CPU);
#endif
        return node.comp_old;
      }
      static void setVal (uint32_t node_id, struct NodeData & node, unsigned int y) {
#ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) set_node_comp_old_cuda(cuda_ctx, node_id, y);
        else if (personality == CPU)
#endif
          node.comp_old = y;
      }
      typedef unsigned int ValTy;
    };


    #ifdef __GALOIS_HET_CUDA__
    	if (personality == GPU_CUDA) {
    		InitializeGraph_cuda(_graph.getGlobalOffset(), cuda_ctx);
    	} else if (personality == CPU)
    #endif
    Galois::do_all(_graph.begin(), _graph.end(), InitializeGraph {_graph.getGlobalOffset(), &_graph}, Galois::loopname("InitGraph"));

    _graph.sync_pull<SyncerPull_0>();
    _graph.sync_pull<SyncerPull_1>();
  }

  void operator()(GNode src) const {
    NodeData& sdata = graph->getData(src);
    sdata.comp_current = src + local_offset;
    sdata.comp_old = src + local_offset;
  }
};

struct FirstItr_ConnectedComp {
  Graph* graph;

  FirstItr_ConnectedComp(Graph* _graph) : graph(_graph){}

  void static go(Graph& _graph){

    	struct Syncer_0 {
    		static unsigned int extract(uint32_t node_id, const struct NodeData & node) {
    		#ifdef __GALOIS_HET_CUDA__
    			if (personality == GPU_CUDA) return get_node_comp_current_cuda(cuda_ctx, node_id);
    			assert (personality == CPU);
    		#endif
    			return node.comp_current;
    		}
    		static void reduce (uint32_t node_id, struct NodeData & node, unsigned int y) {
    		#ifdef __GALOIS_HET_CUDA__
    			if (personality == GPU_CUDA) min_node_comp_current_cuda(cuda_ctx, node_id, y);
    			else if (personality == CPU)
    		#endif
    				{ Galois::atomicMin(node.comp_current, y);}
    		}
    		static void reset (uint32_t node_id, struct NodeData & node ) {
    		#ifdef __GALOIS_HET_CUDA__
    			if (personality == GPU_CUDA) set_node_comp_current_cuda(cuda_ctx, node_id, std::numeric_limits<unsigned int>::max()/4);
    			else if (personality == CPU)
    		#endif
    				{node.comp_current = std::numeric_limits<unsigned int>::max()/4; }
    		}
    		typedef unsigned int ValTy;
    	};
    #ifdef __GALOIS_HET_CUDA__
    	if (personality == GPU_CUDA) {
    		FirstItr_ConnectedComp_cuda(cuda_ctx);
    	} else if (personality == CPU)
    #endif
    Galois::for_each(_graph.begin(), _graph.end(), FirstItr_ConnectedComp { &_graph }, Galois::loopname("cc"), Galois::write_set("sync_push", "this->graph", "struct NodeData &", "struct NodeData &" , "comp_current", "unsigned int" , "{ Galois::atomicMin(node.comp_current, y);}",  "{node.comp_current = std::numeric_limits<unsigned int>::max()/4; }"));
    _graph.sync_push<Syncer_0>();
    

  }

  void operator()(GNode src, Galois::UserContext<GNode>& ctx) const {
    NodeData& snode = graph->getData(src);
    auto& sdist = snode.comp_current;

    for (auto jj = graph->edge_begin(src); jj != graph->edge_end(src); ++jj) {
      GNode dst = graph->getEdgeDst(jj);
      auto& dnode = graph->getData(dst);
      unsigned int new_dist = sdist;
      Galois::atomicMin(dnode.comp_current, new_dist);
    }
  }
};


struct ConnectedComp {
  Graph* graph;
  static Galois::DGAccumulator<int> DGAccumulator_accum;

  ConnectedComp(Graph* _graph) : graph(_graph){}
  void static go(Graph& _graph){
      FirstItr_ConnectedComp::go(_graph);

      iteration = 1;
      do{
        DGAccumulator_accum.reset();

      	struct Syncer_0 {
      		static unsigned int extract(uint32_t node_id, const struct NodeData & node) {
      		#ifdef __GALOIS_HET_CUDA__
      			if (personality == GPU_CUDA) return get_node_comp_current_cuda(cuda_ctx, node_id);
      			assert (personality == CPU);
      		#endif
      			return node.comp_current;
      		}
      		static void reduce (uint32_t node_id, struct NodeData & node, unsigned int y) {
      		#ifdef __GALOIS_HET_CUDA__
      			if (personality == GPU_CUDA) min_node_comp_current_cuda(cuda_ctx, node_id, y);
      			else if (personality == CPU)
      		#endif
      				{ Galois::atomicMin(node.comp_current, y);}
      		}
      		static void reset (uint32_t node_id, struct NodeData & node ) {
      		#ifdef __GALOIS_HET_CUDA__
      			if (personality == GPU_CUDA) set_node_comp_current_cuda(cuda_ctx, node_id, std::numeric_limits<unsigned int>::max()/4);
      			else if (personality == CPU)
      		#endif
      				{node.comp_current = std::numeric_limits<unsigned int>::max()/4; }
      		}
      		typedef unsigned int ValTy;
      	};
      #ifdef __GALOIS_HET_CUDA__
      	if (personality == GPU_CUDA) {
      		int __retval = 0;
      		ConnectedComp_cuda(__retval, cuda_ctx);
      		DGAccumulator_accum += __retval;
      	} else if (personality == CPU)
      #endif
      Galois::for_each(_graph.begin(), _graph.end(), ConnectedComp { &_graph }, Galois::loopname("cc"), Galois::write_set("sync_push", "this->graph", "struct NodeData &", "struct NodeData &" , "comp_current", "unsigned int" , "{ Galois::atomicMin(node.comp_current, y);}",  "{node.comp_current = std::numeric_limits<unsigned int>::max()/4; }"));
      _graph.sync_push<Syncer_0>();
      
      ++iteration;
      }while((iteration < maxIterations) && DGAccumulator_accum.reduce());
  }

  void operator()(GNode src, Galois::UserContext<GNode>& ctx) const {
    NodeData& snode = graph->getData(src);
    auto& sdist = snode.comp_current;

    if(snode.comp_old > snode.comp_current){
      snode.comp_old = snode.comp_current;
      DGAccumulator_accum += 1;
      for (auto jj = graph->edge_begin(src); jj != graph->edge_end(src); ++jj) {
        GNode dst = graph->getEdgeDst(jj);
        auto& dnode = graph->getData(dst);
        unsigned int new_dist = sdist;
        Galois::atomicMin(dnode.comp_current, new_dist);
      }
    }
  }
};
Galois::DGAccumulator<int>  ConnectedComp::DGAccumulator_accum;

int main(int argc, char** argv) {
  try {
    LonestarStart(argc, argv, name, desc, url);
    auto& net = Galois::Runtime::getSystemNetworkInterface();
    Galois::Timer T_total, T_hGraph_init, T_init, T_cc1, T_cc2, T_cc3;

#ifdef __GALOIS_HET_CUDA__
    const unsigned my_host_id = Galois::Runtime::getHostID();
    int gpu_device = gpudevice;
    std::vector<unsigned> scalefactor;
    //Parse arg string when running on multiple hosts and update/override personality
    //with corresponding value.
    if (personality_set.length() == Galois::Runtime::NetworkInterface::Num) {
      switch (personality_set.c_str()[my_host_id]) {
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
#ifdef __GALOIS_SINGLE_HOST_MULTIPLE_GPUS__
      if (gpu_device == -1) {
        gpu_device = 0;
        for (unsigned i = 0; i < my_host_id; ++i) {
          if (personality_set.c_str()[i] != 'c') ++gpu_device;
        }
      }
#endif
      for (unsigned i=0; i<personality_set.length(); ++i) {
        if (personality_set.c_str()[i] == 'c') 
          scalefactor.push_back(scalecpu);
        else
          scalefactor.push_back(scalegpu);
      }
    }
#endif

    T_total.start();

    T_hGraph_init.start();
#ifndef __GALOIS_HET_CUDA__
    Graph hg(inputFile, net.ID, net.Num);
#else
    Graph hg(inputFile, net.ID, net.Num, scalefactor);
    if (personality == GPU_CUDA) {
      cuda_ctx = get_CUDA_context(my_host_id);
      if (!init_CUDA_context(cuda_ctx, gpu_device))
        return -1;
      MarshalGraph m = hg.getMarshalGraph(my_host_id);
      load_graph_CUDA(cuda_ctx, m);
    } else if (personality == GPU_OPENCL) {
      //Galois::OpenCL::cl_env.init(cldevice.Value);
    }
#endif
    T_hGraph_init.stop();

    std::cout << "InitializeGraph::go called\n";
    T_init.start();
    InitializeGraph::go(hg);
    T_init.stop();

    std::cout << "ConnectedComp::go run1 called  on " << net.ID << "\n";
    T_cc1.start();
      ConnectedComp::go(hg);
    T_cc1.stop();

    Galois::Runtime::getHostBarrier().wait();
    InitializeGraph::go(hg);

    std::cout << "ConnectedComp::go run2 called  on " << net.ID << "\n";
    T_cc2.start();
      ConnectedComp::go(hg);
    T_cc2.stop();

    Galois::Runtime::getHostBarrier().wait();
    InitializeGraph::go(hg);

    std::cout << "ConnectedComp::go run3 called  on " << net.ID << "\n";
    T_cc3.start();
      ConnectedComp::go(hg);
    T_cc3.stop();

   T_total.stop();

    auto mean_time = (T_cc1.get() + T_cc2.get() + T_cc3.get())/3;

    std::cout << "[" << net.ID << "]" << " Total Time : " << T_total.get() << " hGraph : " << T_hGraph_init.get() << " Init : " << T_init.get() << " cc1 : " << T_cc1.get() << " cc2 : " << T_cc2.get() << " cc3 : " << T_cc3.get() <<" cc mean time (" << iteration << " iterations) : " << mean_time << "(msec)\n\n";

    // Verify
    if(verify){
#ifdef __GALOIS_HET_CUDA__
      if (personality == CPU) { 
#endif
        for(auto ii = hg.begin(); ii != hg.end(); ++ii) {
          Galois::Runtime::printOutput("% %\n", hg.getGID(*ii), hg.getData(*ii).comp_current);
        }
#ifdef __GALOIS_HET_CUDA__
      } else if(personality == GPU_CUDA)  {
        for(auto ii = hg.begin(); ii != hg.end(); ++ii) {
          Galois::Runtime::printOutput("% %\n", hg.getGID(*ii), get_node_comp_current_cuda(cuda_ctx, *ii));
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
