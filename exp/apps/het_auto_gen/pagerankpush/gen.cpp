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
 */

#include <iostream>
#include <limits>
#include "Galois/Galois.h"
#include "Lonestar/BoilerPlate.h"
#include "Galois/gstl.h"

#include "Galois/Runtime/CompilerHelperFunctions.h"
#include "Galois/Runtime/Tracer.h"

#include "Galois/Dist/hGraph.h"

#ifdef __GALOIS_HET_CUDA__
#include "Galois/Cuda/cuda_mtypes.h"
#include "gen_cuda.h"
struct CUDA_Context *cuda_ctx;
#endif

static const char* const name = "PageRank - Compiler Generated Distributed Heterogeneous";
static const char* const desc = "Residual PageRank on Distributed Galois.";
static const char* const url = 0;

#ifdef __GALOIS_HET_CUDA__
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

namespace cll = llvm::cl;
static cll::opt<std::string> inputFile(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<unsigned int> maxIterations("maxIterations", cll::desc("Maximum iterations"), cll::init(1000));
static cll::opt<unsigned int> src_node("srcNodeId", cll::desc("ID of the source node"), cll::init(0));
static cll::opt<float> tolerance("tolerance", cll::desc("tolerance"), cll::init(0.01));
static cll::opt<bool> verify("verify", cll::desc("Verify ranks by printing to 'page_ranks.#hid.csv' file"), cll::init(false));
#ifdef __GALOIS_HET_CUDA__
static cll::opt<int> gpudevice("gpu", cll::desc("Select GPU to run on, default is to choose automatically"), cll::init(-1));
static cll::opt<Personality> personality("personality", cll::desc("Personality"),
      cll::values(clEnumValN(CPU, "cpu", "Galois CPU"), clEnumValN(GPU_CUDA, "gpu/cuda", "GPU/CUDA"), clEnumValN(GPU_OPENCL, "gpu/opencl", "GPU/OpenCL"), clEnumValEnd),
      cll::init(CPU));
static cll::opt<std::string> personality_set("pset", cll::desc("String specifying personality for each host. 'c'=CPU,'g'=GPU/CUDA and 'o'=GPU/OpenCL"), cll::init(""));
#endif


static const float alpha = (1.0 - 0.85);
struct PR_NodeData {
  float value;
  std::atomic<float> residual;
  unsigned int nout;

};

typedef hGraph<PR_NodeData, void> Graph;
typedef typename Graph::GraphNode GNode;

struct InitializeGraph {
  const float &local_alpha;
  Graph* graph;

  InitializeGraph(const float &_alpha, Graph* _graph) : local_alpha(_alpha), graph(_graph){}
  void static go(Graph& _graph) {
    	struct Syncer_0 {
    		static float extract(uint32_t node_id, const struct PR_NodeData & node) {
    		#ifdef __GALOIS_HET_CUDA__
    			if (personality == GPU_CUDA) return get_node_residual_cuda(cuda_ctx, node_id);
    			assert (personality == CPU);
    		#endif
    			return node.residual;
    		}
    		static void reduce (uint32_t node_id, struct PR_NodeData & node, float y) {
    		#ifdef __GALOIS_HET_CUDA__
    			if (personality == GPU_CUDA) add_node_residual_cuda(cuda_ctx, node_id, y);
    			else if (personality == CPU)
    		#endif
    				{ Galois::atomicAdd(node.residual, y);}
    		}
    		static void reset (uint32_t node_id, struct PR_NodeData & node ) {
    		#ifdef __GALOIS_HET_CUDA__
    			if (personality == GPU_CUDA) set_node_residual_cuda(cuda_ctx, node_id, 0);
    			else if (personality == CPU)
    		#endif
    				{node.residual = 0 ; }
    		}
    		typedef float ValTy;
    	};
    #ifdef __GALOIS_HET_CUDA__
    	if (personality == GPU_CUDA) {
    		InitializeGraph_cuda(alpha, cuda_ctx);
    	} else if (personality == CPU)
    #endif
    Galois::do_all(_graph.begin(), _graph.end(), InitializeGraph{ alpha, &_graph }, Galois::loopname("Init"), Galois::write_set("sync_push", "this->graph", "struct PR_NodeData &", "struct PR_NodeData &" , "residual", "float" , "{ Galois::atomicAdd(node.residual, y);}",  "{node.residual = 0 ; }"));
    _graph.sync_push<Syncer_0>();
    
  }

  void operator()(GNode src) const {
    PR_NodeData& sdata = graph->getData(src);
    sdata.value = 1.0 - local_alpha;
    sdata.nout = std::distance(graph->edge_begin(src), graph->edge_end(src));

    if(sdata.nout > 0 ){
      float delta = sdata.value*local_alpha/sdata.nout;
      for(auto nbr = graph->edge_begin(src); nbr != graph->edge_end(src); ++nbr){
        GNode dst = graph->getEdgeDst(nbr);
        PR_NodeData& ddata = graph->getData(dst);
        Galois::atomicAdd(ddata.residual, delta);
      }
    }
  }
};


static uint32_t num_Hosts_recvd = 0;
static bool didWork = 0;
static std::vector<bool> others_didWork_vec;

static void didWork_landingPad(Galois::Runtime::RecvBuffer& buf){
  //receive didWork from all and decide.
  uint32_t x_id;
  bool x_didWork;

  gDeserialize(buf, x_id, x_didWork);
  ++num_Hosts_recvd;
  others_didWork_vec.push_back(x_didWork);
}

struct PageRank {
  _Bool &local_didWork;
  const float &local_alpha;
  cll::opt<float> &local_tolerance;
  Graph* graph;

  PageRank(cll::opt<float> &_tolerance, const float &_alpha, _Bool &_didWork, Graph* _graph) : local_tolerance(_tolerance), local_alpha(_alpha), local_didWork(_didWork), graph(_graph){}
  void static go(Graph& _graph) {
    	struct Syncer_0 {
    		static float extract(uint32_t node_id, const struct PR_NodeData & node) {
    		#ifdef __GALOIS_HET_CUDA__
    			if (personality == GPU_CUDA) return get_node_residual_cuda(cuda_ctx, node_id);
    			assert (personality == CPU);
    		#endif
    			return node.residual;
    		}
    		static void reduce (uint32_t node_id, struct PR_NodeData & node, float y) {
    		#ifdef __GALOIS_HET_CUDA__
    			if (personality == GPU_CUDA) add_node_residual_cuda(cuda_ctx, node_id, y);
    			else if (personality == CPU)
    		#endif
    				{ Galois::atomicAdd(node.residual, y);}
    		}
    		static void reset (uint32_t node_id, struct PR_NodeData & node ) {
    		#ifdef __GALOIS_HET_CUDA__
    			if (personality == GPU_CUDA) set_node_residual_cuda(cuda_ctx, node_id, 0);
    			else if (personality == CPU)
    		#endif
    				{node.residual = 0 ; }
    		}
    		typedef float ValTy;
    	};
    #ifdef __GALOIS_HET_CUDA__
    	if (personality == GPU_CUDA) {
    		PageRank_cuda(didWork, alpha, tolerance, cuda_ctx);
    	} else if (personality == CPU)
    #endif
    Galois::do_all(_graph.begin(), _graph.end(), PageRank { tolerance, alpha, didWork, &_graph }, Galois::write_set("sync_push", "this->graph", "struct PR_NodeData &", "struct PR_NodeData &" , "residual", "float" , "{ Galois::atomicAdd(node.residual, y);}",  "{node.residual = 0 ; }"));
    _graph.sync_push<Syncer_0>();
    
  }

  void operator()(GNode src)const {
    PR_NodeData& sdata = graph->getData(src);
    float residual_old = sdata.residual.exchange(0.0);
    sdata.value += residual_old;
    //sdata.residual = residual_old;
    if (sdata.nout > 0){
      float delta = residual_old*local_alpha/sdata.nout;
      for(auto nbr = graph->edge_begin(src); nbr != graph->edge_end(src); ++nbr){
        GNode dst = graph->getEdgeDst(nbr);
        PR_NodeData& ddata = graph->getData(dst);
        auto dst_residual_old = Galois::atomicAdd(ddata.residual, delta);
        if((dst_residual_old <= local_tolerance) && ((dst_residual_old + delta) >= local_tolerance)) {
          local_didWork += 1;
        }
      }
    }
  }
};

int main(int argc, char** argv) {
  try {

    LonestarStart(argc, argv, name, desc, url);
    auto& net = Galois::Runtime::getSystemNetworkInterface();
    Galois::Timer T_total, T_offlineGraph_init, T_hGraph_init, T_init, T_pageRank;

#ifdef __GALOIS_HET_CUDA__
    const unsigned my_host_id = Galois::Runtime::getHostID();
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
    }
#endif

    T_total.start();

    T_offlineGraph_init.start();
    OfflineGraph g(inputFile);
    T_offlineGraph_init.stop();
    std::cout << g.size() << " " << g.sizeEdges() << "\n";

    T_hGraph_init.start();
    Graph hg(inputFile, net.ID, net.Num);
#ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      cuda_ctx = get_CUDA_context(my_host_id);
      if (!init_CUDA_context(cuda_ctx, gpudevice))
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

    // Verify
    /*if(verify){
#ifdef __GALOIS_HET_CUDA__
      if (personality == CPU) { 
#endif
        for(auto ii = hg.begin(); ii != hg.end(); ++ii) {
          Galois::Runtime::printOutput("% %\n", hg.getGID(*ii), hg.getData(*ii).nout);
        }
#ifdef __GALOIS_HET_CUDA__
      } else if(personality == GPU_CUDA)  {
        for(auto ii = hg.begin(); ii != hg.end(); ++ii) {
          Galois::Runtime::printOutput("% %\n", hg.getGID(*ii), get_node_nout_cuda(cuda_ctx, *ii));
        }
      }
#endif
    }*/

    std::cout << "PageRank::go called  on " << net.ID << "\n";
    T_pageRank.start();
    for (int i = 0; i < maxIterations; ++i) {
      std::cout << " Iteration : " << i << "\n";
      PageRank::go(hg);

      //broadcast
      //net.broadcast(didWork_landingPad, b);
      for(auto x = 0; x < net.Num; ++x){
        if(x == net.ID)
          continue;
        //Exchange didWorks to decide if done.
        Galois::Runtime::SendBuffer b;
        gSerialize(b, net.ID, didWork);
        net.send(x,didWork_landingPad, b);
      }

      net.flush();
      while(num_Hosts_recvd < (net.Num - 1)){
        net.handleReceives();
      }

      assert(others_didWork_vec.size() == (net.Num -1));
      bool CanTerminate = !didWork;
      for(auto x : others_didWork_vec){
        CanTerminate = (CanTerminate && !x);
        x = false;
      }

      if(CanTerminate){
        break;
      }

      didWork = false;
      num_Hosts_recvd = 0;
      others_didWork_vec.clear();
    }
    T_pageRank.stop();

    T_total.stop();

    std::cout << "[" << net.ID << "]" << " Total Time : " << T_total.get() << " offlineGraph : " << T_offlineGraph_init.get() << " hGraph : " << T_hGraph_init.get() << " Init : " << T_init.get() << " PageRank (" << maxIterations << ") : " << T_pageRank.get() << "(msec)\n\n";

    // Verify
    if(verify){
#ifdef __GALOIS_HET_CUDA__
      if (personality == CPU) { 
#endif
        for(auto ii = hg.begin(); ii != hg.end(); ++ii) {
          Galois::Runtime::printOutput("% %\n", hg.getGID(*ii), hg.getData(*ii).value);
        }
#ifdef __GALOIS_HET_CUDA__
      } else if(personality == GPU_CUDA)  {
        for(auto ii = hg.begin(); ii != hg.end(); ++ii) {
          Galois::Runtime::printOutput("% %\n", hg.getGID(*ii), get_node_value_cuda(cuda_ctx, *ii));
        }
      }
#endif
    }

    return 0;
  } catch (const char* c) {
      std::cerr << "Error: " << c << "\n";
      return 1;
  }
}
