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

#ifdef __GALOIS_VERTEX_CUT_GRAPH__
#include "Galois/Runtime/vGraph.h"
#else
#include "Galois/Runtime/hGraph.h"
#endif
#include "Galois/DistAccumulator.h"

#ifdef __GALOIS_HET_CUDA__
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
#ifdef __GALOIS_VERTEX_CUT_GRAPH__
static cll::opt<std::string> partFolder("partFolder", cll::desc("path to partitionFolder"), cll::init(""));
#endif
static cll::opt<float> tolerance("tolerance", cll::desc("tolerance"), cll::init(0.0000001));
static cll::opt<unsigned int> maxIterations("maxIterations", cll::desc("Maximum iterations: Default 10000"), cll::init(10000));
static cll::opt<bool> verify("verify", cll::desc("Verify ranks by printing to 'page_ranks.#hid.csv' file"), cll::init(false));
#ifdef __GALOIS_SIMULATE_COMMUNICATION__
#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
static cll::opt<unsigned> comm_mode("comm_mode", cll::desc("Communication mode: 0 - original, 1 - simulated net, 2 - simulated bare MPI"), cll::init(0));
#endif
#endif
#ifdef __GALOIS_HET_CUDA__
static cll::opt<int> gpudevice("gpu", cll::desc("Select GPU to run on, default is to choose automatically"), cll::init(-1));
static cll::opt<Personality> personality("personality", cll::desc("Personality"),
      cll::values(clEnumValN(CPU, "cpu", "Galois CPU"), clEnumValN(GPU_CUDA, "gpu/cuda", "GPU/CUDA"), clEnumValN(GPU_OPENCL, "gpu/opencl", "GPU/OpenCL"), clEnumValEnd),
      cll::init(CPU));
static cll::opt<std::string> personality_set("pset", cll::desc("String specifying personality for each host. 'c'=CPU,'g'=GPU/CUDA and 'o'=GPU/OpenCL"), cll::init(""));
static cll::opt<unsigned> scalegpu("scalegpu", cll::desc("Scale GPU workload w.r.t. CPU, default is proportionally equal workload to CPU and GPU (1)"), cll::init(1));
static cll::opt<unsigned> scalecpu("scalecpu", cll::desc("Scale CPU workload w.r.t. GPU, default is proportionally equal workload to CPU and GPU (1)"), cll::init(1));
static cll::opt<bool> single_node("single_node", cll::desc("Single physical node with multiple devices: detect GPU to use for each host/process automatically"), cll::init(false));
#endif


static const float alpha = (1.0 - 0.85);
struct PR_NodeData {
  float value;
  std::atomic<float> residual;
  unsigned int nout;

};

#ifdef __GALOIS_VERTEX_CUT_GRAPH__
typedef vGraph<PR_NodeData, void> Graph;
#else
typedef hGraph<PR_NodeData, void> Graph;
#endif
typedef typename Graph::GraphNode GNode;

unsigned iteration;

struct ResetGraph {
  Graph* graph;

  ResetGraph(Graph* _graph) : graph(_graph){}
  void static go(Graph& _graph) {
    #ifdef __GALOIS_HET_CUDA__
    	if (personality == GPU_CUDA) {
    		ResetGraph_cuda(cuda_ctx);
    	} else if (personality == CPU)
    #endif
    Galois::do_all(_graph.begin(), _graph.end(), ResetGraph{ &_graph }, Galois::loopname("reset"), Galois::write_set("sync_pull", "this->graph", "struct PR_NodeData &", "struct PR_NodeData &", "residual" , "float"));
    
  }

  void operator()(GNode src) const {
    PR_NodeData& sdata = graph->getData(src);
    sdata.value = 0;
    sdata.nout = 0;
    sdata.residual = 0;
  }
};

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
        static bool extract_reset_batch(unsigned from_id, float *y) {
        #ifdef __GALOIS_HET_CUDA__
          if (personality == GPU_CUDA) {
            batch_get_reset_node_residual_cuda(cuda_ctx, from_id, y, 0);
            return true;
          }
          assert (personality == CPU);
        #endif
          return false;
        }
    		static void reduce (uint32_t node_id, struct PR_NodeData & node, float y) {
    		#ifdef __GALOIS_HET_CUDA__
    			if (personality == GPU_CUDA) add_node_residual_cuda(cuda_ctx, node_id, y);
    			else if (personality == CPU)
    		#endif
    				{ Galois::add(node.residual, y); }
    		}
        static bool reduce_batch(unsigned from_id, float *y) {
        #ifdef __GALOIS_HET_CUDA__
          if (personality == GPU_CUDA) {
            batch_add_node_residual_cuda(cuda_ctx, from_id, y);
            return true;
          } 
          assert (personality == CPU);
        #endif
            return false;
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
#ifdef __GALOIS_VERTEX_CUT_GRAPH__
    	struct SyncerPull_0 {
    		static float extract(uint32_t node_id, const struct PR_NodeData & node) {
    		#ifdef __GALOIS_HET_CUDA__
    			if (personality == GPU_CUDA) return get_node_residual_cuda(cuda_ctx, node_id);
    			assert (personality == CPU);
    		#endif
    			return node.residual;
    		}
        static bool extract_batch(unsigned from_id, float *y) {
        #ifdef __GALOIS_HET_CUDA__
          if (personality == GPU_CUDA) {
            batch_get_node_residual_cuda(cuda_ctx, from_id, y);
            return true;
          }
          assert (personality == CPU);
        #endif
          return false;
        }
    		static void setVal (uint32_t node_id, struct PR_NodeData & node, float y) {
    		#ifdef __GALOIS_HET_CUDA__
    			if (personality == GPU_CUDA) set_node_residual_cuda(cuda_ctx, node_id, y);
    			else if (personality == CPU)
    		#endif
    				node.residual = y;
    		}
        static bool setVal_batch(unsigned from_id, float *y) {
        #ifdef __GALOIS_HET_CUDA__
          if (personality == GPU_CUDA) {
            batch_set_node_residual_cuda(cuda_ctx, from_id, y);
            return true;
          } 
          assert (personality == CPU);
        #endif
            return false;
        }
    		typedef float ValTy;
    	};
#endif
    #ifdef __GALOIS_HET_CUDA__
    	if (personality == GPU_CUDA) {
    		InitializeGraph_cuda(alpha, cuda_ctx);
    	} else if (personality == CPU)
    #endif
    Galois::do_all(_graph.begin(), _graph.end(), InitializeGraph{ alpha, &_graph }, Galois::loopname("Init"), Galois::write_set("sync_push", "this->graph", "struct PR_NodeData &", "struct PR_NodeData &" , "residual", "float" , "add",  "0"), Galois::write_set("sync_pull", "this->graph", "struct PR_NodeData &", "struct PR_NodeData &", "residual" , "float"));
    _graph.sync_push<Syncer_0>("InitializeGraph");
#ifdef __GALOIS_VERTEX_CUT_GRAPH__
    _graph.sync_pull<SyncerPull_0>("InitializeGraph");
#endif
    
  }

  void operator()(GNode src) const {
    PR_NodeData& sdata = graph->getData(src);
    sdata.value = local_alpha;
    sdata.nout = std::distance(graph->edge_begin(src), graph->edge_end(src));

    if(sdata.nout > 0 ){
      float delta = sdata.value*(1-local_alpha)/sdata.nout;
      for(auto nbr = graph->edge_begin(src), ee = graph->edge_end(src); nbr != ee; ++nbr){
        GNode dst = graph->getEdgeDst(nbr);
        PR_NodeData& ddata = graph->getData(dst);
        Galois::atomicAdd(ddata.residual, delta);
      }
    }
  }
};


struct PageRank {
  const float &local_alpha;
  cll::opt<float> &local_tolerance;
  Graph* graph;

  PageRank(cll::opt<float> &_tolerance, const float &_alpha, Graph* _graph) : local_tolerance(_tolerance), local_alpha(_alpha), graph(_graph){}
  void static go(Graph& _graph) {
    iteration = 0;
    do{
      DGAccumulator_accum.reset();
      	struct Syncer_0 {
      		static float extract(uint32_t node_id, const struct PR_NodeData & node) {
      		#ifdef __GALOIS_HET_CUDA__
      			if (personality == GPU_CUDA) return get_node_residual_cuda(cuda_ctx, node_id);
      			assert (personality == CPU);
      		#endif
      			return node.residual;
      		}
          static bool extract_reset_batch(unsigned from_id, float *y) {
          #ifdef __GALOIS_HET_CUDA__
            if (personality == GPU_CUDA) {
              batch_get_reset_node_residual_cuda(cuda_ctx, from_id, y, 0);
              return true;
            }
            assert (personality == CPU);
          #endif
            return false;
          }
      		static void reduce (uint32_t node_id, struct PR_NodeData & node, float y) {
      		#ifdef __GALOIS_HET_CUDA__
      			if (personality == GPU_CUDA) add_node_residual_cuda(cuda_ctx, node_id, y);
      			else if (personality == CPU)
      		#endif
      				{ Galois::add(node.residual, y); }
      		}
          static bool reduce_batch(unsigned from_id, float *y) {
          #ifdef __GALOIS_HET_CUDA__
            if (personality == GPU_CUDA) {
              batch_add_node_residual_cuda(cuda_ctx, from_id, y);
              return true;
            } 
            assert (personality == CPU);
          #endif
              return false;
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
#ifdef __GALOIS_VERTEX_CUT_GRAPH__
        struct SyncerPull_0 {
          static float extract(uint32_t node_id, const struct PR_NodeData & node) {
          #ifdef __GALOIS_HET_CUDA__
            if (personality == GPU_CUDA) return get_node_residual_cuda(cuda_ctx, node_id);
            assert (personality == CPU);
          #endif
            return node.residual;
          }
          static bool extract_batch(unsigned from_id, float *y) {
          #ifdef __GALOIS_HET_CUDA__
            if (personality == GPU_CUDA) {
              batch_get_node_residual_cuda(cuda_ctx, from_id, y);
              return true;
            }
            assert (personality == CPU);
          #endif
            return false;
          }
          static void setVal (uint32_t node_id, struct PR_NodeData & node, float y) {
          #ifdef __GALOIS_HET_CUDA__
            if (personality == GPU_CUDA) set_node_residual_cuda(cuda_ctx, node_id, y);
            else if (personality == CPU)
          #endif
              {node.residual = y ; }
          }
          static bool setVal_batch(unsigned from_id, float *y) {
          #ifdef __GALOIS_HET_CUDA__
            if (personality == GPU_CUDA) {
              batch_set_node_residual_cuda(cuda_ctx, from_id, y);
              return true;
            } 
            assert (personality == CPU);
          #endif
              return false;
          }
          typedef float ValTy;
        };
#endif
      #ifdef __GALOIS_HET_CUDA__
      	if (personality == GPU_CUDA) {
      		int __retval = 0;
      		PageRank_cuda(__retval, alpha, tolerance, cuda_ctx);
      		DGAccumulator_accum += __retval;
      	} else if (personality == CPU)
      #endif
      Galois::do_all(_graph.begin(), _graph.end(), PageRank { tolerance, alpha, &_graph }, Galois::loopname("PageRank"), Galois::write_set("sync_push", "this->graph", "struct PR_NodeData &", "struct PR_NodeData &" , "residual", "float" , "add",  "0"));
      _graph.sync_push<Syncer_0>("PageRank");
#ifdef __GALOIS_VERTEX_CUT_GRAPH__
      _graph.sync_pull<SyncerPull_0>("PageRank");
#endif

     ++iteration; 
     if (maxIterations == 5) DGAccumulator_accum += 1;
    }while((iteration < maxIterations) && DGAccumulator_accum.reduce());
  }

  static Galois::DGAccumulator<int> DGAccumulator_accum;
  void operator()(GNode src)const {
    PR_NodeData& sdata = graph->getData(src);
    float residual_old = sdata.residual.exchange(0.0);
    sdata.value += residual_old;
    //sdata.residual = residual_old;
    if (sdata.nout > 0){
      float delta = residual_old*(1-local_alpha)/sdata.nout;
      for(auto nbr = graph->edge_begin(src), ee = graph->edge_end(src); nbr != ee; ++nbr){
        GNode dst = graph->getEdgeDst(nbr);
        PR_NodeData& ddata = graph->getData(dst);
        auto dst_residual_old = Galois::atomicAdd(ddata.residual, delta);
        if((dst_residual_old <= local_tolerance) && ((dst_residual_old + delta) >= local_tolerance)) {
          DGAccumulator_accum+= 1;
        }
      }
    }
  }
};
Galois::DGAccumulator<int>  PageRank::DGAccumulator_accum;

int main(int argc, char** argv) {
  try {

    LonestarStart(argc, argv, name, desc, url);
    Galois::Runtime::reportStat("(NULL)", "Max Iterations", (unsigned long)maxIterations, 0);
    std::ostringstream ss;
    ss << tolerance;
    Galois::Runtime::reportStat("(NULL)", "Tolerance", ss.str(), 0);
    Galois::StatManager statManager;
    auto& net = Galois::Runtime::getSystemNetworkInterface();
    Galois::StatTimer StatTimer_init("TIMER_GRAPH_INIT"), StatTimer_total("TIMER_TOTAL"), StatTimer_hg_init("TIMER_HG_INIT");

    StatTimer_total.start();

    std::vector<unsigned> scalefactor;
#ifdef __GALOIS_HET_CUDA__
    const unsigned my_host_id = Galois::Runtime::getHostID();
    int gpu_device = gpudevice;
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
      if (single_node && (gpu_device == -1)) {
        gpu_device = 0;
        for (unsigned i = 0; i < my_host_id; ++i) {
          if (personality_set.c_str()[i] != 'c') ++gpu_device;
        }
      }
      for (unsigned i=0; i<personality_set.length(); ++i) {
        if (personality_set.c_str()[i] == 'c') 
          scalefactor.push_back(scalecpu);
        else
          scalefactor.push_back(scalegpu);
      }
    }
#endif

    StatTimer_hg_init.start();
#ifdef __GALOIS_VERTEX_CUT_GRAPH__
    Graph hg(inputFile, partFolder, net.ID, net.Num, scalefactor);
#else
    Graph hg(inputFile, net.ID, net.Num, scalefactor);
#endif
#ifdef __GALOIS_SIMULATE_COMMUNICATION__
#ifdef __GALOIS_SIMULATE_COMMUNICATION_WITH_GRAPH_DATA__
    hg.set_comm_mode(comm_mode);
#endif
#endif
#ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      cuda_ctx = get_CUDA_context(my_host_id);
      if (!init_CUDA_context(cuda_ctx, gpu_device))
        return -1;
      MarshalGraph m = hg.getMarshalGraph(my_host_id);
      load_graph_CUDA(cuda_ctx, m, net.Num);
    } else if (personality == GPU_OPENCL) {
      //Galois::OpenCL::cl_env.init(cldevice.Value);
    }
#endif
    StatTimer_hg_init.stop();

    std::cout << "[" << net.ID << "] InitializeGraph::go called\n";
    StatTimer_init.start();
    InitializeGraph::go(hg);
    StatTimer_init.stop();


    for(auto run = 0; run < numRuns; ++run){
      std::cout << "[" << net.ID << "] PageRank::go run " << run << " called\n";
      std::string timer_str("TIMER_" + std::to_string(run));
      Galois::StatTimer StatTimer_main(timer_str.c_str());

      Galois::Runtime::getHostBarrier().wait();
      hg.reset_num_iter(run);

      Galois::Runtime::beginSampling();
      StatTimer_main.start();
    PageRank::go(hg);
      StatTimer_main.stop();
      Galois::Runtime::endSampling();

      if((run + 1) != numRuns){
        hg.reset_num_iter(run);
      ResetGraph::go(hg);
      InitializeGraph::go(hg);
    }
    }

   StatTimer_total.stop();

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
