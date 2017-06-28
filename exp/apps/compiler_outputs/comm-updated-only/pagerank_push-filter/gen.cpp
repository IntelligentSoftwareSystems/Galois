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
#include <algorithm>
#include <vector>
#include "Galois/Galois.h"
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

static cll::opt<unsigned int> numPipelinedPhases("numPipelinedPhases", cll::desc("num of pipelined phases to overlap computation and communication"), cll::init(1));
static cll::opt<unsigned int> numComputeSubsteps("numComputeSubsteps", cll::desc("num of sub steps of computations within a BSP phase"), cll::init(1));

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


static const float alpha = (1.0 - 0.85);
struct PR_NodeData {
  float value;
  float delta;
  std::atomic<float> residual;
  std::atomic<unsigned int> nout;

};

Galois::DynamicBitSet bitset_residual;

typedef hGraph<PR_NodeData, void> Graph;
typedef hGraph_edgeCut<PR_NodeData, void> Graph_edgeCut;
typedef hGraph_vertexCut<PR_NodeData, void> Graph_vertexCut;
typedef hGraph_cartesianCut<PR_NodeData, void> Graph_cartesianCut;

typedef typename Graph::GraphNode GNode;

typedef GNode WorkItem;

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
    Galois::do_all(_graph.begin(), _graph.end(), ResetGraph{ &_graph }, Galois::loopname("ResetGraph"), Galois::numrun(_graph.get_run_identifier()));
  }

  void operator()(GNode src) const {
    PR_NodeData& sdata = graph->getData(src);
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
      	struct Reduce_0 {
      		static unsigned int extract(uint32_t node_id, const struct PR_NodeData & node) {
      		#ifdef __GALOIS_HET_CUDA__
      			if (personality == GPU_CUDA) return get_node_nout_cuda(cuda_ctx, node_id);
      			assert (personality == CPU);
      		#endif
      			return node.nout;
      		}
      		static bool extract_reset_batch(unsigned from_id, unsigned long long int *b, unsigned int *o, unsigned int *y, size_t *s, DataCommMode *data_mode) {
      		#ifdef __GALOIS_HET_CUDA__
      			if (personality == GPU_CUDA) { batch_get_reset_node_nout_cuda(cuda_ctx, from_id, b, o, y, s, data_mode, 0); return true; }
      			assert (personality == CPU);
      		#endif
      			return false;
      		}
      		static bool extract_reset_batch(unsigned from_id, unsigned int *y) {
      		#ifdef __GALOIS_HET_CUDA__
      			if (personality == GPU_CUDA) { batch_get_reset_node_nout_cuda(cuda_ctx, from_id, y, 0); return true; }
      			assert (personality == CPU);
      		#endif
      			return false;
      		}
      		static bool reduce (uint32_t node_id, struct PR_NodeData & node, unsigned int y) {
      		#ifdef __GALOIS_HET_CUDA__
      			if (personality == GPU_CUDA) add_node_nout_cuda(cuda_ctx, node_id, y);
      			else if (personality == CPU)
      		#endif
      				{ Galois::add(node.nout, y); }
            return true;
      		}
      		static bool reduce_batch(unsigned from_id, unsigned long long int *b, unsigned int *o, unsigned int *y, size_t s, DataCommMode data_mode) {
      		#ifdef __GALOIS_HET_CUDA__
      			if (personality == GPU_CUDA) { batch_add_node_nout_cuda(cuda_ctx, from_id, b, o, y, s, data_mode); return true; }
      			assert (personality == CPU);
      		#endif
      			return false;
      		}
      		static void reset (uint32_t node_id, struct PR_NodeData & node ) {
      		#ifdef __GALOIS_HET_CUDA__
      			if (personality == GPU_CUDA) set_node_nout_cuda(cuda_ctx, node_id, 0);
      			else if (personality == CPU)
      		#endif
      				{ node.nout = 0; }
      		}
      		typedef unsigned int ValTy;
      	};
      	struct Broadcast_0 {
      		static unsigned int extract(uint32_t node_id, const struct PR_NodeData & node) {
      		#ifdef __GALOIS_HET_CUDA__
      			if (personality == GPU_CUDA) return get_node_nout_cuda(cuda_ctx, node_id);
      			assert (personality == CPU);
      		#endif
      			return node.nout;
      		}
      		static bool extract_batch(unsigned from_id, unsigned long long int *b, unsigned int *o, unsigned int *y, size_t *s, DataCommMode *data_mode) {
      		#ifdef __GALOIS_HET_CUDA__
      			if (personality == GPU_CUDA) { batch_get_node_nout_cuda(cuda_ctx, from_id, b, o, y, s, data_mode); return true; }
      			assert (personality == CPU);
      		#endif
      			return false;
      		}
      		static bool extract_batch(unsigned from_id, unsigned int *y) {
      		#ifdef __GALOIS_HET_CUDA__
      			if (personality == GPU_CUDA) { batch_get_node_nout_cuda(cuda_ctx, from_id, y); return true; }
      			assert (personality == CPU);
      		#endif
      			return false;
      		}
      		static void setVal (uint32_t node_id, struct PR_NodeData & node, unsigned int y) {
      		#ifdef __GALOIS_HET_CUDA__
      			if (personality == GPU_CUDA) set_node_nout_cuda(cuda_ctx, node_id, y);
      			else if (personality == CPU)
      		#endif
      				node.nout = y;
      		}
      		static bool setVal_batch(unsigned from_id, unsigned long long int *b, unsigned int *o, unsigned int *y, size_t s, DataCommMode data_mode) {
      		#ifdef __GALOIS_HET_CUDA__
      			if (personality == GPU_CUDA) { batch_set_mirror_node_nout_cuda(cuda_ctx, from_id, b, o, y, s, data_mode); return true; }
      			assert (personality == CPU);
      		#endif
      			return false;
      		}
      		typedef unsigned int ValTy;
      	};
      #ifdef __GALOIS_HET_CUDA__
      	if (personality == GPU_CUDA) {
          bitset_nout_clear_cuda(cuda_ctx);
      		std::string impl_str("CUDA_DO_ALL_IMPL_InitializeGraphNout_" + (_graph.get_run_identifier()));
      		Galois::StatTimer StatTimer_cuda(impl_str.c_str());
      		StatTimer_cuda.start();
      		InitializeGraphNout_all_cuda(cuda_ctx);
      		StatTimer_cuda.stop();
      	} else if (personality == CPU)
      #endif
      {
      bitset_residual.clear();
      Galois::do_all(_graph.begin(), _graph.end(), InitializeGraphNout{ &_graph }, Galois::loopname("InitializeGraphNout"), Galois::numrun(_graph.get_run_identifier()), Galois::write_set("reduce", "this->graph", "struct PR_NodeData &", "struct PR_NodeData &" , "nout", "float" , "add",  "0"));
      }
      _graph.sync_exchange<Reduce_0, Broadcast_0>("InitializeGraphNout", bitset_residual);
      
  }

  void operator()(GNode src) const {
    PR_NodeData& sdata = graph->getData(src);
    Galois::atomicAdd(sdata.nout, (unsigned int) std::distance(graph->edge_begin(src), graph->edge_end(src)));
    bitset_residual.set(src);
  }
};

struct InitializeGraph {
  const float &local_alpha;
  Graph* graph;

  InitializeGraph(const float &_alpha, Graph* _graph) : local_alpha(_alpha), graph(_graph){}
  void static go(Graph& _graph) {
      	struct Reduce_0 {
      		static float extract(uint32_t node_id, const struct PR_NodeData & node) {
      		#ifdef __GALOIS_HET_CUDA__
      			if (personality == GPU_CUDA) return get_node_residual_cuda(cuda_ctx, node_id);
      			assert (personality == CPU);
      		#endif
      			return node.residual;
      		}
      		static bool extract_reset_batch(unsigned from_id, unsigned long long int *b, unsigned int *o, float *y, size_t *s, DataCommMode *data_mode) {
      		#ifdef __GALOIS_HET_CUDA__
      			if (personality == GPU_CUDA) { batch_get_reset_node_residual_cuda(cuda_ctx, from_id, b, o, y, s, data_mode, 0); return true; }
      			assert (personality == CPU);
      		#endif
      			return false;
      		}
      		static bool extract_reset_batch(unsigned from_id, float *y) {
      		#ifdef __GALOIS_HET_CUDA__
      			if (personality == GPU_CUDA) { batch_get_reset_node_residual_cuda(cuda_ctx, from_id, y, 0); return true; }
      			assert (personality == CPU);
      		#endif
      			return false;
      		}
      		static bool reduce (uint32_t node_id, struct PR_NodeData & node, float y) {
      		#ifdef __GALOIS_HET_CUDA__
      			if (personality == GPU_CUDA) add_node_residual_cuda(cuda_ctx, node_id, y);
      			else if (personality == CPU)
      		#endif
      				{ Galois::add(node.residual, y); }
            return true;
      		}
      		static bool reduce_batch(unsigned from_id, unsigned long long int *b, unsigned int *o, float *y, size_t s, DataCommMode data_mode) {
      		#ifdef __GALOIS_HET_CUDA__
      			if (personality == GPU_CUDA) { batch_add_node_residual_cuda(cuda_ctx, from_id, b, o, y, s, data_mode); return true; }
      			assert (personality == CPU);
      		#endif
      			return false;
      		}
      		static void reset (uint32_t node_id, struct PR_NodeData & node ) {
      		#ifdef __GALOIS_HET_CUDA__
      			if (personality == GPU_CUDA) set_node_residual_cuda(cuda_ctx, node_id, 0);
      			else if (personality == CPU)
      		#endif
      				{ node.residual = 0; }
      		}
      		typedef float ValTy;
      	};
      	struct Broadcast_0 {
      		static float extract(uint32_t node_id, const struct PR_NodeData & node) {
      		#ifdef __GALOIS_HET_CUDA__
      			if (personality == GPU_CUDA) return get_node_residual_cuda(cuda_ctx, node_id);
      			assert (personality == CPU);
      		#endif
      			return node.residual;
      		}
      		static bool extract_batch(unsigned from_id, unsigned long long int *b, unsigned int *o, float *y, size_t *s, DataCommMode *data_mode) {
      		#ifdef __GALOIS_HET_CUDA__
      			if (personality == GPU_CUDA) { batch_get_node_residual_cuda(cuda_ctx, from_id, b, o, y, s, data_mode); return true; }
      			assert (personality == CPU);
      		#endif
      			return false;
      		}
      		static bool extract_batch(unsigned from_id, float *y) {
      		#ifdef __GALOIS_HET_CUDA__
      			if (personality == GPU_CUDA) { batch_get_node_residual_cuda(cuda_ctx, from_id, y); return true; }
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
      		static bool setVal_batch(unsigned from_id, unsigned long long int *b, unsigned int *o, float *y, size_t s, DataCommMode data_mode) {
      		#ifdef __GALOIS_HET_CUDA__
      			if (personality == GPU_CUDA) { batch_set_mirror_node_residual_cuda(cuda_ctx, from_id, b, o, y, s, data_mode); return true; }
      			assert (personality == CPU);
      		#endif
      			return false;
      		}
      		typedef float ValTy;
      	};
      #ifdef __GALOIS_HET_CUDA__
      	if (personality == GPU_CUDA) {
          bitset_residual_clear_cuda(cuda_ctx);
      		std::string impl_str("CUDA_DO_ALL_IMPL_InitializeGraph_" + (_graph.get_run_identifier()));
      		Galois::StatTimer StatTimer_cuda(impl_str.c_str());
      		StatTimer_cuda.start();
      		InitializeGraph_all_cuda(alpha, cuda_ctx);
      		StatTimer_cuda.stop();
      	} else if (personality == CPU)
      #endif
      {
      bitset_residual.clear();
      Galois::do_all(_graph.begin(), _graph.end(), InitializeGraph{ alpha, &_graph }, Galois::loopname("InitializeGraph"), Galois::numrun(_graph.get_run_identifier()), Galois::write_set("reduce", "this->graph", "struct PR_NodeData &", "struct PR_NodeData &" , "residual", "float" , "add",  "0"));
      }
      _graph.sync_forward<Reduce_0, Broadcast_0>("InitializeGraph", bitset_residual);
      
  }

  void operator()(GNode src) const {
    PR_NodeData& sdata = graph->getData(src);
    sdata.value = local_alpha;

    if(sdata.nout > 0 ){
      float delta = sdata.value*(1-local_alpha)/sdata.nout;
      for(auto nbr = graph->edge_begin(src), ee = graph->edge_end(src); nbr != ee; ++nbr){
        GNode dst = graph->getEdgeDst(nbr);
        PR_NodeData& ddata = graph->getData(dst);
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

  PageRankCopy(const float & _local_alpha,cll::opt<float> & _local_tolerance,Graph * _graph):local_alpha(_local_alpha),local_tolerance(_local_tolerance),graph(_graph){}
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
      Galois::do_all(_graph.begin(), _graph.end(), PageRankCopy{ alpha, tolerance, &_graph }, Galois::loopname("PageRank"), Galois::numrun(_graph.get_run_identifier()));
    }
  }

  void operator()(WorkItem src) const {
    PR_NodeData& sdata = graph->getData(src);
    if (sdata.residual > this->local_tolerance){
      float residual_old = sdata.residual;
      sdata.residual = 0;
      sdata.value += residual_old;
      if (sdata.nout > 0){
        sdata.delta = residual_old*(1-local_alpha)/sdata.nout;
      }
    }
  }
};

struct FirstItr_PageRank{
Graph * graph;
FirstItr_PageRank(Graph * _graph):graph(_graph){}
void static go(Graph& _graph) {
	struct Reduce_0 {
		static float extract(uint32_t node_id, const struct PR_NodeData & node) {
		#ifdef __GALOIS_HET_CUDA__
			if (personality == GPU_CUDA) return get_node_residual_cuda(cuda_ctx, node_id);
			assert (personality == CPU);
		#endif
			return node.residual;
		}
		static bool extract_reset_batch(unsigned from_id, unsigned long long int *b, unsigned int *o, float *y, size_t *s, DataCommMode *data_mode) {
		#ifdef __GALOIS_HET_CUDA__
			if (personality == GPU_CUDA) { batch_get_reset_node_residual_cuda(cuda_ctx, from_id, b, o, y, s, data_mode, 0); return true; }
			assert (personality == CPU);
		#endif
			return false;
		}
		static bool extract_reset_batch(unsigned from_id, float *y) {
		#ifdef __GALOIS_HET_CUDA__
			if (personality == GPU_CUDA) { batch_get_reset_node_residual_cuda(cuda_ctx, from_id, y, 0); return true; }
			assert (personality == CPU);
		#endif
			return false;
		}
		static bool reduce (uint32_t node_id, struct PR_NodeData & node, float y) {
		#ifdef __GALOIS_HET_CUDA__
			if (personality == GPU_CUDA) add_node_residual_cuda(cuda_ctx, node_id, y);
			else if (personality == CPU)
		#endif
				{ Galois::add(node.residual, y); }
      return true;
		}
		static bool reduce_batch(unsigned from_id, unsigned long long int *b, unsigned int *o, float *y, size_t s, DataCommMode data_mode) {
		#ifdef __GALOIS_HET_CUDA__
			if (personality == GPU_CUDA) { batch_add_node_residual_cuda(cuda_ctx, from_id, b, o, y, s, data_mode); return true; }
			assert (personality == CPU);
		#endif
			return false;
		}
		static void reset (uint32_t node_id, struct PR_NodeData & node ) {
		#ifdef __GALOIS_HET_CUDA__
			if (personality == GPU_CUDA) set_node_residual_cuda(cuda_ctx, node_id, 0);
			else if (personality == CPU)
		#endif
				{ node.residual = 0; }
		}
		typedef float ValTy;
	};
	struct Broadcast_0 {
		static float extract(uint32_t node_id, const struct PR_NodeData & node) {
		#ifdef __GALOIS_HET_CUDA__
			if (personality == GPU_CUDA) return get_node_residual_cuda(cuda_ctx, node_id);
			assert (personality == CPU);
		#endif
			return node.residual;
		}
		static bool extract_batch(unsigned from_id, unsigned long long int *b, unsigned int *o, float *y, size_t *s, DataCommMode *data_mode) {
		#ifdef __GALOIS_HET_CUDA__
			if (personality == GPU_CUDA) { batch_get_node_residual_cuda(cuda_ctx, from_id, b, o, y, s, data_mode); return true; }
			assert (personality == CPU);
		#endif
			return false;
		}
		static bool extract_batch(unsigned from_id, float *y) {
		#ifdef __GALOIS_HET_CUDA__
			if (personality == GPU_CUDA) { batch_get_node_residual_cuda(cuda_ctx, from_id, y); return true; }
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
		static bool setVal_batch(unsigned from_id, unsigned long long int *b, unsigned int *o, float *y, size_t s, DataCommMode data_mode) {
		#ifdef __GALOIS_HET_CUDA__
			if (personality == GPU_CUDA) { batch_set_mirror_node_residual_cuda(cuda_ctx, from_id, b, o, y, s, data_mode); return true; }
			assert (personality == CPU);
		#endif
			return false;
		}
		typedef float ValTy;
	};
#ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      bitset_residual_clear_cuda(cuda_ctx);
      std::string impl_str("CUDA_DO_ALL_IMPL_PageRank_" + (_graph.get_run_identifier()));
      Galois::StatTimer StatTimer_cuda(impl_str.c_str());
      StatTimer_cuda.start();
      FirstItr_PageRank_all_cuda(cuda_ctx);
      StatTimer_cuda.stop();
    } else if (personality == CPU)
#endif
    {
      bitset_residual.clear();
      Galois::do_all(_graph.begin(), _graph.end(), FirstItr_PageRank{&_graph}, Galois::loopname("PageRank"), Galois::numrun(_graph.get_run_identifier()), Galois::write_set("reduce", "this->graph", "struct PR_NodeData &", "struct PR_NodeData &" , "residual", "float" , "add",  "0"));
    }
_graph.sync_forward<Reduce_0, Broadcast_0>("PageRank", bitset_residual);

Galois::Runtime::reportStat("(NULL)", "NUM_WORK_ITEMS_" + (_graph.get_run_identifier()), _graph.end() - _graph.begin(), 0);

}
void operator()(WorkItem src) const {
    PR_NodeData& sdata = graph->getData(src);
    if (sdata.delta > 0) {
      float delta = sdata.delta;
      sdata.delta = 0;
      for(auto nbr = graph->edge_begin(src), ee = graph->edge_end(src); nbr != ee; ++nbr){
        GNode dst = graph->getEdgeDst(nbr);
        PR_NodeData& ddata = graph->getData(dst);
        Galois::atomicAdd(ddata.residual, delta);
        bitset_residual.set(dst);
      }
    }
  }

};
struct PageRank {
  Graph* graph;

  PageRank(Graph* _g): graph(_g){}
  void static go(Graph& _graph) {
    
    PageRankCopy::go(_graph);
    FirstItr_PageRank::go(_graph);
    
    unsigned _num_iterations = 1;
    
    do { 
     _graph.set_num_iter(_num_iterations);
    PageRankCopy::go(_graph);
    DGAccumulator_accum.reset();
    	struct Reduce_0 {
    		static float extract(uint32_t node_id, const struct PR_NodeData & node) {
    		#ifdef __GALOIS_HET_CUDA__
    			if (personality == GPU_CUDA) return get_node_residual_cuda(cuda_ctx, node_id);
    			assert (personality == CPU);
    		#endif
    			return node.residual;
    		}
    		static bool extract_reset_batch(unsigned from_id, unsigned long long int *b, unsigned int *o, float *y, size_t *s, DataCommMode *data_mode) {
    		#ifdef __GALOIS_HET_CUDA__
    			if (personality == GPU_CUDA) { batch_get_reset_node_residual_cuda(cuda_ctx, from_id, b, o, y, s, data_mode, 0); return true; }
    			assert (personality == CPU);
    		#endif
    			return false;
    		}
    		static bool extract_reset_batch(unsigned from_id, float *y) {
    		#ifdef __GALOIS_HET_CUDA__
    			if (personality == GPU_CUDA) { batch_get_reset_node_residual_cuda(cuda_ctx, from_id, y, 0); return true; }
    			assert (personality == CPU);
    		#endif
    			return false;
    		}
    		static bool reduce (uint32_t node_id, struct PR_NodeData & node, float y) {
    		#ifdef __GALOIS_HET_CUDA__
    			if (personality == GPU_CUDA) add_node_residual_cuda(cuda_ctx, node_id, y);
    			else if (personality == CPU)
    		#endif
    				{ Galois::add(node.residual, y); }
          return true;
    		}
    		static bool reduce_batch(unsigned from_id, unsigned long long int *b, unsigned int *o, float *y, size_t s, DataCommMode data_mode) {
    		#ifdef __GALOIS_HET_CUDA__
    			if (personality == GPU_CUDA) { batch_add_node_residual_cuda(cuda_ctx, from_id, b, o, y, s, data_mode); return true; }
    			assert (personality == CPU);
    		#endif
    			return false;
    		}
    		static void reset (uint32_t node_id, struct PR_NodeData & node ) {
    		#ifdef __GALOIS_HET_CUDA__
    			if (personality == GPU_CUDA) set_node_residual_cuda(cuda_ctx, node_id, 0);
    			else if (personality == CPU)
    		#endif
    				{ node.residual = 0; }
    		}
    		typedef float ValTy;
    	};
    	struct Broadcast_0 {
    		static float extract(uint32_t node_id, const struct PR_NodeData & node) {
    		#ifdef __GALOIS_HET_CUDA__
    			if (personality == GPU_CUDA) return get_node_residual_cuda(cuda_ctx, node_id);
    			assert (personality == CPU);
    		#endif
    			return node.residual;
    		}
    		static bool extract_batch(unsigned from_id, unsigned long long int *b, unsigned int *o, float *y, size_t *s, DataCommMode *data_mode) {
    		#ifdef __GALOIS_HET_CUDA__
    			if (personality == GPU_CUDA) { batch_get_node_residual_cuda(cuda_ctx, from_id, b, o, y, s, data_mode); return true; }
    			assert (personality == CPU);
    		#endif
    			return false;
    		}
    		static bool extract_batch(unsigned from_id, float *y) {
    		#ifdef __GALOIS_HET_CUDA__
    			if (personality == GPU_CUDA) { batch_get_node_residual_cuda(cuda_ctx, from_id, y); return true; }
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
    		static bool setVal_batch(unsigned from_id, unsigned long long int *b, unsigned int *o, float *y, size_t s, DataCommMode data_mode) {
    		#ifdef __GALOIS_HET_CUDA__
    			if (personality == GPU_CUDA) { batch_set_mirror_node_residual_cuda(cuda_ctx, from_id, b, o, y, s, data_mode); return true; }
    			assert (personality == CPU);
    		#endif
    			return false;
    		}
    		typedef float ValTy;
    	};
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          bitset_residual_clear_cuda(cuda_ctx);
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
          bitset_residual.clear();
          Galois::do_all(_graph.begin(), _graph.end(), PageRank{ &_graph }, Galois::loopname("PageRank"), Galois::write_set("reduce", "this->graph", "struct PR_NodeData &", "struct PR_NodeData &" , "residual", "float" , "add",  "0"), Galois::numrun(_graph.get_run_identifier()));
        }
    _graph.sync_forward<Reduce_0, Broadcast_0>("PageRank", bitset_residual);
    
    Galois::Runtime::reportStat("(NULL)", "NUM_WORK_ITEMS_" + (_graph.get_run_identifier()), (unsigned long)DGAccumulator_accum.read_local(), 0);
    ++_num_iterations;
    }while((_num_iterations < maxIterations) && DGAccumulator_accum.reduce());
    if (Galois::Runtime::getSystemNetworkInterface().ID == 0) {
      Galois::Runtime::reportStat("(NULL)", "NUM_ITERATIONS_" + std::to_string(_graph.get_run_num()), (unsigned long)_num_iterations, 0);
    }
    
  }

  static Galois::DGAccumulator<int> DGAccumulator_accum;
  void operator()(WorkItem src) const {
    PR_NodeData& sdata = graph->getData(src);

    if(sdata.delta > 0){
      float delta = sdata.delta;
      sdata.delta = 0;
      for(auto nbr = graph->edge_begin(src), ee = graph->edge_end(src); nbr != ee; ++nbr){
        GNode dst = graph->getEdgeDst(nbr);
        PR_NodeData& ddata = graph->getData(dst);
        Galois::atomicAdd(ddata.residual, delta);
        bitset_residual.set(dst);
      }
      DGAccumulator_accum+= 1;
    }
  }
};
Galois::DGAccumulator<int>  PageRank::DGAccumulator_accum;


int main(int argc, char** argv) {
  try {

    LonestarStart(argc, argv, name, desc, url);
    auto& net = Galois::Runtime::getSystemNetworkInterface();
    Galois::StatManager statManager(statOutputFile);
    {
    if (net.ID == 0) {
      Galois::Runtime::reportStat("(NULL)", "Max Iterations", (unsigned long)maxIterations, 0);
      std::ostringstream ss;
      ss << tolerance;
      Galois::Runtime::reportStat("(NULL)", "Tolerance", ss.str(), 0);
    }
    Galois::StatTimer StatTimer_init("TIMER_GRAPH_INIT"), StatTimer_total("TIMER_TOTAL"), StatTimer_hg_init("TIMER_HG_INIT");

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
    Graph* hg;
    if (numPipelinedPhases > 1) {
      numPipelinedPhases = 1;
      if (net.ID == 0) {
        std::cerr << "WARNING: numPipelinedPhases is not supported\n";
      }
    }
    if(enableVCut){
      if(vertexcut == CART_VCUT)
        hg = new Graph_cartesianCut(inputFile,partFolder, net.ID, net.Num, scalefactor, transpose);
      else if(vertexcut == PL_VCUT)
        hg = new Graph_vertexCut(inputFile,partFolder, net.ID, net.Num, scalefactor, transpose, VCutThreshold);
    }
    else {
      hg = new Graph_edgeCut(inputFile,partFolder, net.ID, net.Num, scalefactor, transpose);
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
    StatTimer_hg_init.stop();

    std::cout << "[" << net.ID << "] InitializeGraph::go called\n";
    StatTimer_init.start();
      InitializeGraphNout::go((*hg));
      InitializeGraph::go((*hg));
    StatTimer_init.stop();

    for(auto run = 0; run < numRuns; ++run){
      std::cout << "[" << net.ID << "] PageRank::go run " << run << " called\n";
      std::string timer_str("TIMER_" + std::to_string(run));
      Galois::StatTimer StatTimer_main(timer_str.c_str());

      StatTimer_main.start();
        PageRank::go((*hg));
      StatTimer_main.stop();

      if((run + 1) != numRuns){
        //Galois::Runtime::getHostBarrier().wait();
        (*hg).reset_num_iter(run+1);
        ResetGraph::go((*hg));
        InitializeGraphNout::go((*hg));
        InitializeGraph::go((*hg));
      }
    }

   StatTimer_total.stop();

    // Verify
    if(verify){
#ifdef __GALOIS_HET_CUDA__
      if (personality == CPU) { 
#endif
        for(auto ii = (*hg).begin(); ii != (*hg).end(); ++ii) {
          if ((*hg).isOwned((*hg).getGID(*ii))) Galois::Runtime::printOutput("% %\n", (*hg).getGID(*ii), (*hg).getData(*ii).value);
        }
#ifdef __GALOIS_HET_CUDA__
      } else if(personality == GPU_CUDA)  {
        for(auto ii = (*hg).begin(); ii != (*hg).end(); ++ii) {
          if ((*hg).isOwned((*hg).getGID(*ii))) Galois::Runtime::printOutput("% %\n", (*hg).getGID(*ii), get_node_value_cuda(cuda_ctx, *ii));
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
