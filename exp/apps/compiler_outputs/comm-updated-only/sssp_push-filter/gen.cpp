/** SSSP -*- C++ -*-
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
 * Compute Single Source Shortest Path on distributed Galois using worklist.
 *
 * @author Gurbinder Gill <gurbinder533@gmail.com>
 */

#include <iostream>
#include <limits>
#include "Galois/Galois.h"
#include "Galois/gstl.h"
#include "Lonestar/BoilerPlate.h"
#include "Galois/Runtime/CompilerHelperFunctions.h"

#include "Galois/Runtime/dGraph_edgeCut.h"
#include "Galois/Runtime/dGraph_cartesianCut.h"

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
   assert(false&& "Invalid personality");
   return "";
}
#endif

static const char* const name = "SSSP - Distributed Heterogeneous with worklist.";
static const char* const desc = "Bellman-Ford SSSP on Distributed Galois.";
static const char* const url = 0;

namespace cll = llvm::cl;
static cll::opt<std::string> inputFile(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<std::string> partFolder("partFolder", cll::desc("path to partitionFolder"), cll::init(""));
static cll::opt<bool> transpose("transpose", cll::desc("transpose the graph in memory after partitioning"), cll::init(false));
static cll::opt<unsigned int> maxIterations("maxIterations", cll::desc("Maximum iterations: Default 1000"), cll::init(1000));
static cll::opt<unsigned int> src_node("srcNodeId", cll::desc("ID of the source node"), cll::init(0));
static cll::opt<bool> verify("verify", cll::desc("Verify ranks by printing to 'page_ranks.#hid.csv' file"), cll::init(false));

static cll::opt<bool> enableVCut("enableVertexCut", cll::desc("Use vertex cut for graph partitioning."), cll::init(false));

static cll::opt<unsigned int> numPipelinedPhases("numPipelinedPhases", cll::desc("num of pipelined phases to overlap computation and communication"), cll::init(1));
static cll::opt<unsigned int> numComputeSubsteps("numComputeSubsteps", cll::desc("num of sub steps of computations within a BSP phase"), cll::init(1));

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

const unsigned int infinity = std::numeric_limits<unsigned int>::max()/4;


struct NodeData {
  std::atomic<unsigned int> dist_current;
  unsigned int dist_old;
};

Galois::DynamicBitSet bitset_dist_current;

typedef hGraph<NodeData, unsigned int> Graph;
typedef hGraph_edgeCut<NodeData, unsigned int> Graph_edgeCut;
typedef hGraph_cartesianCut<NodeData, unsigned int> Graph_cartesianCut;

typedef typename Graph::GraphNode GNode;

struct InitializeGraph {
  const unsigned int &local_infinity;
  cll::opt<unsigned int> &local_src_node;
  Graph *graph;

  InitializeGraph(cll::opt<unsigned int> &_src_node, const unsigned int &_infinity, Graph* _graph) : local_src_node(_src_node), local_infinity(_infinity), graph(_graph){}

  void static go(Graph& _graph) {
    	struct SyncerPull_0 {
    		static unsigned int extract(uint32_t node_id, const struct NodeData & node) {
    		#ifdef __GALOIS_HET_CUDA__
    			if (personality == GPU_CUDA) return get_node_dist_current_cuda(cuda_ctx, node_id);
    			assert (personality == CPU);
    		#endif
    			return node.dist_current;
    		}
    		static bool extract_batch(unsigned from_id, unsigned long long int *b, unsigned int *o, unsigned int *y, size_t *s, DataCommMode *data_mode) {
    		#ifdef __GALOIS_HET_CUDA__
    			if (personality == GPU_CUDA) { batch_get_node_dist_current_cuda(cuda_ctx, from_id, b, o, y, s, data_mode); return true; }
    			assert (personality == CPU);
    		#endif
    			return false;
    		}
    		static bool extract_batch(unsigned from_id, unsigned int *y) {
    		#ifdef __GALOIS_HET_CUDA__
    			if (personality == GPU_CUDA) { batch_get_node_dist_current_cuda(cuda_ctx, from_id, y); return true; }
    			assert (personality == CPU);
    		#endif
    			return false;
    		}
    		static void setVal (uint32_t node_id, struct NodeData & node, unsigned int y) {
    		#ifdef __GALOIS_HET_CUDA__
    			if (personality == GPU_CUDA) set_node_dist_current_cuda(cuda_ctx, node_id, y);
    			else if (personality == CPU)
    		#endif
    				node.dist_current = y;
    		}
    		static bool setVal_batch(unsigned from_id, unsigned long long int *b, unsigned int *o, unsigned int *y, size_t s, DataCommMode data_mode) {
    		#ifdef __GALOIS_HET_CUDA__
    			if (personality == GPU_CUDA) { batch_set_slave_node_dist_current_cuda(cuda_ctx, from_id, b, o, y, s, data_mode); return true; }
    			assert (personality == CPU);
    		#endif
    			return false;
    		}
    		typedef unsigned int ValTy;
    	};
    	struct Syncer_vertexCut_0 {
    		static unsigned int extract(uint32_t node_id, const struct NodeData & node) {
    		#ifdef __GALOIS_HET_CUDA__
    			if (personality == GPU_CUDA) return get_node_dist_current_cuda(cuda_ctx, node_id);
    			assert (personality == CPU);
    		#endif
    			return node.dist_current;
    		}
    		static bool extract_reset_batch(unsigned from_id, unsigned long long int *b, unsigned int *o, unsigned int *y, size_t *s, DataCommMode *data_mode) {
    		#ifdef __GALOIS_HET_CUDA__
    			if (personality == GPU_CUDA) { batch_get_slave_node_dist_current_cuda(cuda_ctx, from_id, b, o, y, s, data_mode); return true; }
    			assert (personality == CPU);
    		#endif
    			return false;
    		}
    		static bool extract_reset_batch(unsigned from_id, unsigned int *y) {
    		#ifdef __GALOIS_HET_CUDA__
    			if (personality == GPU_CUDA) { batch_get_slave_node_dist_current_cuda(cuda_ctx, from_id, y); return true; }
    			assert (personality == CPU);
    		#endif
    			return false;
    		}
    		static bool reduce (uint32_t node_id, struct NodeData & node, unsigned int y) {
    		#ifdef __GALOIS_HET_CUDA__
    			if (personality == GPU_CUDA) set_node_dist_current_cuda(cuda_ctx, node_id, y);
    			else if (personality == CPU)
    		#endif
    				{ Galois::set(node.dist_current, y); }
          return true;
    		}
    		static bool reduce_batch(unsigned from_id, unsigned long long int *b, unsigned int *o, unsigned int *y, size_t s, DataCommMode data_mode) {
    		#ifdef __GALOIS_HET_CUDA__
    			if (personality == GPU_CUDA) { batch_set_node_dist_current_cuda(cuda_ctx, from_id, b, o, y, s, data_mode); return true; }
    			assert (personality == CPU);
    		#endif
    			return false;
    		}
    		static void reset (uint32_t node_id, struct NodeData & node ) {
    		}
    		typedef unsigned int ValTy;
    	};
    #ifdef __GALOIS_HET_CUDA__
    	if (personality == GPU_CUDA) {
        bitset_dist_current_clear_cuda(cuda_ctx);
    		std::string impl_str("CUDA_DO_ALL_IMPL_InitializeGraph_" + (_graph.get_run_identifier()));
    		Galois::StatTimer StatTimer_cuda(impl_str.c_str());
    		StatTimer_cuda.start();
    		InitializeGraph_all_cuda(infinity, src_node, cuda_ctx);
    		StatTimer_cuda.stop();
    	} else if (personality == CPU)
    #endif
    {
    bitset_dist_current.clear();
    Galois::do_all(_graph.begin(), _graph.end(), InitializeGraph {src_node, infinity, &_graph}, Galois::loopname("InitializeGraph"), Galois::numrun(_graph.get_run_identifier()), Galois::write_set("sync_pull", "this->graph", "struct NodeData &", "struct NodeData &", "dist_current" , "unsigned int" , "set",  ""));
    }
    _graph.sync_backward<Syncer_vertexCut_0, SyncerPull_0>("InitializeGraph", bitset_dist_current);
    
  }

  void operator()(GNode src) const {
    NodeData& sdata = graph->getData(src);
    sdata.dist_current = (graph->getGID(src) == local_src_node) ? 0 : local_infinity;
    sdata.dist_old = (graph->getGID(src) == local_src_node) ? 0 : local_infinity;
    bitset_dist_current.set(src);
  }
};

struct FirstItr_SSSP{
Graph * graph;
FirstItr_SSSP(Graph * _graph):graph(_graph){}
void static go(Graph& _graph) {
		unsigned int __begin, __end;
		if (_graph.isLocal(src_node)) {
			__begin = _graph.getLID(src_node);
			__end = __begin + 1;
		} else {
			__begin = 0;
			__end = 0;
		}
	struct Syncer_0 {
		static unsigned int extract(uint32_t node_id, const struct NodeData & node) {
		#ifdef __GALOIS_HET_CUDA__
			if (personality == GPU_CUDA) return get_node_dist_current_cuda(cuda_ctx, node_id);
			assert (personality == CPU);
		#endif
			return node.dist_current;
		}
		static bool extract_reset_batch(unsigned from_id, unsigned long long int *b, unsigned int *o, unsigned int *y, size_t *s, DataCommMode *data_mode) {
		#ifdef __GALOIS_HET_CUDA__
			if (personality == GPU_CUDA) { batch_get_slave_node_dist_current_cuda(cuda_ctx, from_id, b, o, y, s, data_mode); return true; }
			assert (personality == CPU);
		#endif
			return false;
		}
		static bool extract_reset_batch(unsigned from_id, unsigned int *y) {
		#ifdef __GALOIS_HET_CUDA__
			if (personality == GPU_CUDA) { batch_get_slave_node_dist_current_cuda(cuda_ctx, from_id, y); return true; }
			assert (personality == CPU);
		#endif
			return false;
		}
		static bool reduce (uint32_t node_id, struct NodeData & node, unsigned int y) {
		#ifdef __GALOIS_HET_CUDA__
			if (personality == GPU_CUDA) return min_node_dist_current_cuda(cuda_ctx, node_id, y);
			else if (personality == CPU)
		#endif
				{ return y < Galois::min(node.dist_current, y); }
      return false;
		}
		static bool reduce_batch(unsigned from_id, unsigned long long int *b, unsigned int *o, unsigned int *y, size_t s, DataCommMode data_mode) {
		#ifdef __GALOIS_HET_CUDA__
			if (personality == GPU_CUDA) { batch_min_node_dist_current_cuda(cuda_ctx, from_id, b, o, y, s, data_mode); return true; }
			assert (personality == CPU);
		#endif
			return false;
		}
		static void reset (uint32_t node_id, struct NodeData & node ) {
		}
		typedef unsigned int ValTy;
	};
	struct SyncerPull_vertexCut_0 {
		static unsigned int extract(uint32_t node_id, const struct NodeData & node) {
		#ifdef __GALOIS_HET_CUDA__
			if (personality == GPU_CUDA) return get_node_dist_current_cuda(cuda_ctx, node_id);
			assert (personality == CPU);
		#endif
			return node.dist_current;
		}
		static bool extract_batch(unsigned from_id, unsigned long long int *b, unsigned int *o, unsigned int *y, size_t *s, DataCommMode *data_mode) {
		#ifdef __GALOIS_HET_CUDA__
			if (personality == GPU_CUDA) { batch_get_node_dist_current_cuda(cuda_ctx, from_id, b, o, y, s, data_mode); return true; }
			assert (personality == CPU);
		#endif
			return false;
		}
		static bool extract_batch(unsigned from_id, unsigned int *y) {
		#ifdef __GALOIS_HET_CUDA__
			if (personality == GPU_CUDA) { batch_get_node_dist_current_cuda(cuda_ctx, from_id, y); return true; }
			assert (personality == CPU);
		#endif
			return false;
		}
		static void setVal (uint32_t node_id, struct NodeData & node, unsigned int y) {
		#ifdef __GALOIS_HET_CUDA__
			if (personality == GPU_CUDA) set_node_dist_current_cuda(cuda_ctx, node_id, y);
			else if (personality == CPU)
		#endif
				node.dist_current = y;
		}
		static bool setVal_batch(unsigned from_id, unsigned long long int *b, unsigned int *o, unsigned int *y, size_t s, DataCommMode data_mode) {
		#ifdef __GALOIS_HET_CUDA__
			if (personality == GPU_CUDA) { batch_set_slave_node_dist_current_cuda(cuda_ctx, from_id, b, o, y, s, data_mode); return true; }
			assert (personality == CPU);
		#endif
			return false;
		}
		typedef unsigned int ValTy;
	};
#ifdef __GALOIS_HET_CUDA__
	if (personality == GPU_CUDA) {
    bitset_dist_current_clear_cuda(cuda_ctx);
		std::string impl_str("CUDA_DO_ALL_IMPL_SSSP_" + (_graph.get_run_identifier()));
		Galois::StatTimer StatTimer_cuda(impl_str.c_str());
		StatTimer_cuda.start();
		FirstItr_SSSP_cuda(__begin, __end, cuda_ctx);
		StatTimer_cuda.stop();
	} else if (personality == CPU)
#endif
{
bitset_dist_current.clear();
Galois::do_all(_graph.begin() + __begin, _graph.begin() + __end, FirstItr_SSSP{&_graph}, Galois::loopname("SSSP"), Galois::numrun(_graph.get_run_identifier()), Galois::write_set("sync_push", "this->graph", "struct NodeData &", "struct NodeData &" , "dist_current", "unsigned int" , "min",  ""));
}
_graph.sync_forward<Syncer_0, SyncerPull_vertexCut_0>("SSSP", bitset_dist_current);

Galois::Runtime::reportStat("(NULL)", "NUM_WORK_ITEMS_" + (_graph.get_run_identifier()), __end - __begin, 0);

}
void operator()(GNode src) const {
    NodeData& snode = graph->getData(src);
    snode.dist_old = snode.dist_current;

    for (auto jj = graph->edge_begin(src), ee = graph->edge_end(src); jj != ee; ++jj) {
      GNode dst = graph->getEdgeDst(jj);
      auto& dnode = graph->getData(dst);
      unsigned int new_dist = graph->getEdgeData(jj) + snode.dist_current;
      unsigned int old_dist = Galois::atomicMin(dnode.dist_current, new_dist);
      if (old_dist > new_dist) bitset_dist_current.set(dst);
      
    }
  }

};
struct SSSP {
  Graph* graph;

  SSSP(Graph* _graph) : graph(_graph){}
  void static go(Graph& _graph){
    using namespace Galois::WorkList;
    typedef dChunkedFIFO<64> dChunk;
    
    FirstItr_SSSP::go(_graph);
    
    unsigned _num_iterations = 1;
    
    do { 
     _graph.set_num_iter(_num_iterations);
    DGAccumulator_accum.reset();
    	struct Syncer_0 {
    		static unsigned int extract(uint32_t node_id, const struct NodeData & node) {
    		#ifdef __GALOIS_HET_CUDA__
    			if (personality == GPU_CUDA) return get_node_dist_current_cuda(cuda_ctx, node_id);
    			assert (personality == CPU);
    		#endif
    			return node.dist_current;
    		}
    		static bool extract_reset_batch(unsigned from_id, unsigned long long int *b, unsigned int *o, unsigned int *y, size_t *s, DataCommMode *data_mode) {
    		#ifdef __GALOIS_HET_CUDA__
    			if (personality == GPU_CUDA) { batch_get_slave_node_dist_current_cuda(cuda_ctx, from_id, b, o, y, s, data_mode); return true; }
    			assert (personality == CPU);
    		#endif
    			return false;
    		}
    		static bool extract_reset_batch(unsigned from_id, unsigned int *y) {
    		#ifdef __GALOIS_HET_CUDA__
    			if (personality == GPU_CUDA) { batch_get_slave_node_dist_current_cuda(cuda_ctx, from_id, y); return true; }
    			assert (personality == CPU);
    		#endif
    			return false;
    		}
    		static bool reduce (uint32_t node_id, struct NodeData & node, unsigned int y) {
    		#ifdef __GALOIS_HET_CUDA__
    			if (personality == GPU_CUDA) return min_node_dist_current_cuda(cuda_ctx, node_id, y);
    			else if (personality == CPU)
    		#endif
    				{ return y < Galois::min(node.dist_current, y); }
          return false;
    		}
    		static bool reduce_batch(unsigned from_id, unsigned long long int *b, unsigned int *o, unsigned int *y, size_t s, DataCommMode data_mode) {
    		#ifdef __GALOIS_HET_CUDA__
    			if (personality == GPU_CUDA) { batch_min_node_dist_current_cuda(cuda_ctx, from_id, b, o, y, s, data_mode); return true; }
    			assert (personality == CPU);
    		#endif
    			return false;
    		}
    		static void reset (uint32_t node_id, struct NodeData & node ) {
    		}
    		typedef unsigned int ValTy;
    	};
    	struct SyncerPull_vertexCut_0 {
    		static unsigned int extract(uint32_t node_id, const struct NodeData & node) {
    		#ifdef __GALOIS_HET_CUDA__
    			if (personality == GPU_CUDA) return get_node_dist_current_cuda(cuda_ctx, node_id);
    			assert (personality == CPU);
    		#endif
    			return node.dist_current;
    		}
    		static bool extract_batch(unsigned from_id, unsigned long long int *b, unsigned int *o, unsigned int *y, size_t *s, DataCommMode *data_mode) {
    		#ifdef __GALOIS_HET_CUDA__
    			if (personality == GPU_CUDA) { batch_get_node_dist_current_cuda(cuda_ctx, from_id, b, o, y, s, data_mode); return true; }
    			assert (personality == CPU);
    		#endif
    			return false;
    		}
    		static bool extract_batch(unsigned from_id, unsigned int *y) {
    		#ifdef __GALOIS_HET_CUDA__
    			if (personality == GPU_CUDA) { batch_get_node_dist_current_cuda(cuda_ctx, from_id, y); return true; }
    			assert (personality == CPU);
    		#endif
    			return false;
    		}
    		static void setVal (uint32_t node_id, struct NodeData & node, unsigned int y) {
    		#ifdef __GALOIS_HET_CUDA__
    			if (personality == GPU_CUDA) set_node_dist_current_cuda(cuda_ctx, node_id, y);
    			else if (personality == CPU)
    		#endif
    				node.dist_current = y;
    		}
    		static bool setVal_batch(unsigned from_id, unsigned long long int *b, unsigned int *o, unsigned int *y, size_t s, DataCommMode data_mode) {
    		#ifdef __GALOIS_HET_CUDA__
    			if (personality == GPU_CUDA) { batch_set_slave_node_dist_current_cuda(cuda_ctx, from_id, b, o, y, s, data_mode); return true; }
    			assert (personality == CPU);
    		#endif
    			return false;
    		}
    		typedef unsigned int ValTy;
    	};
    unsigned int totalSize = std::distance(_graph.begin(), _graph.end());
    if (totalSize > 0) {
    unsigned int pipeSize = totalSize / numPipelinedPhases;
    assert(pipeSize > numPipelinedPhases);
    if ((totalSize % numPipelinedPhases) > 0) ++pipeSize;
    assert((pipeSize * numPipelinedPhases) >= totalSize);
    for (unsigned int __begin = 0; __begin < totalSize; __begin+=pipeSize) {
      unsigned int __end = __begin + pipeSize;
      if (__end > totalSize) __end = totalSize;
      unsigned int stepTotalSize = __end - __begin;
      unsigned int stepSize = stepTotalSize / numComputeSubsteps;
      assert(stepSize > numComputeSubsteps);
      if ((stepTotalSize % numComputeSubsteps) > 0) ++stepSize;
      assert((stepSize * numComputeSubsteps) >= stepTotalSize);
      for (unsigned int __begin2 = __begin; __begin2 < __end; __begin2+=stepSize) {
        unsigned int __end2 = __begin2 + stepSize;
        if (__end2 > __end) __end2 = __end;
      #ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) {
          if (__begin2 == __begin) bitset_dist_current_clear_cuda(cuda_ctx);
          std::string impl_str("CUDA_DO_ALL_IMPL_SSSP_" + (_graph.get_run_identifier()));
          Galois::StatTimer StatTimer_cuda(impl_str.c_str());
          StatTimer_cuda.start();
          int __retval = 0;
          //SSSP_all_cuda(__retval, cuda_ctx);
          SSSP_cuda(__begin2, __end2, __retval, cuda_ctx);
          DGAccumulator_accum += __retval;
          StatTimer_cuda.stop();
        } else if (personality == CPU)
      #endif
        {
          if (__begin2 == __begin) bitset_dist_current.clear();
          Galois::do_all(_graph.begin() + __begin2, _graph.begin() + __end2, SSSP (&_graph), Galois::loopname("SSSP"), Galois::write_set("sync_push", "this->graph", "struct NodeData &", "struct NodeData &" , "dist_current", "unsigned int" , "min",  ""), Galois::numrun(_graph.get_run_identifier()));
        }
      }
      _graph.sync_forward_pipe<Syncer_0, SyncerPull_vertexCut_0>("SSSP", bitset_dist_current);
    }
    } else {
    for (unsigned int __begin = 0; __begin < numPipelinedPhases; ++__begin) {
      _graph.sync_forward_pipe<Syncer_0, SyncerPull_vertexCut_0>("BFS", bitset_dist_current);
    }
    }
    _graph.sync_forward_wait<Syncer_0, SyncerPull_vertexCut_0>("SSSP", bitset_dist_current);
    
    Galois::Runtime::reportStat("(NULL)", "NUM_WORK_ITEMS_" + (_graph.get_run_identifier()), (unsigned long)DGAccumulator_accum.read_local(), 0);
    ++_num_iterations;
    }while((_num_iterations < maxIterations) && DGAccumulator_accum.reduce());
    if (Galois::Runtime::getSystemNetworkInterface().ID == 0) {
      Galois::Runtime::reportStat("(NULL)", "NUM_ITERATIONS_" + std::to_string(_graph.get_run_num()), (unsigned long)_num_iterations, 0);
    }
    
  }

  static Galois::DGAccumulator<int> DGAccumulator_accum;
void operator()(GNode src) const {
    NodeData& snode = graph->getData(src);

    if(snode.dist_old > snode.dist_current){
        snode.dist_old = snode.dist_current;

    for (auto jj = graph->edge_begin(src), ee = graph->edge_end(src); jj != ee; ++jj) {
      GNode dst = graph->getEdgeDst(jj);
      auto& dnode = graph->getData(dst);
      unsigned int new_dist = graph->getEdgeData(jj) + snode.dist_current;
      unsigned int old_dist = Galois::atomicMin(dnode.dist_current, new_dist);
      if (old_dist > new_dist) bitset_dist_current.set(dst);
      
    }

DGAccumulator_accum+= 1;
      }
  }
};
Galois::DGAccumulator<int>  SSSP::DGAccumulator_accum;


int main(int argc, char** argv) {
  try {
    LonestarStart(argc, argv, name, desc, url);
    Galois::StatManager statManager(statOutputFile);
    {
    auto& net = Galois::Runtime::getSystemNetworkInterface();
    if (net.ID == 0) {
      Galois::Runtime::reportStat("(NULL)", "Max Iterations", (unsigned long)maxIterations, 0);
      Galois::Runtime::reportStat("(NULL)", "Source Node ID", (unsigned long)src_node, 0);
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
    if(enableVCut){
      if (numPipelinedPhases > 1) {
        numPipelinedPhases = 1;
        if (net.ID == 0) {
          std::cerr << "WARNING: numPipelinedPhases is not supported for vertex-cut\n";
        }
      }
      hg = new Graph_cartesianCut(inputFile,partFolder, net.ID, net.Num, scalefactor, transpose);
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
    bitset_dist_current.resize(hg->get_local_total_nodes());
    StatTimer_hg_init.stop();

    std::cout << "[" << net.ID << "] InitializeGraph::go called\n";
    StatTimer_init.start();
      InitializeGraph::go((*hg));
    StatTimer_init.stop();

    for(auto run = 0; run < numRuns; ++run){
      std::cout << "[" << net.ID << "] SSSP::go run " << run << " called\n";
      std::string timer_str("TIMER_" + std::to_string(run));
      Galois::StatTimer StatTimer_main(timer_str.c_str());

      StatTimer_main.start();
        SSSP::go((*hg));
      StatTimer_main.stop();

      if((run + 1) != numRuns){
        //Galois::Runtime::getHostBarrier().wait();
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
        for(auto ii = (*hg).begin(); ii != (*hg).end(); ++ii) {
          if ((*hg).isOwned((*hg).getGID(*ii))) Galois::Runtime::printOutput("% %\n", (*hg).getGID(*ii), (*hg).getData(*ii).dist_current);
        }
#ifdef __GALOIS_HET_CUDA__
      } else if(personality == GPU_CUDA)  {
        for(auto ii = (*hg).begin(); ii != (*hg).end(); ++ii) {
          if ((*hg).isOwned((*hg).getGID(*ii))) Galois::Runtime::printOutput("% %\n", (*hg).getGID(*ii), get_node_dist_current_cuda(cuda_ctx, *ii));
        }
      }
#endif
    }

    // Verify
    if(verifyMax){
      uint32_t max_distance = 0;
#ifdef __GALOIS_HET_CUDA__
      if (personality == CPU) { 
#endif
        for(auto ii = (*hg).begin(); ii != (*hg).end(); ++ii) {
          if(max_distance < (*hg).getData(*ii).dist_current && ((*hg).getData(*ii).dist_current != 1073741823))
            max_distance = (*hg).getData(*ii).dist_current;
        }
#ifdef __GALOIS_HET_CUDA__
      } else if(personality == GPU_CUDA)  {
        for(auto ii = (*hg).begin(); ii != (*hg).end(); ++ii) {
          if(max_distance < get_node_dist_current_cuda(cuda_ctx, *ii) && (get_node_dist_current_cuda(cuda_ctx, *ii) != 1073741823))
            max_distance = get_node_dist_current_cuda(cuda_ctx, *ii);
        }
      }
#endif
      Galois::Runtime::reportStat("(NULL)", "MAX DISTANCE ", (unsigned long)max_distance, 0);
    }

    }
    statManager.reportStat();

    return 0;
  } catch(const char* c) {
    std::cerr << "Error: " << c << "\n";
      return 1;
  }
}
