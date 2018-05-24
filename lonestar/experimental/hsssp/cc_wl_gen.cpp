#include <iostream>
#include <limits>
#include "galois/Galois.h"
#include "galois/gstl.h"
#include "Lonestar/BoilerPlate.h"
#include "galois/runtime/CompilerHelperFunctions.h"

#include "galois/graphs/OfflineGraph.h"
#include "galois/Dist/DistGraph.h"
#include "galois/DistAccumulator.h"
#include "galois/runtime/Tracer.h"


static const char* const name = "ConnectedComp - Distributed Heterogeneous with worklist.";
static const char* const desc = "Bellman-Ford ConnectedComp on Distributed Galois.";
static const char* const url = 0;

namespace cll = llvm::cl;
static cll::opt<std::string> inputFile(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<unsigned int> maxIterations("maxIterations", cll::desc("Maximum iterations: Default 1024"), cll::init(1024));
static cll::opt<unsigned int> src_node("startNode", cll::desc("ID of the source node"), cll::init(0));
static cll::opt<bool> verify("verify", cll::desc("Verify ranks by printing to 'page_ranks.#hid.csv' file"), cll::init(false));


struct NodeData {
  std::atomic<unsigned long long> comp_current;
};

typedef DistGraph<NodeData, void> Graph;
typedef typename Graph::GraphNode GNode;


struct InitializeGraph {
  Graph *graph;

  InitializeGraph(Graph* _graph) : graph(_graph){}
  void static go(Graph& _graph) {
    struct SyncerPull_0 {
      static unsigned long long extract(uint32_t node_id, const struct NodeData & node) {
#ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) return get_node_comp_current_cuda(cuda_ctx, node_id);
        assert (personality == CPU);
#endif
        return node.comp_current;
      }
      static void setVal (uint32_t node_id, struct NodeData & node, unsigned long long y) {
#ifdef __GALOIS_HET_CUDA__
        if (personality == GPU_CUDA) set_node_comp_current_cuda(cuda_ctx, node_id, y);
        else if (personality == CPU)
#endif
          node.comp_current = y;
      }
      typedef unsigned long long ValTy;
    };

#ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      InitializeGraph_cuda(cuda_ctx);
    } else if (personality == CPU)
    #endif
    #ifdef __GALOIS_HET_CUDA__
    	if (personality == GPU_CUDA) {
    		InitializeGraph_cuda(cuda_ctx);
    	} else if (personality == CPU)
    #endif
    #ifdef __GALOIS_HET_CUDA__
    	if (personality == GPU_CUDA) {
    		InitializeGraph_cuda(cuda_ctx);
    	} else if (personality == CPU)
    #endif
    galois::do_all(_graph.begin(), _graph.end(), InitializeGraph {&_graph}, galois::loopname("InitGraph"));

    _graph.sync_pull<SyncerPull_0>();
  }

  void operator()(GNode src) const {
    NodeData& sdata = graph->getData(src);
    sdata.comp_current = graph->getGID(src);
  }
};

template <typename GraphTy>
struct Get_info_functor : public galois::op_tag {
	GraphTy &graph;
	struct Syncer_0 {
		static unsigned long long extract(uint32_t node_id, const struct NodeData & node) {
		#ifdef __GALOIS_HET_CUDA__
			if (personality == GPU_CUDA) return get_node_comp_current_cuda(cuda_ctx, node_id);
			assert (personality == CPU);
		#endif
			return node.comp_current;
		}
		static void reduce (uint32_t node_id, struct NodeData & node, unsigned long long y) {
		#ifdef __GALOIS_HET_CUDA__
			if (personality == GPU_CUDA) add_node_comp_current_cuda(cuda_ctx, node_id, y);
			else if (personality == CPU)
		#endif
				{ galois::atomicMin(node.comp_current, y);}
		}
		static void reset (uint32_t node_id, struct NodeData & node ) {
		#ifdef __GALOIS_HET_CUDA__
			if (personality == GPU_CUDA) set_node_comp_current_cuda(cuda_ctx, node_id, 0);
			else if (personality == CPU)
		#endif
				{node.comp_current = std::numeric_limits<unsigned long long>::max()/4; }
		}
		typedef unsigned long long ValTy;
	};
	struct SyncerPull_0 {
		static unsigned long long extract(uint32_t node_id, const struct NodeData & node) {
		#ifdef __GALOIS_HET_CUDA__
			if (personality == GPU_CUDA) return get_node_comp_current_cuda(cuda_ctx, node_id);
			assert (personality == CPU);
		#endif
			return node.comp_current;
		}
		static void setVal (uint32_t node_id, struct NodeData & node, unsigned long long y) {
		#ifdef __GALOIS_HET_CUDA__
			if (personality == GPU_CUDA) set_node_comp_current_cuda(cuda_ctx, node_id, y);
			else if (personality == CPU)
		#endif
				node.comp_current = y;
		}
		typedef unsigned long long ValTy;
	};
	Get_info_functor(GraphTy& _g): graph(_g){}
	unsigned operator()(GNode n) const {
		return graph.getHostID(n);
	}
	GNode getGNode(uint32_t local_id) const {
		return GNode(graph.getGID(local_id));
	}
	uint32_t getLocalID(GNode n) const {
		return graph.getLID(n);
	}
	void sync_graph(){
		 sync_graph_static(graph);
	}
	void static sync_graph_static(Graph& _graph) {

		_graph.sync_push<Syncer_0>();

		_graph.sync_pull<SyncerPull_0>();
	}
};

struct ConnectedComp {
  Graph* graph;

  ConnectedComp(Graph* _graph) : graph(_graph){}
  void static go(Graph& _graph){
    using namespace galois::worklists;
    typedef dChunkedFIFO<64> dChunk;

      #ifdef __GALOIS_HET_CUDA__
      	if (personality == GPU_CUDA) {
      		galois::Timer T_compute, T_comm_syncGraph, T_comm_bag;
      		unsigned num_iter = 0;
      		auto __sync_functor = Get_info_functor<Graph>(_graph);
      		typedef galois::DGBag<GNode, Get_info_functor<Graph> > DBag;
      		DBag dbag(__sync_functor);
      		auto &local_wl = DBag::get();
      		T_compute.start();
      		cuda_wl.num_in_items = _graph.getNumOwned();
      		for (int __i = 0; __i < cuda_wl.num_in_items; ++__i) cuda_wl.in_items[__i] = __i;
      		if (cuda_wl.num_in_items > 0)
      			ConnectedComp_cuda(cuda_ctx);
      		T_compute.stop();
      		T_comm_syncGraph.start();
      		__sync_functor.sync_graph();
      		T_comm_syncGraph.stop();
      		T_comm_bag.start();
      		dbag.set_local(cuda_wl.out_items, cuda_wl.num_out_items);
      		dbag.sync();
      		cuda_wl.num_out_items = 0;
      		T_comm_bag.stop();
      		//std::cout << "[" << galois::runtime::getSystemNetworkInterface().ID << "] Iter : " << num_iter << " T_compute : " << T_compute.get() << "(msec) T_comm_syncGraph : " << T_comm_syncGraph.get() << "(msec) T_comm_bag : " << T_comm_bag.get() << "(msec) \n";
      		while (!dbag.canTerminate()) {
      		++num_iter;
      		//std::cout << "[" << galois::runtime::getSystemNetworkInterface().ID << "] Iter : " << num_iter << " Total items to work on : " << cuda_wl.num_in_items << "\n";
      		T_compute.start();
      		cuda_wl.num_in_items = local_wl.size();
      		std::copy(local_wl.begin(), local_wl.end(), cuda_wl.in_items);
      		if (cuda_wl.num_in_items > 0)
      			ConnectedComp_cuda(cuda_ctx);
      		T_compute.stop();
      		T_comm_syncGraph.start();
      		__sync_functor.sync_graph();
      		T_comm_syncGraph.stop();
      		T_comm_bag.start();
      		dbag.set_local(cuda_wl.out_items, cuda_wl.num_out_items);
      		dbag.sync();
      		cuda_wl.num_out_items = 0;
      		T_comm_bag.stop();
      		//std::cout << "[" << galois::runtime::getSystemNetworkInterface().ID << "] Iter : " << num_iter << " T_compute : " << T_compute.get() << "(msec) T_comm_syncGraph : " << T_comm_syncGraph.get() << "(msec) T_comm_bag : " << T_comm_bag.get() << "(msec) \n";
      		}
      	} else if (personality == CPU)
      #endif
      galois::for_each(_graph.begin(), _graph.end(), ConnectedComp (&_graph), galois::loopname("ConnectedComp"), galois::workList_version(), galois::write_set("sync_push", "this->graph", "struct NodeData &", "struct NodeData &" , "comp_current", "unsigned long long" , "{ galois::atomicMin(node.comp_current, y);}",  "{node.comp_current = std::numeric_limits<unsigned long long>::max()/4; }"), galois::write_set("sync_pull", "this->graph", "struct NodeData &", "struct NodeData &", "comp_current" , "unsigned long long"), Get_info_functor<Graph>(_graph));

  }

  void operator()(GNode src, galois::UserContext<GNode>& ctx) const {
    NodeData& snode = graph->getData(src);
    auto& sdist = snode.comp_current;

    for (auto jj = graph->edge_begin(src), ej = graph->edge_end(src); jj != ej; ++jj) {
      GNode dst = graph->getEdgeDst(jj);
      auto& dnode = graph->getData(dst);
      unsigned long long new_dist = sdist;
      auto old_dist = galois::atomicMin(dnode.comp_current, new_dist);
      if(old_dist > new_dist){
        ctx.push(graph->getGID(dst));
      }
    }
  }
};

/********Set source Node ************/
void setSource(Graph& _graph){
  auto& net = galois::runtime::getSystemNetworkInterface();
  if(net.ID == 0){
    auto& nd = _graph.getData(src_node);
    nd.comp_current = 0;
  }
}

int main(int argc, char** argv) {
  try {
    LonestarStart(argc, argv, name, desc, url);
    auto& net = galois::runtime::getSystemNetworkInterface();
    galois::Timer T_total, T_offlineGraph_init, T_DistGraph_init, T_init, T_ConnectedComp1, T_ConnectedComp2, T_ConnectedComp3;

    T_total.start();

    T_DistGraph_init.start();
    Graph hg(inputFile, net.ID, net.Num);
    T_DistGraph_init.stop();

    std::cout << "InitializeGraph::go called\n";
    T_init.start();
    InitializeGraph::go(hg);
    T_init.stop();

   // Verify
/*
    if(verify){
      if(net.ID == 0) {
        for(auto ii = hg.begin(); ii != hg.end(); ++ii) {
          std::cout << "[" << *ii << "]  " << hg.getData(*ii).comp_current << "\n";
        }
      }
    }
*/


    std::cout << "ConnectedComp::go run1 called  on " << net.ID << "\n";
    T_ConnectedComp1.start();
      ConnectedComp::go(hg);
    T_ConnectedComp1.stop();

    std::cout << "[" << net.ID << "]" << " Total Time : " << T_total.get() << " offlineGraph : " << T_offlineGraph_init.get() << " DistGraph : " << T_DistGraph_init.get() << " Init : " << T_init.get() << " ConnectedComp1 : " << T_ConnectedComp1.get() << " (msec)\n\n";

    galois::runtime::getHostBarrier().wait();
    InitializeGraph::go(hg);

    std::cout << "ConnectedComp::go run2 called  on " << net.ID << "\n";
    T_ConnectedComp2.start();
      ConnectedComp::go(hg);
    T_ConnectedComp2.stop();

    std::cout << "[" << net.ID << "]" << " Total Time : " << T_total.get() << " offlineGraph : " << T_offlineGraph_init.get() << " DistGraph : " << T_DistGraph_init.get() << " Init : " << T_init.get() << " ConnectedComp2 : " << T_ConnectedComp2.get() << " (msec)\n\n";

    galois::runtime::getHostBarrier().wait();
    InitializeGraph::go(hg);

    std::cout << "ConnectedComp::go run3 called  on " << net.ID << "\n";
    T_ConnectedComp3.start();
      ConnectedComp::go(hg);
    T_ConnectedComp3.stop();

    std::cout << "[" << net.ID << "]" << " Total Time : " << T_total.get() << " offlineGraph : " << T_offlineGraph_init.get() << " DistGraph : " << T_DistGraph_init.get() << " Init : " << T_init.get() << " ConnectedComp3 : " << T_ConnectedComp3.get() << " (msec)\n\n";


   T_total.stop();

    auto mean_time = (T_ConnectedComp1.get() + T_ConnectedComp2.get() + T_ConnectedComp3.get())/3;

    std::cout << "[" << net.ID << "]" << " Total Time : " << T_total.get() << " offlineGraph : " << T_offlineGraph_init.get() << " DistGraph : " << T_DistGraph_init.get() << " Init : " << T_init.get() << " ConnectedComp1 : " << T_ConnectedComp1.get() << " ConnectedComp2 : " << T_ConnectedComp2.get() << " ConnectedComp3 : " << T_ConnectedComp3.get() <<" ConnectedComp mean time (3 runs ) (" << maxIterations << ") : " << mean_time << "(msec)\n\n";

    if(verify){
      for(auto ii = hg.begin(); ii != hg.end(); ++ii) {
        galois::runtime::printOutput("% %\n", hg.getGID(*ii), hg.getData(*ii).comp_current);
      }
    }
    return 0;
  } catch(const char* c) {
    std::cerr << "Error: " << c << "\n";
      return 1;
  }
}
