/**
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of XYZ License (a copy is located in
 * LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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
 */

#include <iostream>
#include <limits>
#include <algorithm>
#include <vector>
#include "galois/Galois.h"
#include "Lonestar/BoilerPlate.h"
#include "galois/gstl.h"

#include "galois/runtime/CompilerHelperFunctions.h"

#include "galois/graphs/OfflineGraph.h"
#include "galois/Dist/DistGraph.h"

static const char* const name = "PageRank - Compiler Generated Distributed Heterogeneous";
static const char* const desc = "Residual PageRank on Distributed Galois.";
static const char* const url = 0;

namespace cll = llvm::cl;
static cll::opt<std::string> inputFile(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<unsigned int> maxIterations("maxIterations", cll::desc("Maximum iterations"), cll::init(4));
static cll::opt<unsigned int> src_node("startNode", cll::desc("ID of the source node"), cll::init(0));
static cll::opt<float> tolerance("tolerance", cll::desc("tolerance"), cll::init(0.01));
static cll::opt<bool> verify("verify", cll::desc("Verify ranks by printing to 'page_ranks.#hid.csv' file"), cll::init(false));


static const float alpha = (1.0 - 0.85);
//static const float TOLERANCE = 0.01;
struct PR_NodeData {
  float value;
  std::atomic<float> residual;
  unsigned int nout;

};

typedef DistGraph<PR_NodeData, void> Graph;
typedef typename Graph::GraphNode GNode;

typedef GNode WorkItem;

struct InitializeGraph {
  Graph* graph;

  void static go(Graph& _graph) {

       struct Syncer_0 {
      	static float extract( const struct PR_NodeData & node){ return node.residual; }
      	static void reduce (struct PR_NodeData & node, float y) { galois::atomicAdd(node.residual, y);}
      	static void reset (struct PR_NodeData & node ){node.residual = 0 ; }
      	typedef float ValTy;
      };
       struct SyncerPull_0 {
      	static float extract( const struct PR_NodeData & node){ return node.value; }
      	static void setVal (struct PR_NodeData & node, float y) {node.value = y; }
      	typedef float ValTy;
      };
       struct SyncerPull_1 {
      	static unsigned int extract( const struct PR_NodeData & node){ return node.nout; }
      	static void setVal (struct PR_NodeData & node, unsigned int y) {node.nout = y; }
      	typedef unsigned int ValTy;
      };
      galois::do_all(_graph.begin(), _graph.end(), InitializeGraph{ &_graph }, galois::loopname("Init"), galois::write_set("sync_pull", "this->graph", "struct PR_NodeData &", "struct PR_NodeData &", "value" , "float"), galois::write_set("sync_pull", "this->graph", "struct PR_NodeData &", "struct PR_NodeData &", "nout" , "unsigned int"), galois::write_set("sync_push", "this->graph", "struct PR_NodeData &", "struct PR_NodeData &" , "residual", "float" , "{ galois::atomicAdd(node.residual, y);}",  "{node.residual = 0 ; }"));
      _graph.sync_push<Syncer_0>();
      
      _graph.sync_pull<SyncerPull_0>();
      
      _graph.sync_pull<SyncerPull_1>();
      
      
  }

  void operator()(GNode src) const {
    PR_NodeData& sdata = graph->getData(src);
    sdata.value = 1.0 - alpha;
    sdata.nout = std::distance(graph->edge_begin(src), graph->edge_end(src));

    if(sdata.nout > 0 ){
      float delta = sdata.value*alpha/sdata.nout;
      for(auto nbr = graph->edge_begin(src); nbr != graph->edge_end(src); ++nbr){
        GNode dst = graph->getEdgeDst(nbr);
        PR_NodeData& ddata = graph->getData(dst);
        galois::atomicAdd(ddata.residual, delta);
      }
    }
  }
};
/*

template <typename GraphTy>
struct Get_info_functor : public galois::op_tag {
  GraphTy &graph;

  struct Syncer_0 {
    static float extract( const struct PR_NodeData & node){ return node.residual; }
    static void reduce (struct PR_NodeData & node, float y) { galois::atomicAdd(node.residual, y);}
    static void reset (struct PR_NodeData & node ) { node.residual = 0; }
    typedef float ValTy;
  };

  struct SyncerPull_0 {
    static float extract( const struct PR_NodeData & node){ return node.value; }
    static void setVal (struct PR_NodeData & node, float y) {node.value = y; }
    typedef float ValTy;
  };

  Get_info_functor(GraphTy& _g): graph(_g){}
  unsigned operator()(GNode n){
    return graph.getHostID(n);
  }

  uint32_t getLocalID(GNode n){
    return graph.getLID(n);
  }

  void sync_push(){
    sync_push_static(graph);
    //XXX: Why this is not working?
    //graph.sync_push<Syncer_0>();
  }
  void static sync_push_static(Graph& _g){
    _g.sync_push<Syncer_0>();
    _g.sync_pull<SyncerPull_0>();
  }
};
*/


template <typename GraphTy>
struct Get_info_functor : public galois::op_tag {
	GraphTy &graph;
	struct Syncer_0 {
		static float extract( const struct PR_NodeData & node){ return node.residual; }
		static void reduce (struct PR_NodeData & node, float y) { galois::atomicAdd(node.residual, y);}
		static void reset (struct PR_NodeData & node ){node.residual = 0 ; }
		typedef float ValTy;
	};
	struct SyncerPull_0 {
		static float extract( const struct PR_NodeData & node){ return node.value; }
		static void setVal (struct PR_NodeData & node, float y) {node.value = y; }
		typedef float ValTy;
	};
	Get_info_functor(GraphTy& _g): graph(_g){}
	unsigned operator()(GNode n) {
		return graph.getHostID(n);
	}
	uint32_t getLocalID(GNode n){
		return graph.getLID(n);
	}
	void sync_graph(){
		 sync_graph_static(graph);
	}
	void static sync_graph_static(Graph& _graph) {

		_graph.sync_push<Syncer_0>();

		//_graph.sync_pull<SyncerPull_0>();
	}
};

struct PageRank {
  Graph* graph;

  PageRank(Graph* _g): graph(_g){}
  void static go(Graph& _graph) {
     using namespace galois::worklists;
     typedef PerSocketChunkFIFO<64> PSchunk;

     //galois::for_each(_graph.begin(), _graph.end(), PageRank(&_graph), Get_info_functor<Graph>(_graph), galois::wl<PSchunk>());
     galois::for_each(_graph.begin(), _graph.end(), PageRank(&_graph), galois::write_set("sync_pull", "this->graph", "struct PR_NodeData &", "struct PR_NodeData &", "value" , "float"), galois::write_set("sync_push", "this->graph", "struct PR_NodeData &", "struct PR_NodeData &" , "residual", "float" , "{ galois::atomicAdd(node.residual, y);}",  "{node.residual = 0 ; }"), Get_info_functor<Graph>(_graph), galois::wl<PSchunk>());

  }

  void operator()(WorkItem& src, galois::UserContext<WorkItem>& ctx) const {
    PR_NodeData& sdata = graph->getData(src);
    float residual_old = sdata.residual.exchange(0.0);
    sdata.value += residual_old;
    //sdata.residual = residual_old;
    if (sdata.nout > 0){
      float delta = residual_old*alpha/sdata.nout;
      for(auto nbr = graph->edge_begin(src); nbr != graph->edge_end(src); ++nbr){
        GNode dst = graph->getEdgeDst(nbr);
        PR_NodeData& ddata = graph->getData(dst);
        auto dst_residual_old = galois::atomicAdd(ddata.residual, delta);

        //Schedule TOLERANCE threshold crossed.
          //std::cout << "out  : " << (dst_residual_old + delta) << "\n";
        if((dst_residual_old <= tolerance) && ((dst_residual_old + delta) >= tolerance)) {
            //std::cout << "pushed : " << graph->getGID(dst) << " host : " << graph->getHostID(graph->getGID(dst)) << "\n";
          ctx.push(WorkItem(graph->getGID(dst)));
        }
      }
    }
  }
};

int main(int argc, char** argv) {
  try {

    LonestarStart(argc, argv, name, desc, url);
    auto& net = galois::runtime::getSystemNetworkInterface();
    galois::Timer T_total, T_offlineGraph_init, T_DistGraph_init, T_init, T_pageRank;

    T_total.start();

    T_offlineGraph_init.start();
    OfflineGraph g(inputFile);
    T_offlineGraph_init.stop();
    std::cout << g.size() << " " << g.sizeEdges() << "\n";

    T_DistGraph_init.start();
    Graph hg(inputFile, net.ID, net.Num);
    T_DistGraph_init.stop();

    std::cout << "InitializeGraph::go called\n";

    T_init.start();
    InitializeGraph::go(hg);
    T_init.stop();

    // Verify
    if(verify){
      if(net.ID == 0) {
        for(auto ii = hg.begin(); ii != hg.end(); ++ii) {
          std::cout << "[" << *ii << "]  " << hg.getData(*ii).value << "\n";
        }
      }
    }

    std::cout << "PageRank::go called\n";
    T_pageRank.start();
    std::cout << " Starting PageRank with worklist. " << "\n";
    PageRank::go(hg);
    std::cout << " Done. " << "\n";
    T_pageRank.stop();

    // Verify
    if(verify){
      if(net.ID == 0) {
        for(auto ii = hg.begin(); ii != hg.end(); ++ii) {
          std::cout << "[" << *ii << "]  " << hg.getData(*ii).value << "\n";
        }
      }
    }

    T_total.stop();

    std::cout << "[" << net.ID << "]" << " Total Time : " << T_total.get() << " offlineGraph : " << T_offlineGraph_init.get() << " DistGraph : " << T_DistGraph_init.get() << " Init : " << T_init.get() << " PageRank (" << maxIterations << ") : " << T_pageRank.get() << "(msec)\n\n";

    return 0;
  } catch (const char* c) {
      std::cerr << "Error: " << c << "\n";
      return 1;
  }
}
