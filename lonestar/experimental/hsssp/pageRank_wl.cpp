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
#include "galois/Galois.h"
#include "Lonestar/BoilerPlate.h"
#include "galois/gstl.h"

#include "galois/runtime/CompilerHelperFunctions.h"
#include "galois/runtime/Tracer.h"

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


struct InitializeGraph_ZeroResidual{
  Graph* graph;

  void static go(Graph& _graph){
    galois::do_all(_graph.begin(), _graph.end(), InitializeGraph_ZeroResidual{ &_graph }, galois::loopname("Init zero residual"));
  }

    void operator()(GNode src) const {
      PR_NodeData& sdata = graph->getData(src);
      sdata.residual = 0.0;
    }
};


struct InitializeGraph {
  Graph* graph;

  void static go(Graph& _graph) {
     struct Syncer_0 {
    	static float extract( GNode src,const struct PR_NodeData & node){ return node.residual; }
    	static void reduce (GNode src,struct PR_NodeData & node, float y) { galois::atomicAdd(node.residual, y);}
    	static void reset (GNode src,struct PR_NodeData & node ) { node.residual = 0; }
    	typedef float ValTy;
    };
     struct SyncerPull_0 {
    	static float extract( GNode src,const struct PR_NodeData & node){ return node.value; }
    	static void setVal (GNode src,struct PR_NodeData & node, float y) {node.value = y; }
    	typedef float ValTy;
    };
     struct SyncerPull_1 {
    	static unsigned int extract( GNode src,const struct PR_NodeData & node){ return node.nout; }
    	static void setVal (GNode src,struct PR_NodeData & node, unsigned int y) {node.nout = y; }
    	typedef unsigned int ValTy;
    };
    galois::do_all(_graph.begin(), _graph.end(), InitializeGraph{ &_graph }, galois::loopname("Init"), galois::write_set("sync_pull", "this->graph", "struct PR_NodeData &", "struct PR_NodeData &", "value" , "float"), galois::write_set("sync_pull", "this->graph", "struct PR_NodeData &", "struct PR_NodeData &", "nout" , "unsigned int"), galois::write_set("sync_push", "this->graph", "struct PR_NodeData &", "struct PR_NodeData &" , "residual", "float" , "{ galois::atomicAdd(node.residual, y);}",  "0"));
    _graph.sync_push<Syncer_0>();
    
    //_graph.sync_pull<SyncerPull_0>();
    
    //_graph.sync_pull<SyncerPull_1>();
    
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



struct PageRank {
  Graph* graph;

  PageRank(Graph* _g): graph(_g){}

  template <typename GraphTy>
    struct Get_info_functor : public galois::op_tag {
      GraphTy &graph;

      struct Syncer_0 {
        static float extract(GNode src, const struct PR_NodeData & node){ return node.residual; }
        static void reduce (GNode src,struct PR_NodeData & node, float y) { galois::atomicAdd(node.residual, y);}
        static void reset (GNode src,struct PR_NodeData & node ) { node.residual = 0; }
        typedef float ValTy;
      };

      struct SyncerPull_0 {
        static float extract( GNode src,const struct PR_NodeData & node){ return node.value; }
        static void setVal (GNode src,struct PR_NodeData & node, float y) {node.value = y; }
        typedef float ValTy;
      };

      Get_info_functor(GraphTy& _g): graph(_g){}
      unsigned operator()(GNode n) const {
        return graph.getHostID(n);
      }

      uint32_t getLocalID(GNode n) const {
        return graph.getLID(n);
      }

      void sync_graph(){
        sync_graph_static(graph);
        //XXX: Why this is not working?
        //graph.sync_push<Syncer_0>();
      }
      void static sync_graph_static(Graph& _g){
        _g.sync_push<Syncer_0>();
        //_g.sync_pull<SyncerPull_0>();
      }

      GNode getGNode(uint32_t local_id) const {
              return GNode(graph.getGID(local_id));
          }
    };


  void static go(Graph& _graph) {
    using namespace galois::worklists;
    typedef dChunkedFIFO<64> dChunk;

    galois::for_each(_graph.begin(), _graph.end(), PageRank(&_graph), Get_info_functor<Graph>(_graph), galois::wl<dChunk>());

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
        if((dst_residual_old <= tolerance) && ((dst_residual_old + delta) >= tolerance)) {
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
    galois::Timer T_total, T_offlineGraph_init, T_DistGraph_init, T_init, T_pageRank1, T_pageRank2, T_pageRank3;

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
    galois::runtime::getHostBarrier().wait();

    // Verify
   /*
    if(verify){
      if(net.ID == 0) {
        for(auto ii = hg.begin(); ii != hg.end(); ++ii) {
          std::cout << "[" << *ii << "]  " << hg.getData(*ii).value << "\n";
        }
      }
    }
    */

    std::cout << "PageRank::go wl run1 called  on " << net.ID << "\n";
    T_pageRank1.start();
      PageRank::go(hg);
    T_pageRank1.stop();


    std::cout << "[" << net.ID << "]" << " Total Time : " << T_total.get() << " offlineGraph : " << T_offlineGraph_init.get() << " DistGraph : " << T_DistGraph_init.get() << " Init : " << T_init.get() << " PageRank1 : " << T_pageRank1.get() << " (msec)\n\n";

    galois::runtime::getHostBarrier().wait();

    /** To make residual zero before we start anything **/
    InitializeGraph_ZeroResidual::go(hg);
    galois::runtime::getHostBarrier().wait();
    InitializeGraph::go(hg);


    std::cout << "PageRank::go wl run2 called  on " << net.ID << "\n";
    T_pageRank2.start();
      PageRank::go(hg);
    T_pageRank2.stop();


    std::cout << "[" << net.ID << "]" << " Total Time : " << T_total.get() << " offlineGraph : " << T_offlineGraph_init.get() << " DistGraph : " << T_DistGraph_init.get() << " Init : " << T_init.get() << " PageRank2 : " << T_pageRank2.get() << " (msec)\n\n";

    //galois::runtime::getHostBarrier().wait();

    /** To make residual zero before we start anything **/
    InitializeGraph_ZeroResidual::go(hg);
    galois::runtime::getHostBarrier().wait();
    InitializeGraph::go(hg);

    std::cout << "PageRank::go wl run3 called  on " << net.ID << "\n";
    T_pageRank3.start();
      PageRank::go(hg);
    T_pageRank3.stop();


    std::cout << " Done. " << "\n";



    // Verify
   /*
    if(verify){
      if(net.ID == 0) {
        for(auto ii = hg.begin(); ii != hg.end(); ++ii) {
          std::cout << "[" << *ii << "]  " << hg.getData(*ii).value << "\n";
        }
      }
    }
    */

    T_total.stop();


    auto mean_time = (T_pageRank1.get() + T_pageRank2.get() + T_pageRank3.get())/3;

    std::cout << "[" << net.ID << "]" << " Total Time : " << T_total.get() << " offlineGraph : " << T_offlineGraph_init.get() << " DistGraph : " << T_DistGraph_init.get() << " Init : " << T_init.get() << " PageRank1 : " << T_pageRank1.get() << " PageRank2 : " << T_pageRank2.get() << " PageRank3 : " << T_pageRank3.get() <<" PageRank mean time (3 runs ) (" << maxIterations << ") : " << mean_time << "(msec)\n\n";

    if(verify){
      for(auto ii = hg.begin(); ii != hg.end(); ++ii) {
        galois::runtime::printOutput("% %\n", hg.getGID(*ii), hg.getData(*ii).value);
      }
    }
    return 0;
  } catch (const char* c) {
    std::cerr << "Error: " << c << "\n";
    return 1;
  }
}
