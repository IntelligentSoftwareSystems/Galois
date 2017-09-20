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

#include "Galois/Runtime/OfflineGraph.h"
#include "Galois/Runtime/hGraph.h"

static const char* const name = "PageRank - Compiler Generated Distributed Heterogeneous";
static const char* const desc = "Residual PageRank on Distributed Galois.";
static const char* const url = 0;

namespace cll = llvm::cl;
static cll::opt<std::string> inputFile(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<unsigned int> maxIterations("maxIterations", cll::desc("Maximum iterations"), cll::init(1000));
static cll::opt<unsigned int> src_node("srcNodeId", cll::desc("ID of the source node"), cll::init(0));
static cll::opt<float> tolerance("tolerance", cll::desc("tolerance"), cll::init(0.01));
static cll::opt<bool> verify("verify", cll::desc("Verify ranks by printing to 'page_ranks.#hid.csv' file"), cll::init(false));


static const float alpha = (1.0 - 0.85);
struct PR_NodeData {
  float value;
  std::atomic<float> residual;
  unsigned int nout;

};

typedef hGraph<PR_NodeData, void> Graph;
typedef typename Graph::GraphNode GNode;

struct InitializeGraph {
  Graph* graph;

  void static go(Graph& _graph) {

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
     struct SyncerPull_1 {
    	static unsigned int extract( const struct PR_NodeData & node){ return node.nout; }
    	static void setVal (struct PR_NodeData & node, unsigned int y) {node.nout = y; }
    	typedef unsigned int ValTy;
    };
    galois::do_all(_graph.begin(), _graph.end(), InitializeGraph{ &_graph }, galois::loopname("Init"), galois::write_set("sync_pull", "this->graph", "struct PR_NodeData &", "struct PR_NodeData &", "value" , "float"), galois::write_set("sync_pull", "this->graph", "struct PR_NodeData &", "struct PR_NodeData &", "nout" , "unsigned int"), galois::write_set("sync_push", "this->graph", "struct PR_NodeData &", "struct PR_NodeData &" , "residual", "float" , "{ galois::atomicAdd(node.residual, y);}",  "0"));
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


static uint32_t num_Hosts_recvd = 0;
static bool didWork = false;
static std::vector<bool> others_didWork_vec;

static void didWork_landingPad(galois::Runtime::RecvBuffer& buf){
  //receive didWork from all and decide.
  uint32_t x_id;
  bool x_didWork;

  gDeserialize(buf, x_id, x_didWork);
  ++num_Hosts_recvd;
  others_didWork_vec.push_back(x_didWork);
}
struct PageRank {
  Graph* graph;

  void static go(Graph& _graph) {
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

     galois::Timer T_compute, T_comm, T_comm_push, T_comm_pull;

    T_compute.start(); 
    galois::do_all(_graph.begin(), _graph.end(), PageRank { &_graph }, galois::loopname("pageRank"), galois::write_set("sync_pull", "this->graph", "struct PR_NodeData &", "struct PR_NodeData &", "value" , "float"), galois::write_set("sync_push", "this->graph", "struct PR_NodeData &", "struct PR_NodeData &" , "residual", "float" , "{ galois::atomicAdd(node.residual, y);}",  "0"));
    T_compute.stop();

    T_comm.start();
    T_comm_push.start();
    _graph.sync_push<Syncer_0>();
    T_comm_push.stop();

    T_comm_pull.start();
    //_graph.sync_pull<SyncerPull_0>();
    T_comm_pull.stop();
    T_comm.stop();


    //std::cout << "[" << galois::Runtime::getSystemNetworkInterface().ID  << "] T_compute : " << T_compute.get() << "(msec)  T_comm : " << T_comm.get() << "(msec)\n";
    //std::cout << "[" << galois::Runtime::getSystemNetworkInterface().ID  << "] T_comm_total : " << T_comm.get()  <<"(msec) T_comm_push : " << T_comm_push.get() << "(msec)  T_comm_pull : " << T_comm_pull.get() << "(msec)\n";
  

  }

  void operator()(GNode src)const {
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
        if(!didWork && (dst_residual_old <= tolerance) && ((dst_residual_old + delta) >= tolerance)) {
          didWork = true;
        }
      }
    }
  }
};

int main(int argc, char** argv) {
  try {

    LonestarStart(argc, argv, name, desc, url);
    auto& net = galois::Runtime::getSystemNetworkInterface();
    galois::Timer T_total, T_offlineGraph_init, T_hGraph_init, T_init, T_pageRank;

    T_total.start();

    T_offlineGraph_init.start();
    OfflineGraph g(inputFile);
    T_offlineGraph_init.stop();
    std::cout << g.size() << " " << g.sizeEdges() << "\n";

    T_hGraph_init.start();
    Graph hg(inputFile, net.ID, net.Num);
    T_hGraph_init.stop();

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
        galois::Runtime::SendBuffer b;
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

    std::cout << "[" << net.ID << "]" << " Total Time : " << T_total.get() << " offlineGraph : " << T_offlineGraph_init.get() << " hGraph : " << T_hGraph_init.get() << " Init : " << T_init.get() << " PageRank (" << maxIterations << ") : " << T_pageRank.get() << "(msec)\n\n";

    if(verify){
      for(auto ii = hg.begin(); ii != hg.end(); ++ii) {
        std::cout << "[" << *ii << "]  " << hg.getData(*ii).value << "\n";
        //galois::Runtime::printOutput("% %\n", hg.getGID(*ii), hg.getData(*ii).value);
      }
    }
    return 0;
  } catch (const char* c) {
      std::cerr << "Error: " << c << "\n";
      return 1;
  }
}
