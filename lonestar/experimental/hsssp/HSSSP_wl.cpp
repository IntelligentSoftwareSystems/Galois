/** DataDriven SSSP -*- C++ -*-
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
 * Compute Data Driven Single Source Shortest Path on distributed Galois.
 *
 * @author Gurbinder Gill <gurbinder533@gmail.com>
 */

#include <iostream>
#include <limits>
#include "galois/Galois.h"
#include "galois/gstl.h"
#include "Lonestar/BoilerPlate.h"
#include "galois/runtime/CompilerHelperFunctions.h"

#include "galois/graphs/OfflineGraph.h"
#include "DistGraph.h"


static const char* const name = "SSSP - Distributed Heterogeneous";
static const char* const desc = "Data Driven Bellman-Ford SSSP on Distributed Galois.";
static const char* const url = 0;

namespace cll = llvm::cl;
static cll::opt<std::string> inputFile(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<unsigned int> maxIterations("maxIterations", cll::desc("Maximum iterations"), cll::init(4));
static cll::opt<unsigned int> src_node("srcNodeId", cll::desc("ID of the source node"), cll::init(0));
static cll::opt<bool> verify("verify", cll::desc("Verify ranks by printing to 'page_ranks.#hid.csv' file"), cll::init(false));


struct NodeData {
  std::atomic<int> dist_current;
};

typedef DistGraph<NodeData, unsigned int> Graph;
typedef typename Graph::GraphNode GNode;

typedef GNode WorkItem;


struct InitializeGraph {
  Graph *graph;

  void static go(Graph& _graph) {

     struct SyncerPull_0 {
    	static int extract( const struct NodeData & node){return node.dist_current; }
    	static void setVal (struct NodeData & node, int y) {node.dist_current = y; }
    	typedef int ValTy;
    };
    galois::do_all(_graph.begin(), _graph.end(), InitializeGraph {&_graph}, galois::loopname("InitGraph"), galois::write_set("sync_pull", "this->graph", "struct NodeData &", "", "dist_current" , "int"));
    _graph.sync_pull<SyncerPull_0>();

  }

  void operator()(GNode src) const {
    NodeData& sdata = graph->getData(src);
    sdata.dist_current = std::numeric_limits<int>::max()/4;
  }
};


/*
template <typename GraphTy>
struct OP : public foo_tag {

  // returns the hostID it belongs to.
  GraphTy* graph;

  OP(GraphTy* _g):graph(_g){}
  int operator()(GNode n){
    return // hostID it belongs to.
  }
};

Then call the same for_each, but in Executor_ForEach.h inside function for_each_gen check if the foo_tag exists (using exists_by_suoertype), if exists then call the separate for_each_gen_exp else just continue the same path.

use get_by_super to extract the functor and call it on all the items in the bag to get the ID they belong to.

Also, no change is required to for_each_imp as it simply forwards the arguments.. so it's cool. just simply pass bag to it.

LET'S DO IT.

*/
template <typename GraphTy>
struct Get_info_functor : public galois::op_tag {
  GraphTy &graph;

    struct Syncer_0 {
    	static int extract(const struct NodeData & node){ return node.dist_current; }
    	static void reduce (struct NodeData & node, int y) {galois::min(node.dist_current, y);}
    	static void reset (struct NodeData & node ) { /*node.dist_current = std::numeric_limits<int>::max();*/ }
    	typedef int ValTy;
    };

  Get_info_functor(GraphTy& _g): graph(_g){}
  unsigned operator()(GNode n){
    return graph.getHostID(n);
  }

  uint32_t getLocalID(GNode n){
    return graph.getLID(n);
  }

  void sync_graph(){
    sync_graph_static(graph);
    //XXX: Why this is not working?
    //graph.sync_push<Syncer_0>();
  }
  void static sync_graph_static(Graph& _g){
    _g.sync_push<Syncer_0>();
  }
};

struct SSSP {
  Graph* graph;

  SSSP(Graph* _g):graph(_g){}
  void static go(Graph& _graph){
    using namespace galois::worklists;
    typedef dChunkedFIFO<64> dChunk;

    //XXX: Need a better way to fix this!!
    if(galois::runtime::getSystemNetworkInterface().ID == 0)
      _graph.getData(src_node).dist_current = 0;
    else
      _graph.getData(src_node).dist_current = std::numeric_limits<int>::max()/4;

    galois::for_each(WorkItem(src_node), SSSP(&_graph), Get_info_functor<Graph>(_graph), galois::wl<dChunk>());
  }

  void operator()(WorkItem& src, galois::UserContext<WorkItem>& ctx) const {
    //GNode src = item.first;
    NodeData& snode = graph->getData(src);
    auto& net = galois::runtime::getSystemNetworkInterface();
    auto& sdist = snode.dist_current;

    for (auto jj = graph->edge_begin(src), ej = graph->edge_end(src); jj != ej; ++jj) {
      GNode dst = graph->getEdgeDst(jj);
      auto& dnode = graph->getData(dst);
      std::atomic<int>& ddist = dnode.dist_current;
      int old_dist = ddist;
      int new_dist = sdist + 1;
      while (old_dist > new_dist){
        if(ddist.compare_exchange_strong(old_dist, new_dist)) {
          ctx.push(WorkItem(graph->getGID(dst)));
          break;
        }
      }
    }
  }
};


int main(int argc, char** argv) {
  try {
    LonestarStart(argc, argv, name, desc, url);
    auto& net = galois::runtime::getSystemNetworkInterface();
    galois::Timer T_total, T_offlineGraph_init, T_DistGraph_init, T_init, T_HSSSP;

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

    // Set node 0 to be source.
    if(net.ID == 1)
      hg.getData(src_node).dist_current = 0;

    // Verify
    if(verify){
      if(net.ID == 0) {
        for(auto ii = hg.begin(); ii != hg.end(); ++ii) {
          std::cout << "[" << *ii << "]  " << hg.getData(*ii).dist_current << "\n";
        }
      }
    }


    std::cout << "SSSP::go called\n";
    T_HSSSP.start();
    for (int i = 0; i < maxIterations; ++i) {
      std::cout << " Iteration : " << i << "\n";
      SSSP::go(hg);
    }
    T_HSSSP.stop();

    // Verify
    if(verify){
      if(net.ID == 0) {
        for(auto ii = hg.begin(); ii != hg.end(); ++ii) {
          std::cout << "[" << *ii << "]  " << hg.getData(*ii).dist_current << "\n";
        }
      }
    }


   T_total.stop();

    std::cout << "[" << net.ID << "]" << " Total Time : " << T_total.get() << " offlineGraph : " << T_offlineGraph_init.get() << " DistGraph : " << T_DistGraph_init.get() << " Init : " << T_init.get() << " HSSSP (" << maxIterations << ") : " << T_HSSSP.get() << "(msec)\n\n";

    return 0;
  } catch(const char* c) {
    std::cerr << "Error: " << c << "\n";
      return 1;
  }
}
