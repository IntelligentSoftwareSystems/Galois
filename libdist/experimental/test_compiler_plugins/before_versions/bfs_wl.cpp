/** BFS -*- C++ -*-
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
#include "galois/Galois.h"
#include "galois/gstl.h"
#include "Lonestar/BoilerPlate.h"
#include "galois/runtime/CompilerHelperFunctions.h"

#include "galois/graphs/OfflineGraph.h"
#include "galois/Dist/DistGraph.h"
#include "galois/DistAccumulator.h"
#include "galois/runtime/Tracer.h"


static const char* const name = "BFS - Distributed Heterogeneous with worklist.";
static const char* const desc = "Bellman-Ford BFS on Distributed Galois.";
static const char* const url = 0;

namespace cll = llvm::cl;
static cll::opt<std::string> inputFile(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<unsigned int> maxIterations("maxIterations", cll::desc("Maximum iterations: Default 1024"), cll::init(1024));
static cll::opt<unsigned int> src_node("startNode", cll::desc("ID of the source node"), cll::init(0));
static cll::opt<bool> verify("verify", cll::desc("Verify ranks by printing to 'page_ranks.#hid.csv' file"), cll::init(false));


struct NodeData {
  std::atomic<unsigned long long> dist_current;
};

typedef DistGraph<NodeData, void> Graph;
typedef typename Graph::GraphNode GNode;


struct InitializeGraph {
  Graph *graph;
  llvm::cl::opt<unsigned int> &local_src_node;

  InitializeGraph(llvm::cl::opt<unsigned int>& _src_node, Graph* _graph) : local_src_node(_src_node), graph(_graph){}
  void static go(Graph& _graph) {

    galois::do_all(_graph.begin(), _graph.end(), InitializeGraph {src_node, &_graph}, galois::loopname("InitGraph"));

  }

  void operator()(GNode src) const {
    NodeData& sdata = graph->getData(src);
    sdata.dist_current = std::numeric_limits<unsigned long long>::max()/4;
    auto& net = galois::runtime::getSystemNetworkInterface();
    if((net.ID == 0) && (src == local_src_node)){
      sdata.dist_current = 0;
    }
  }
};

struct BFS {
  Graph* graph;

  BFS(Graph* _graph) : graph(_graph){}
  void static go(Graph& _graph){
    using namespace galois::worklists;
    typedef dChunkedFIFO<64> dChunk;

      galois::for_each(_graph.begin(), _graph.end(), BFS (&_graph), galois::loopname("BFS"), galois::workList_version());

  }

  void operator()(GNode src, galois::UserContext<GNode>& ctx) const {
    NodeData& snode = graph->getData(src);
    auto& sdist = snode.dist_current;

    for (auto jj = graph->edge_begin(src), ej = graph->edge_end(src); jj != ej; ++jj) {
      GNode dst = graph->getEdgeDst(jj);
      auto& dnode = graph->getData(dst);
      unsigned long long new_dist = 1 + sdist;
      auto old_dist = galois::atomicMin(dnode.dist_current, new_dist);
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
    nd.dist_current = 0;
  }
}

int main(int argc, char** argv) {
  try {
    LonestarStart(argc, argv, name, desc, url);
    auto& net = galois::runtime::getSystemNetworkInterface();
    galois::Timer T_total, T_offlineGraph_init, T_DistGraph_init, T_init, T_BFS1, T_BFS2, T_BFS3;

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
          std::cout << "[" << *ii << "]  " << hg.getData(*ii).dist_current << "\n";
        }
      }
    }
*/


    std::cout << "BFS::go run1 called  on " << net.ID << "\n";
    T_BFS1.start();
      BFS::go(hg);
    T_BFS1.stop();

    std::cout << "[" << net.ID << "]" << " Total Time : " << T_total.get() << " offlineGraph : " << T_offlineGraph_init.get() << " DistGraph : " << T_DistGraph_init.get() << " Init : " << T_init.get() << " BFS1 : " << T_BFS1.get() << " (msec)\n\n";

    galois::runtime::getHostBarrier().wait();
    InitializeGraph::go(hg);

    std::cout << "BFS::go run2 called  on " << net.ID << "\n";
    T_BFS2.start();
      BFS::go(hg);
    T_BFS2.stop();

    std::cout << "[" << net.ID << "]" << " Total Time : " << T_total.get() << " offlineGraph : " << T_offlineGraph_init.get() << " DistGraph : " << T_DistGraph_init.get() << " Init : " << T_init.get() << " BFS2 : " << T_BFS2.get() << " (msec)\n\n";

    galois::runtime::getHostBarrier().wait();
    InitializeGraph::go(hg);

    std::cout << "BFS::go run3 called  on " << net.ID << "\n";
    T_BFS3.start();
      BFS::go(hg);
    T_BFS3.stop();

    std::cout << "[" << net.ID << "]" << " Total Time : " << T_total.get() << " offlineGraph : " << T_offlineGraph_init.get() << " DistGraph : " << T_DistGraph_init.get() << " Init : " << T_init.get() << " BFS3 : " << T_BFS3.get() << " (msec)\n\n";


   T_total.stop();

    auto mean_time = (T_BFS1.get() + T_BFS2.get() + T_BFS3.get())/3;

    std::cout << "[" << net.ID << "]" << " Total Time : " << T_total.get() << " offlineGraph : " << T_offlineGraph_init.get() << " DistGraph : " << T_DistGraph_init.get() << " Init : " << T_init.get() << " BFS1 : " << T_BFS1.get() << " BFS2 : " << T_BFS2.get() << " BFS3 : " << T_BFS3.get() <<" BFS mean time (3 runs ) (" << maxIterations << ") : " << mean_time << "(msec)\n\n";

    if(verify){
      for(auto ii = hg.begin(); ii != hg.end(); ++ii) {
        galois::runtime::printOutput("% %\n", hg.getGID(*ii), hg.getData(*ii).dist_current);
      }
    }
    return 0;
  } catch(const char* c) {
    std::cerr << "Error: " << c << "\n";
      return 1;
  }
}
