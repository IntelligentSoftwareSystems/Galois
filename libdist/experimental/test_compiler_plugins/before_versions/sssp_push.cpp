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


static const char* const name = "SSSP - Distributed Heterogeneous";
static const char* const desc = "Bellman-Ford SSSP on Distributed Galois.";
static const char* const url = 0;

namespace cll = llvm::cl;
static cll::opt<std::string> inputFile(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<unsigned int> maxIterations("maxIterations", cll::desc("Maximum iterations: Default 1024"), cll::init(1024));
static cll::opt<unsigned int> src_node("startNode", cll::desc("ID of the source node"), cll::init(0));
static cll::opt<bool> verify("verify", cll::desc("Verify ranks by printing to 'page_ranks.#hid.csv' file"), cll::init(false));


struct NodeData {
  std::atomic<unsigned long long> dist_current;
};

typedef DistGraph<NodeData, unsigned int> Graph;
typedef typename Graph::GraphNode GNode;


struct InitializeGraph {
  Graph *graph;

  InitializeGraph(Graph* _graph) :graph(_graph){}
  void static go(Graph& _graph) {

    galois::do_all(_graph.begin(), _graph.end(), InitializeGraph {&_graph}, galois::loopname("InitGraph"));

  }

  void operator()(GNode src) const {
    NodeData& sdata = graph->getData(src);
    sdata.dist_current = std::numeric_limits<unsigned long long>::max()/4;
    auto& net = galois::runtime::getSystemNetworkInterface();
    if((net.ID == 0) && (src == src_node)){
      sdata.dist_current = 0;
    }
  }
};

struct SSSP {
  Graph* graph;
  static galois::DGAccumulator<int> DGAccumulator_accum;

  SSSP(Graph* _graph) : graph(_graph){}
  void static go(Graph& _graph){
    unsigned iteration = 0;
    do{
      DGAccumulator_accum.reset();

      galois::do_all(_graph.begin(), _graph.end(), SSSP { &_graph }, galois::loopname("sssp"));
     ++iteration;
    }while(DGAccumulator_accum.reduce());

    std::cout << " Total iteration run : " << iteration << "\n";
  }

  void operator()(GNode src) const {
    NodeData& snode = graph->getData(src);
    auto& sdist = snode.dist_current;

    for (auto jj = graph->edge_begin(src), ej = graph->edge_end(src); jj != ej; ++jj) {
      GNode dst = graph->getEdgeDst(jj);
      auto& dnode = graph->getData(dst);
      unsigned long long new_dist = graph->getEdgeData(jj) + sdist;
      auto old_dist = galois::atomicMin(dnode.dist_current, new_dist);
      if(old_dist > new_dist){
        DGAccumulator_accum += 1;
      }
    }
  }
};
galois::DGAccumulator<int>  SSSP::DGAccumulator_accum;


int main(int argc, char** argv) {
  try {
    LonestarStart(argc, argv, name, desc, url);
    auto& net = galois::runtime::getSystemNetworkInterface();
    galois::Timer T_total, T_offlineGraph_init, T_DistGraph_init, T_init, T_sssp1, T_sssp2, T_sssp3;

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


    std::cout << "SSSP::go run1 called  on " << net.ID << "\n";
    T_sssp1.start();
      SSSP::go(hg);
    T_sssp1.stop();

    std::cout << "[" << net.ID << "]" << " Total Time : " << T_total.get() << " offlineGraph : " << T_offlineGraph_init.get() << " DistGraph : " << T_DistGraph_init.get() << " Init : " << T_init.get() << " sssp1 : " << T_sssp1.get() << " (msec)\n\n";

    galois::runtime::getHostBarrier().wait();
    InitializeGraph::go(hg);

    std::cout << "SSSP::go run2 called  on " << net.ID << "\n";
    T_sssp2.start();
      SSSP::go(hg);
    T_sssp2.stop();

    std::cout << "[" << net.ID << "]" << " Total Time : " << T_total.get() << " offlineGraph : " << T_offlineGraph_init.get() << " DistGraph : " << T_DistGraph_init.get() << " Init : " << T_init.get() << " sssp2 : " << T_sssp2.get() << " (msec)\n\n";

    galois::runtime::getHostBarrier().wait();
    InitializeGraph::go(hg);

    std::cout << "SSSP::go run3 called  on " << net.ID << "\n";
    T_sssp3.start();
      SSSP::go(hg);
    T_sssp3.stop();

    std::cout << "[" << net.ID << "]" << " Total Time : " << T_total.get() << " offlineGraph : " << T_offlineGraph_init.get() << " DistGraph : " << T_DistGraph_init.get() << " Init : " << T_init.get() << " sssp3 : " << T_sssp3.get() << " (msec)\n\n";


   T_total.stop();

    auto mean_time = (T_sssp1.get() + T_sssp2.get() + T_sssp3.get())/3;

    std::cout << "[" << net.ID << "]" << " Total Time : " << T_total.get() << " offlineGraph : " << T_offlineGraph_init.get() << " DistGraph : " << T_DistGraph_init.get() << " Init : " << T_init.get() << " sssp1 : " << T_sssp1.get() << " sssp2 : " << T_sssp2.get() << " sssp3 : " << T_sssp3.get() <<" sssp mean time (3 runs ) (" << maxIterations << ") : " << mean_time << "(msec)\n\n";

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
