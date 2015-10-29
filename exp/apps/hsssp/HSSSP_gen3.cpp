#include <iostream>
#include <limits>
#include "Galois/Galois.h"
#include "Galois/gstl.h"
#include "Lonestar/BoilerPlate.h"
#include "Galois/Runtime/CompilerHelperFunctions.h"

#include "OfflineGraph.h"
#include "hGraph.h"


static const char* const name = "SSSP - Distributed Heterogeneous";
static const char* const desc = "Bellman-Ford SSSP on Distributed Galois.";
static const char* const url = 0;

namespace cll = llvm::cl;
static cll::opt<std::string> inputFile(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<unsigned int> maxIterations("maxIterations", cll::desc("Maximum iterations"), cll::init(4));
static cll::opt<unsigned int> src_node("srcNodeId", cll::desc("ID of the source node"), cll::init(0));


struct NodeData {
  std::atomic<int> dist_current;
};

typedef hGraph<NodeData, unsigned int> Graph;
typedef typename Graph::GraphNode GNode;


struct InitializeGraph {
  Graph *graph;

  void static go(Graph& _graph) {
    Galois::do_all(_graph.begin(), _graph.end(), InitializeGraph {&_graph}, Galois::loopname("InitGraph"), Galois::write_set( "this->graph", "struct NodeData &", "dist_current" , "sync_pull"));
  }

  void operator()(GNode src) const {
    NodeData& sdata = graph->getData(src);
    sdata.dist_current = std::numeric_limits<int>::max()/4;
  }
};

struct SSSP {
  Graph* graph;

  void static go(Graph& _graph){
     struct Syncer_0 {
    	static int extract( const struct NodeData & node){ return node.dist_current; }
    	static void reduce (struct NodeData & node, int y) {Galois::min(node.dist_current, y);}
    	static void reset (struct NodeData & node ) { node.dist_current = std::numeric_limits<int>::max(); }
    	typedef int ValTy;
    };
    Galois::do_all(_graph.begin(), _graph.end(), SSSP { &_graph }, Galois::loopname("sssp"), Galois::write_set( "this->graph", "struct NodeData &", "dist_current" , "sync_pull"), Galois::write_set( "this->graph", "struct NodeData &", "struct std::atomic<int> &" , "dist_current", "int" , "{Galois::min(node.dist_current, y);}",  "std::numeric_limits<int>::max()", "sync_push"));
    _graph.sync_push<Syncer_0>();
    
     }

  void operator()(GNode src) const {
    NodeData& snode = graph->getData(src);
    auto& sdist = snode.dist_current;

    for (auto jj = graph->edge_begin(src), ej = graph->edge_end(src); jj != ej; ++jj) {
      GNode dst = graph->getEdgeDst(jj);
      auto& dnode = graph->getData(dst);
      std::atomic<int>& ddist = dnode.dist_current;
      int old_dist = ddist;
      int new_dist = graph->getEdgeData(jj) + sdist;
      while (ddist > new_dist){
        ddist.compare_exchange_strong(old_dist, new_dist);
      }
    }
  }
};


int main(int argc, char** argv) {
  try {
    LonestarStart(argc, argv, name, desc, url);
    auto& net = Galois::Runtime::getSystemNetworkInterface();

    OfflineGraph g(inputFile);
    std::cout << g.size() << " " << g.sizeEdges() << "\n";

    Graph hg(inputFile, net.ID, net.Num);

    InitializeGraph::go(hg);
    SSSP::go(hg);

    return 0;
  } catch(const char* c) {
    std::cerr << "Error: " << c << "\n";
      return 1;
  }
}
