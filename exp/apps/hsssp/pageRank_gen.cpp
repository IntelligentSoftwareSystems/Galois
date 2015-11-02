#include <iostream>
#include <limits>
#include "Galois/Galois.h"
#include "Lonestar/BoilerPlate.h"
#include "Galois/gstl.h"

#include "Galois/Runtime/CompilerHelperFunctions.h"

#include "OfflineGraph.h"
#include "hGraph.h"

static const char* const name = "PageRank - Compiler Generated Distributed Heterogeneous";
static const char* const desc = "Residual PageRank on Distributed Galois.";
static const char* const url = 0;

namespace cll = llvm::cl;
static cll::opt<std::string> inputFile(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<unsigned int> maxIterations("maxIterations", cll::desc("Maximum iterations"), cll::init(4));
static cll::opt<unsigned int> src_node("srcNodeId", cll::desc("ID of the source node"), cll::init(0));
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
    	static void reduce (struct PR_NodeData & node, float y) { Galois::atomicAdd(node.residual, y);}
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
    Galois::do_all(_graph.begin(), _graph.end(), InitializeGraph{ &_graph }, Galois::loopname("Init"), Galois::write_set("sync_pull", "this->graph", "struct PR_NodeData &", "struct PR_NodeData &", "value" , "float"), Galois::write_set("sync_pull", "this->graph", "struct PR_NodeData &", "struct PR_NodeData &", "nout" , "unsigned int"), Galois::write_set("sync_push", "this->graph", "struct PR_NodeData &", "struct PR_NodeData &" , "residual", "float" , "{ Galois::atomicAdd(node.residual, y);}",  "0"));
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
        Galois::atomicAdd(ddata.residual, delta);
      }
    }
  }
};


struct PageRank {
  Graph* graph;

  void static go(Graph& _graph) {
     struct Syncer_0 {
    	static float extract( const struct PR_NodeData & node){ return node.residual; }
    	static void reduce (struct PR_NodeData & node, float y) { Galois::atomicAdd(node.residual, y);}
    	static void reset (struct PR_NodeData & node ) { node.residual = 0; }
    	typedef float ValTy;
    };
     struct SyncerPull_0 {
    	static float extract( const struct PR_NodeData & node){ return node.value; }
    	static void setVal (struct PR_NodeData & node, float y) {node.value = y; }
    	typedef float ValTy;
    };
    Galois::do_all(_graph.begin(), _graph.end(), PageRank { &_graph }, Galois::loopname("pageRank"), Galois::write_set("sync_pull", "this->graph", "struct PR_NodeData &", "struct PR_NodeData &", "value" , "float"), Galois::write_set("sync_push", "this->graph", "struct PR_NodeData &", "struct PR_NodeData &" , "residual", "float" , "{ Galois::atomicAdd(node.residual, y);}",  "0"));
    _graph.sync_push<Syncer_0>();
    
    _graph.sync_pull<SyncerPull_0>();
    
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
        Galois::atomicAdd(ddata.residual, delta);
      }
    }
  }
};

int main(int argc, char** argv) {
  try {

    LonestarStart(argc, argv, name, desc, url);
    auto& net = Galois::Runtime::getSystemNetworkInterface();
    Galois::Timer T_total, T_offlineGraph_init, T_hGraph_init, T_init, T_pageRank;

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

    std::cout << "PageRank::go called\n";
    T_pageRank.start();
    for (int i = 0; i < maxIterations; ++i) {
      std::cout << " Iteration : " << i << "\n";
      PageRank::go(hg);
    }
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

    std::cout << "[" << net.ID << "]" << " Total Time : " << T_total.get() << " offlineGraph : " << T_offlineGraph_init.get() << " hGraph : " << T_hGraph_init.get() << " Init : " << T_init.get() << " PageRank (" << maxIterations << ") : " << T_pageRank.get() << "(msec)\n\n";

    return 0;
  } catch (const char* c) {
      std::cerr << "Error: " << c << "\n";
      return 1;
  }
}
