#include <iostream>
#include <limits>
#include "galois/Galois.h"
#include "Lonestar/BoilerPlate.h"
#include "galois/gstl.h"

#include "galois/runtime/CompilerHelperFunctions.h"

#include "galois/graphs/OfflineGraph.h"
#include "galois/runtime/DistGraph.h"

static const char* const name = "Connected Component Label Propagation - Compiler Generated Distributed Heterogeneous";
static const char* const desc = "Connected Component Propagation on Distributed Galois.";
static const char* const url = 0;

namespace cll = llvm::cl;
static cll::opt<std::string> inputFile(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<unsigned int> maxIterations("maxIterations", cll::desc("Maximum iterations"), cll::init(4));
static cll::opt<unsigned int> src_node("startNode", cll::desc("ID of the source node"), cll::init(0));
static cll::opt<bool> verify("verify", cll::desc("Verify ranks by printing to 'page_ranks.#hid.csv' file"), cll::init(false));


static const float alpha = (1.0 - 0.85);
struct CC_NodeData {
  std::atomic<uint32_t> id;
  std::atomic<uint32_t> comp;
};

typedef DistGraph<CC_NodeData, void> Graph;
typedef typename Graph::GraphNode GNode;

struct InitializeGraph {
  Graph* graph;

  void static go(Graph& _graph) {
     struct SyncerPull_0 {
       static uint32_t extract(uint32_t id, const struct CC_NodeData & node){ return node.id; }
       static void setVal (uint32_t id, struct CC_NodeData & node, uint32_t y) {node.id = y; }
    	typedef uint32_t ValTy;
    };
     struct SyncerPull_1 {
       static uint32_t extract(uint32_t id,  const struct CC_NodeData & node){ return node.comp; }
       static void setVal (uint32_t id, struct CC_NodeData & node, uint32_t y) {node.comp = y; }
    	typedef uint32_t ValTy;
    };
    galois::do_all(_graph.begin(), _graph.end(), InitializeGraph{ &_graph }, galois::loopname("Init"), galois::write_set("sync_pull", "this->graph", "struct CC_NodeData &", "struct CC_NodeData &", "id" , "uint32_t"), galois::write_set("sync_pull", "this->graph", "struct CC_NodeData &", "struct CC_NodeData &", "comp" , "uint32_t"));
    _graph.sync_pull<SyncerPull_0>("");
    
    _graph.sync_pull<SyncerPull_1>("");
    
  }

  void operator()(GNode src) const {
    CC_NodeData& sdata = graph->getData(src);
    sdata.id = graph->getGID(src);
    //std::cout << "[" << galois::runtime::getSystemNetworkInterface().ID <<"] sdata.id : " << sdata.id << "\n";
    sdata.comp = sdata.id.load();
  }
};


struct LabelPropAlgo {
  Graph* graph;

  void static go(Graph& _graph) {
     struct Syncer_0 {
       static uint32_t extract(uint32_t id,  const struct CC_NodeData & node){ return node.comp; }
       static void reduce (uint32_t id, struct CC_NodeData & node, uint32_t y) {galois::atomicMin(node.comp, y);}
       static void reset (uint32_t id, struct CC_NodeData & node ) { node.comp = std::numeric_limits<uint32_t>::max(); }
    	typedef uint32_t ValTy;
    };
    galois::do_all(_graph.begin(), _graph.end(), LabelPropAlgo { &_graph }, galois::loopname("LabelPropAlgo"), galois::write_set("sync_push", "this->graph", "struct CC_NodeData &", "struct std::atomic<unsigned int> &" , "comp", "uint32_t" , "{galois::min(node.comp, y);}",  "std::numeric_limits<uint32_t>::max()"));
    _graph.sync_push<Syncer_0>("");
    
  }

  void operator()(GNode src)const {
    CC_NodeData& sdata = graph->getData(src);
    auto& s_comp = sdata.comp;

    for (auto jj = graph->edge_begin(src), ej = graph->edge_end(src); jj != ej; ++jj) {
      GNode dst = graph->getEdgeDst(jj);
      auto& dnode = graph->getData(dst);
      auto& d_comp = dnode.comp;
      uint32_t old_comp = d_comp;
      uint32_t new_comp = s_comp;
      while(d_comp > new_comp) {
       d_comp.compare_exchange_strong(old_comp, new_comp);
      }
      //while(old_comp > new_comp && !d_comp.compare_exchange_strong(old_comp, new_comp)){}
    }
  }
};

int main(int argc, char** argv) {
  try {

    LonestarStart(argc, argv, name, desc, url);
    auto& net = galois::runtime::getSystemNetworkInterface();
    galois::Timer T_total, T_offlineGraph_init, T_DistGraph_init, T_init, T_labelProp;

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
          std::cout << "[" << *ii << "]  " << hg.getData(*ii).comp << "\n";
        }
      }
    }

    std::cout << "PageRank::go called\n";
    T_labelProp.start();
    for (int i = 0; i < maxIterations; ++i) {
      std::cout << " Iteration : " << i << "\n";
      LabelPropAlgo::go(hg);
    }
    T_labelProp.stop();

    // Verify
    if(verify){
      if(net.ID == 0) {
        for(auto ii = hg.begin(); ii != hg.end(); ++ii) {
          std::cout << "[" << *ii << "]  " << hg.getData(*ii).comp<< "\n";
        }
      }
    }

    T_total.stop();

    std::cout << "[" << net.ID << "]" << " Total Time : " << T_total.get() << " offlineGraph : " << T_offlineGraph_init.get() << " DistGraph : " << T_DistGraph_init.get() << " Init : " << T_init.get() << " PageRank (" << maxIterations << ") : " << T_labelProp.get() << "(msec)\n\n";

    return 0;
  } catch (const char* c) {
      std::cerr << "Error: " << c << "\n";
      return 1;
  }
}
