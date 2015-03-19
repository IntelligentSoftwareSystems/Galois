#include "Galois/Galois.h"
#include "Galois/Bag.h"
#include "Galois/Timer.h"
#include "Galois/Graph/LCGraph.h"
#include "Galois/Graph/TypeTraits.h"
#include "Lonestar/BoilerPlate.h"

#include <deque>
#include <iostream>

namespace cll = llvm::cl;
static cll::opt<std::string> filename(cll::Positional, cll::desc("<input graph>"), cll::Required);
static cll::opt<bool> symmetricGraph("symmetricGraph", cll::desc("Input graph is symmetric"));

struct Node {
  Node* parent;
  Node(): parent() { }
};

typedef Galois::Graph::LC_CSR_Graph<Node, int> Graph;
typedef Graph::GraphNode GNode;

struct Edge {
  GNode src;
  GNode dst;
  int weight;
};

class Process {
  Graph& graph;
  Galois::GAccumulator<size_t>& weight;

public:
  struct LocalState {
    LocalState(Process& self, Galois::PerIterAllocTy& alloc) { }
  };

  struct DeterministicId {
    uintptr_t operator()(const Edge& e) const {
      return e.weight;
    }
  };

  typedef std::tuple<
    Galois::has_fixed_neighborhood<>,
    Galois::has_deterministic_id<DeterministicId>,
    Galois::has_deterministic_local_state<LocalState>,
    Galois::needs_per_iter_alloc<>
  > function_traits;

  Process(Graph& g, Galois::GAccumulator<size_t>& w): graph(g), weight(w) { }

  void operator()(const Edge& e, Galois::UserContext<Edge>& ctx) const {
    if (!ctx.isFirstPass())
      return;

    Node& n1 = graph.getData(e.src);
    Node& n2 = graph.getData(e.dst);
  }
};

int main(int argc, char** argv) {
  Galois::StatManager statManager;
  LonestarStart(argc, argv, "kruskal", nullptr, nullptr);
  
  Galois::StatTimer Ttotal("TotalTime");
  Ttotal.start();

  Graph graph;
  Galois::Graph::readGraph(graph, filename);

  //Galois::InsertBag<Edge> edges;
  std::deque<Edge> edges;
  Galois::do_all_local(graph, [&](GNode n1) {
    for (auto edge : graph.out_edges(n1)) {
      GNode n2 = graph.getEdgeDst(edge);
      if (n1 == n2)
        continue;
      if (symmetricGraph && n1 > n2)
        continue;
      Edge e = { n1, n2, graph.getEdgeData(edge) };
      //edges.push(e);
      edges.push_back(e);
    }
  });
  /// XXX
  std::sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) -> bool {
    if (a.weight == b.weight)
      return a.src == b.src ? a.dst < b.dst : a.src < b.src;
    return a.weight < b.weight; 
  });

  Galois::reportPageAlloc("MeminfoPre");
  Galois::StatTimer T;
  T.start();
  Galois::GAccumulator<size_t> weight;
  Galois::for_each(edges.begin(), edges.end(), Process(graph, weight), Galois::wl<Galois::WorkList::Deterministic<>>());
  T.stop();
  Galois::reportPageAlloc("MeminfoPost");

  std::cout << "MST weight: " << weight.reduce() << "\n";
  Ttotal.stop();

  return 0;
}
