#include "Galois/Galois.h"
#include "Galois/Accumulator.h"
#include "Galois/Graph/Graph.h"
#include <iostream>

typedef Galois::Graph::LC_CSR_Graph<int, void> Graph;
typedef Graph::GraphNode GNode;

//! [Id]
struct DeterministicId {
  uintptr_t operator()(GNode x) {
    return x;
  }
};
//! [Id]

struct Matching {
  Graph& graph;
  Galois::GAccumulator<int>& size;

  void operator()(GNode x, Galois::UserContext<GNode>& ctx) const {
    for (auto edge : graph.out_edges(x)) {
      GNode dst = graph.getEdgeDst(edge);
      if (graph.getData(dst) != 0)
        return;
    }

    graph.getData(x, Galois::MethodFlag::WRITE); 
    ctx.cautiousPoint();
    
    graph.getData(x, Galois::MethodFlag::WRITE) = 1;
    for (auto edge : graph.out_edges(x)) {
      GNode dst = graph.getEdgeDst(edge);
      graph.getData(dst) = 1;
    }
    size += 1;
  }
};

//! [Local state]
struct MatchingWithLocalState {
  struct LocalState {
    bool toMark;
    LocalState(MatchingWithLocalState& self, Galois::PerIterAllocTy& alloc): toMark(false) { }
  };
  
  Graph& graph;
  Galois::GAccumulator<int>& size;

  void operator()(GNode x, Galois::UserContext<GNode>& ctx) const { 
    bool used;
    LocalState* p = (LocalState*) ctx.getLocalState(used);
    if (used) {
      // operator is being resumed; use p
      if (!p->toMark)
        return;
      graph.getData(x) = 1;
      for (auto edge : graph.out_edges(x)) {
        GNode dst = graph.getEdgeDst(edge);
        graph.getData(dst) = 1;
      }
      size += 1;
    } else {
      // operator hasn't been suspended yet; execute normally
      // save state into p to be used when operator resumes
      for (auto edge : graph.out_edges(x)) {
        GNode dst = graph.getEdgeDst(edge);
        if (graph.getData(dst) != 0)
          return;
      }
      p->toMark = true;
    }
  }
};

void runLocalStateMatching(const std::string& name) {
  Graph graph;
  Galois::Graph::readGraph(graph, name);
  Galois::GAccumulator<int> size;

  Galois::for_each_local(graph,
      MatchingWithLocalState { graph, size },
      Galois::wl<Galois::WorkList::Deterministic<>>(),
      Galois::has_deterministic_local_state<MatchingWithLocalState::LocalState>(),
      Galois::needs_per_iter_alloc<>(),
      Galois::has_deterministic_id<DeterministicId>());
  std::cout << "Deterministic matching (with local state) size: " << size.reduce() << "\n";
}
//! [Local state]

void runDetMatching(const std::string& name) {
  Graph graph;
  Galois::Graph::readGraph(graph, name);
  Galois::GAccumulator<int> size;

  Galois::for_each(graph.begin(), graph.end(), Matching { graph, size },
      Galois::wl<Galois::WorkList::Deterministic<>>());
  std::cout << "Deterministic matching size: " << size.reduce() << "\n";
}

void runNDMatching(const std::string& name) {
  Graph graph;
  Galois::Graph::readGraph(graph, name);
  Galois::GAccumulator<int> size;

  Galois::for_each(graph.begin(), graph.end(), Matching { graph, size });
  std::cout << "Non-deterministic matching size: " << size.reduce() << "\n";
}

int main(int argc, char** argv) {
  GALOIS_ASSERT(argc > 1);

  Galois::setActiveThreads(2);
  runNDMatching(argv[1]);
  runDetMatching(argv[1]);
  runLocalStateMatching(argv[1]);
  
  return 0;
}
