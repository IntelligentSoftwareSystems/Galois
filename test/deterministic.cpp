#include "galois/Galois.h"
#include "galois/Accumulator.h"
#include "galois/graphs/Graph.h"
#include <iostream>

typedef galois::graphs::LC_CSR_Graph<int, void> Graph;
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
  galois::GAccumulator<int>& size;

  void operator()(GNode x, galois::UserContext<GNode>& ctx) const {
    for (auto edge : graph.out_edges(x)) {
      GNode dst = graph.getEdgeDst(edge);
      if (graph.getData(dst) != 0)
        return;
    }

    graph.getData(x, galois::MethodFlag::WRITE); 
    ctx.cautiousPoint();
    
    graph.getData(x, galois::MethodFlag::WRITE) = 1;
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
    LocalState(MatchingWithLocalState& self, galois::PerIterAllocTy& alloc): toMark(false) { }
  };
  
  Graph& graph;
  galois::GAccumulator<int>& size;

  void operator()(GNode x, galois::UserContext<GNode>& ctx) const { 
    LocalState* p = (LocalState*) ctx.getLocalState();
    if (!ctx.isFirstPass()) {
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
  galois::graphs::readGraph(graph, name);
  galois::GAccumulator<int> size;

  galois::for_each_local(graph,
      MatchingWithLocalState { graph, size },
      galois::wl<galois::worklists::Deterministic<>>(),
      galois::has_deterministic_local_state<MatchingWithLocalState::LocalState>(),
      galois::needs_per_iter_alloc<>(),
      galois::has_deterministic_id<DeterministicId>());
  std::cout << "Deterministic matching (with local state) size: " << size.reduce() << "\n";
}
//! [Local state]

void runDetMatching(const std::string& name) {
  Graph graph;
  galois::graphs::readGraph(graph, name);
  galois::GAccumulator<int> size;

  galois::for_each(graph.begin(), graph.end(), Matching { graph, size },
      galois::wl<galois::worklists::Deterministic<>>());
  std::cout << "Deterministic matching size: " << size.reduce() << "\n";
}

void runNDMatching(const std::string& name) {
  Graph graph;
  galois::graphs::readGraph(graph, name);
  galois::GAccumulator<int> size;

  galois::for_each(graph.begin(), graph.end(), Matching { graph, size });
  std::cout << "Non-deterministic matching size: " << size.reduce() << "\n";
}

int main(int argc, char** argv) {
  GALOIS_ASSERT(argc > 1);

  galois::setActiveThreads(2);
  runNDMatching(argv[1]);
  runDetMatching(argv[1]);
  runLocalStateMatching(argv[1]);
  
  return 0;
}
