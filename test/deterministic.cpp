/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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
 */

#include "galois/Galois.h"
#include "galois/Reduction.h"
#include "galois/graphs/Graph.h"
#include <iostream>

typedef galois::graphs::LC_CSR_Graph<int, void> Graph;
typedef Graph::GraphNode GNode;

//! [Id]
struct DeterministicId {
  uintptr_t operator()(GNode x) { return x; }
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
      GNode dst          = graph.getEdgeDst(edge);
      graph.getData(dst) = 1;
    }
    size += 1;
  }
};

//! [Local state]
struct MatchingWithLocalState {
  struct LocalState {
    bool toMark;
    LocalState(MatchingWithLocalState& self, galois::PerIterAllocTy& alloc)
        : toMark(false) {}
  };

  Graph& graph;
  galois::GAccumulator<int>& size;

  template <typename C>
  void operator()(GNode x, C& ctx) const {
    LocalState* p = ctx.template getLocalState<LocalState>();
    if (!ctx.isFirstPass()) {
      // operator is being resumed; use p
      if (!p->toMark)
        return;
      graph.getData(x) = 1;
      for (auto edge : graph.out_edges(x)) {
        GNode dst          = graph.getEdgeDst(edge);
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

  galois::for_each(galois::iterate(graph), MatchingWithLocalState{graph, size},
                   galois::wl<galois::worklists::Deterministic<>>(),
                   galois::local_state<MatchingWithLocalState::LocalState>(),
                   galois::per_iter_alloc(), galois::det_id<DeterministicId>());
  std::cout << "Deterministic matching (with local state) size: "
            << size.reduce() << "\n";
}
//! [Local state]

void runDetMatching(const std::string& name) {
  Graph graph;
  galois::graphs::readGraph(graph, name);
  galois::GAccumulator<int> size;

  galois::for_each(galois::iterate(graph), Matching{graph, size},
                   galois::wl<galois::worklists::Deterministic<>>());
  std::cout << "Deterministic matching size: " << size.reduce() << "\n";
}

void runNDMatching(const std::string& name) {
  Graph graph;
  galois::graphs::readGraph(graph, name);
  galois::GAccumulator<int> size;

  galois::for_each(galois::iterate(graph), Matching{graph, size});
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
