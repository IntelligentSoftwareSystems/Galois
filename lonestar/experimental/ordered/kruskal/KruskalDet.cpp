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
#include "galois/Bag.h"
#include "galois/Timer.h"
#include "galois/graphs/LCGraph.h"
#include "galois/graphs/TypeTraits.h"
#include "Lonestar/BoilerPlate.h"

#include <deque>
#include <iostream>

namespace cll = llvm::cl;
static cll::opt<std::string>
    filename(cll::Positional, cll::desc("<input graph>"), cll::Required);
static cll::opt<bool> symmetricGraph("symmetricGraph",
                                     cll::desc("Input graph is symmetric"));

struct Node {
  Node* parent;
  Node() : parent() {}
};

typedef galois::graphs::LC_CSR_Graph<Node, int> Graph;
typedef Graph::GraphNode GNode;

struct Edge {
  GNode src;
  GNode dst;
  int weight;
};

class Process {
  Graph& graph;
  galois::GAccumulator<size_t>& weight;

public:
  struct LocalState {
    LocalState(Process& self, galois::PerIterAllocTy& alloc) {}
  };

  struct DeterministicId {
    uintptr_t operator()(const Edge& e) const { return e.weight; }
  };

  typedef std::tuple<
      galois::fixed_neighborhood, galois::det_id<DeterministicId>, galois::local_state<LocalState>, galois::per_iter_alloc, >
      function_traits;

  Process(Graph& g, galois::GAccumulator<size_t>& w) : graph(g), weight(w) {}

  void operator()(const Edge& e, galois::UserContext<Edge>& ctx) const {
    if (!ctx.isFirstPass())
      return;

    Node& n1 = graph.getData(e.src);
    Node& n2 = graph.getData(e.dst);
  }
};

int main(int argc, char** argv) {
  galois::StatManager statManager;
  LonestarStart(argc, argv, "kruskal", nullptr, nullptr);

  galois::StatTimer Ttotal("TotalTime");
  Ttotal.start();

  Graph graph;
  galois::graphs::readGraph(graph, filename);

  // galois::InsertBag<Edge> edges;
  std::deque<Edge> edges;
  galois::do_all(graph, [&](GNode n1) {
    for (auto edge : graph.out_edges(n1)) {
      GNode n2 = graph.getEdgeDst(edge);
      if (n1 == n2)
        continue;
      if (symmetricGraph && n1 > n2)
        continue;
      Edge e = {n1, n2, graph.getEdgeData(edge)};
      // edges.push(e);
      edges.push_back(e);
    }
  });
  /// XXX
  std::sort(edges.begin(), edges.end(),
            [](const Edge& a, const Edge& b) -> bool {
              if (a.weight == b.weight)
                return a.src == b.src ? a.dst < b.dst : a.src < b.src;
              return a.weight < b.weight;
            });

  galois::reportPageAlloc("MeminfoPre");
  galois::StatTimer T;
  T.start();
  galois::GAccumulator<size_t> weight;
  galois::for_each(edges.begin(), edges.end(), Process(graph, weight),
                   galois::wl<galois::worklists::Deterministic<>>());
  T.stop();
  galois::reportPageAlloc("MeminfoPost");

  std::cout << "MST weight: " << weight.reduce() << "\n";
  Ttotal.stop();

  return 0;
}
