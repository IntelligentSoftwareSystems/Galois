#include "Galois/Graph/FileGraph.h"
#include "Galois/Runtime/ll/gio.h"

typedef Galois::Graph::FileGraph Graph;

template<typename Fn>
void testBasic(Graph&& graph, const std::string& f, Fn fn) {
  Graph g = std::move(graph);
  fn(g, f);
  auto numNodes = g.size();
  auto numEdges = g.sizeEdges();
  for (auto n : g) {
    numNodes -= 1;
    numEdges -= std::distance(g.edge_begin(n), g.edge_end(n));
  }
  GALOIS_ASSERT(numNodes == 0);
  GALOIS_ASSERT(numEdges == 0);
}

int main(int argc, char** argv) {
  GALOIS_ASSERT(argc > 1);
  testBasic(Galois::Graph::FileGraph(), argv[1], [](Graph& g, std::string f) { g.fromFile(f); });
  testBasic(Galois::Graph::FileGraph(), argv[1], [](Graph& g, std::string f) { g.fromFileInterleaved<void>(f); });

  return 0;
}
