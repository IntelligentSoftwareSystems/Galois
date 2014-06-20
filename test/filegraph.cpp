#include "Galois/Graph/FileGraph.h"
#include "Galois/Runtime/ll/gio.h"

int main(int argc, char** argv) {
  GALOIS_ASSERT(argc > 1);
  Galois::Graph::FileGraph g;

  g.structureFromFile(argv[1]);
  auto numNodes = g.size();
  auto numEdges = g.sizeEdges();
  for (auto n : g) {
    numNodes -= 1;
    numEdges -= std::distance(g.edge_begin(n), g.edge_end(n));
  }
  GALOIS_ASSERT(numNodes == 0);
  GALOIS_ASSERT(numEdges == 0);
  return 0;
}
