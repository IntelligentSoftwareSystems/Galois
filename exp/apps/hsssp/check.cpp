#include "OfflineGraph.h"

#include <iostream>

int main(int argc, char** argv) {
  try {
    OfflineGraph g(argv[1]);
    std::cout << g.size() << " " << g.sizeEdges() << "\n";
    for (auto N : g)
      std::cout << N << " " << *g.edge_begin(N) << " " << *g.edge_end(N) << " " << std::distance(g.edge_begin(N), g.edge_end(N)) << "\n";
    return 0;
  } catch(const char* c) {
    std::cerr << "ERROR: " << c << "\n";
    return 1;
  }
}
