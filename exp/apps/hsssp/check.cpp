#include <iostream>

#include "OfflineGraph.h"
#include "hGraph.h"

#include <iostream>

int main(int argc, char** argv) {
  try {
    OfflineGraph g(argv[1]);
    std::cout << g.size() << " " << g.sizeEdges() << "\n";
    for (auto N : g)
      if (N % (1024*128) == 0)
        std::cout << N << " " << *g.edge_begin(N) << " " << *g.edge_end(N) << " " << std::distance(g.edge_begin(N), g.edge_end(N)) << "\n";
    hGraph<int, void> hg(argv[1], 0, 4);
    return 0;
  } catch(const char* c) {
    std::cerr << "ERROR: " << c << "\n";
    return 1;
  }
}
