#include <iostream>

#include "Galois/Runtime/OfflineGraph.h"
#include "hGraph.h"

#include <iostream>

struct nd {
  int x;
  double y;
};

struct Syncer {
  static int extract(const nd& i) { return i.x; }
  static void reduce(nd& i, int y) { i.x = std::min(i.x, y); }
  static void reset(nd& i) { i.x = 0; }
  typedef int ValTy;
};

int main(int argc, char** argv) {
  try {
    OfflineGraph g(argv[1]);
    std::cout << g.size() << " " << g.sizeEdges() << "\n";
    for (auto N : g)
      if (N % (1024*128) == 0)
        std::cout << N << " " << *g.edge_begin(N) << " " << *g.edge_end(N) << " " << std::distance(g.edge_begin(N), g.edge_end(N)) << "\n";
    hGraph<nd, void> hg(argv[1], 0, 4);

    hg.sync_push<Syncer>();

    return 0;
  } catch(const char* c) {
    std::cerr << "ERROR: " << c << "\n";
    return 1;
  }
}

