/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
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

#include <iostream>

#include "galois/graphs/OfflineGraph.h"
#include "DistGraph.h"

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
      if (N % (1024 * 128) == 0)
        std::cout << N << " " << *g.edge_begin(N) << " " << *g.edge_end(N)
                  << " " << std::distance(g.edge_begin(N), g.edge_end(N))
                  << "\n";
    DistGraph<nd, void> hg(argv[1], 0, 4);

    hg.sync_push<Syncer>();

    return 0;
  } catch (const char* c) {
    std::cerr << "ERROR: " << c << "\n";
    return 1;
  }
}
