/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2024, The University of Texas at Austin. All rights reserved.
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
#include <queue>

#include "galois/Galois.h"
#include "galois/graphs/LS_LC_CSR_Graph.h"

template <typename GraphTy>
void check() {
  GraphTy g(4);

  g.addEdgesTopologyOnly(0, {1, 2});
  g.addEdgesTopologyOnly(1, {2, 3});
  g.addEdgesTopologyOnly(2, {3});

  auto print_graph = [&g](std::string_view msg) {
    std::cout << "- " << msg << " -" << std::endl;
    for (auto src : g.vertices()) {
      for (auto dst : g.edges(src)) {
        std::cout << src << "->" << dst << std::endl;
      }
    }
  };

  print_graph("initial graph");

  g.addEdgesTopologyOnly(2, {1});

  print_graph("added 2->1");

  g.deleteEdges(1, {3});

  print_graph("deleted 1->3");

  g.compact();

  print_graph("compacted");

  std::cout << std::endl << std::endl;
}

int main() {
  galois::SharedMemSys Galois_runtime;
  check<galois::graphs::LS_LC_CSR_Graph<>>();

  return 0;
}
