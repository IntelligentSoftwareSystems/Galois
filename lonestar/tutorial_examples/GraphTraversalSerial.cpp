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

// This example shows
// 0. reading in a graph from a file
// 1. serial iteration over nodes
// 2. access to node and edge data
// 3. usage of galois::StatTimer
#include "galois/Galois.h"
#include "galois/graphs/LCGraph.h"
#include "galois/Timer.h"
#include <iostream>

int main(int argc, char* argv[]) {
  galois::SharedMemSys G;

  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " filename" << std::endl;
    return 1;
  }

  //! [Define LC_CSR_Graph]
  // An LC_CSR_Graph whose node data type is int and edge data type is int
  using Graph = galois::graphs::LC_CSR_Graph<int, int>;
  //! [Define LC_CSR_Graph]

  //! [Read a graph]
  Graph g;
  galois::graphs::readGraph(g, argv[1]); // argv[1] is the file name for graph
  //! [Read a graph]

  //! [use of a StatTimer]
  //******************************************************
  // serial traversal over a graph
  // sum over nodes and edges in C++11 syntax
  galois::StatTimer T("sum_serial");
  T.start();
  //! [Graph traversal]
  // iterate over nodes
  for (auto n : g) {
    auto& sum = g.getData(n); // get node data of n
    sum       = 0;
    // iterate over edges from node n
    for (auto e : g.edges(n)) {
      sum += g.getEdgeData(e); // get edge data of e
    }
  }
  //! [Graph traversal]
  T.stop();
  //! [use of a StatTimer]

  return 0;
}
