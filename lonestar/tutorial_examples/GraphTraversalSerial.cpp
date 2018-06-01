// This example shows 
// 0. reading in a graph from a file
// 1. serial iteration over nodes 
// 2. access to node and edge data
// 3. usage of galois::StatTimer
#include "galois/Galois.h"
#include "galois/graphs/LCGraph.h"
#include "galois/Timer.h"
#include <iostream>

int main(int argc, char *argv[]) {
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

  //******************************************************
  // serial traversal over a graph
  // sum over nodes and edges in C++11 syntax
  galois::StatTimer T("sum_serial");
  T.start();
  //! [Graph traversal]
  // iterate over nodes
  for (auto n: g) {
    auto& sum = g.getData(n); // get node data of n
    sum = 0;
    // iterate over edges from node n
    for (auto e: g.edges(n)) {
      sum += g.getEdgeData(e); // get edge data of e
    }
  }
  //! [Graph traversal]
  T.stop();

  return 0;
}
