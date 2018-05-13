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

  using Graph = galois::graphs::LC_CSR_Graph<int, int>;

  Graph g;
  galois::graphs::readGraph(g, argv[1]); // argv[1] is the file name for graph

  //******************************************************
  // serial traversal over a graph
  // sum over nodes and edges in C++11 syntax
  galois::StatTimer T("sum_serial");
  T.start();
  for (auto n: g) {
    auto& sum = g.getData(n);
    sum = 0;
    for (auto e: g.edges(n)) {
      sum += g.getEdgeData(e);
    }
  }
  T.stop();

  return 0;
}
