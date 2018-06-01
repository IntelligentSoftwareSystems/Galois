// This example shows 
// 0. reading in a graph from a file
// 1. serial iteration over nodes 
// 2. do_all iteration over nodes
// 3. access to node and edge data
// 4. usage of galois::StatTimer
// 5. how to change # of threads
#include "galois/Galois.h"
#include "galois/graphs/LCGraph.h"
#include "galois/Timer.h"
#include <iostream>

using Graph = galois::graphs::LC_CSR_Graph<int, int>;
using GNode = Graph::GraphNode;

int main(int argc, char *argv[]) {
  galois::SharedMemSys G;

  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " filename num_threads" << std::endl;
    return 1;
  }

  Graph g;
  galois::graphs::readGraph(g, argv[1]);        // argv[1] is the file name for graph
  galois::setActiveThreads(std::atoi(argv[2])); // argv[2] is # of threads

  //******************************************************************************************
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

  //*****************************************************************************************
  // parallel traversal over a graph using galois::do_all w/o work stealing
  // 1. operator is specified using lambda expression
  // 2. do_all is named "sum_in_do_all_with_lambda" to show stat after this program finishes
  //! [Graph traversal in pull using do_all]
  galois::do_all(
      galois::iterate(g.begin(), g.end()),             // range
      [&] (GNode n) {                                  // operator
        auto& sum = g.getData(n);
        sum = 0;
        for (auto e: g.edges(n)) {
          sum += g.getEdgeData(e);
        }
      }
      , galois::loopname("sum_in_do_all_with_lambda")  // options
  );
  //! [Graph traversal in pull using do_all]

  return 0;
}
