// This example shows 
// 0. reading in a graph from a file
// 1. serial iteration over nodes 
// 2. for_each iteration over nodes
// 3. access to node and edge data
// 4. usage of galois::StatTimer
// 5. how to change # of threads
// 6. push-style operator using atomic intrinsics in do_all
// 7. push-style operator using atomic intrinsics in for_each w/o conflict detection
#include "galois/Galois.h"
#include "galois/graphs/LCGraph.h"
#include "galois/Timer.h"
#include <iostream>

using Graph = galois::graphs::LC_CSR_Graph<int, int>;
using GNode = Graph::GraphNode;

void initialize(Graph& g) {
  galois::do_all(
      galois::iterate(g.begin(), g.end()), // range
      [&] (GNode n) { g.getData(n) = 0; }  // operator
  );
};

int main(int argc, char *argv[]) {
  galois::SharedMemSys G;

  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " filename num_threads" << std::endl;
    return 1;
  }

  Graph g;
  galois::graphs::readGraph(g, argv[1]);        // argv[1] is the file name for graph
  galois::setActiveThreads(std::atoi(argv[2])); // argv[2] is # of threads

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

  //******************************************************
  // parallel traversal over a graph using galois::for_each
  // 1. push operator is specified using lambda expression
  // 2. for_each is named "sum_in_for_each_with_push_operator" to show stat after this program finishes
  initialize(g);
  galois::for_each(
      galois::iterate(g.begin(), g.end()),                      // range
      [&] (GNode n, auto& ctx) {                                // operator
        for (auto e: g.edges(n)) {                              // cautious point
          auto dst = g.getEdgeDst(e);
          g.getData(dst) += g.getEdgeData(e);
        }
      }
      , galois::loopname("sum_in_for_each_with_push_operator")  // options
  );

  // define lambda expression as a varible for reuse
  auto sumEdgeWeightsAtomically = [&] (GNode n) {
    for (auto e: g.edges(n)) {
      auto dst = g.getEdgeDst(e);
      auto& dstData = g.getData(dst);
      auto edgeWeight = g.getEdgeData(e);
      __sync_fetch_and_add(&dstData, edgeWeight);
    }
  };

  //******************************************************
  // parallel traversal over a graph using galois::do_all w/o work stealing
  // 1. push operator uses atomic intrinsic
  // 2. do_all is named "sum_in_do_all_with_push_atomic" to show stat after this program finishes
  initialize(g);
  galois::do_all(
      galois::iterate(g.begin(), g.end()),                      // range
      sumEdgeWeightsAtomically                                  // operator
      , galois::loopname("sum_in_do_all_with_push_atomic")      // options
  );

  //******************************************************
  // parallel traversal over a graph using galois::for_each
  // 1. push operator uses atomic intrinsic
  // 2. for_each is named "sum_in_do_for_each_with_push_atomic" to show stat after this program finishes
  initialize(g);
  galois::for_each(
      galois::iterate(g.begin(), g.end()),                      // range
      [&] (GNode n, auto& ctx) { sumEdgeWeightsAtomically(n); } // operator
      , galois::loopname("sum_in_for_each_with_push_atomic")    // options
      , galois::no_pushes()
      , galois::no_conflicts()
  );

  return 0;
}
