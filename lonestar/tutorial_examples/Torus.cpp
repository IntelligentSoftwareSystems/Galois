/**
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of XYZ License (a copy is located in
 * LICENSE.txt at the top-level directory).
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

#include "galois/Galois.h"
#include "galois/Timer.h"
#include "galois/graphs/Graph.h"
#include <iostream>

//! Graph has int node data, void edge data and is directed
//! [define a graph]
typedef galois::graphs::MorphGraph<int,void,true> Graph;
//! [define a graph]
//! Opaque pointer to graph node
typedef Graph::GraphNode GNode;

//! Construct a simple torus graph
void constructTorus(Graph& g, int height, int width) {
  // Construct set of nodes
  int numNodes = height * width;
  std::vector<GNode> nodes(numNodes);
  for (int i = 0; i < numNodes; ++i) {
    //! [create and add node]
    GNode n = g.createNode(0);
    g.addNode(n);
    nodes[i] = n;
    //! [create and add node]
  }

  //! [add edges]
  for (int x = 0; x < width; ++x) {
    for (int y = 0; y < height; ++y) {
      GNode c = nodes[x*height + y];
      GNode n = nodes[x*height + ((y+1) % height)];
      GNode s = nodes[x*height + ((y-1+height) % height)];
      GNode e = nodes[((x+1) % width)*height + y];
      GNode w = nodes[((x-1+width) % width)*height + y];
      g.addEdge(c, n);
      g.addEdge(c, s);
      g.addEdge(c, e);
      g.addEdge(c, w);
    }
  }
  //! [add edges]
}

void verify(Graph& graph, int n) {
  // Verify
  int count = std::count_if(graph.begin(), graph.end(), [&] (GNode n) -> bool { return graph.getData(n) == 4; });
  if (count != n * n) {
    std::cerr << "Expected " << n * n << " nodes with value = 4 but found " << count << " instead.\n";
  } else {
    std::cout << "Correct!\n";
  }
}

void initialize(Graph& graph) {
  galois::do_all(
      galois::iterate(graph),
      [&] (GNode n) { graph.getData(n) = 0; }
  );
}

int main(int argc, char** argv) {
  galois::SharedMemSys G;

  if (argc < 3) {
    std::cerr << "<num threads> <sqrt grid size>\n";
    return 1;
  }
  unsigned int numThreads = atoi(argv[1]);
  int N = atoi(argv[2]);

  numThreads = galois::setActiveThreads(numThreads);
  std::cout << "Using " << numThreads << " thread(s) and " << N << " x " << N << " torus\n";

  Graph graph;
  constructTorus(graph, N, N);

  // read/write only a node itself
  galois::do_all(
      galois::iterate(graph),
      [&] (GNode n) {
        graph.getData(n) = std::distance(graph.edge_begin(n), graph.edge_end(n));
      }
      , galois::loopname("do_all")
  );
  verify(graph, N);

  // push operator with Galois synchronization
  initialize(graph);
  galois::for_each(
      galois::iterate(graph),
      [&] (GNode n, auto& ctx) {
        for (auto ii: graph.edges(n)) {
          GNode dst = graph.getEdgeDst(ii);
          auto& data = graph.getData(dst);
          data += 1;
        }
      }
      , galois::loopname("for_each")
      , galois::no_pushes()
  );
  verify(graph, N);

  auto incrementNeighborsAtomically = [&] (GNode n) {
    for (auto e: graph.edges(n)) {
      auto dst = graph.getEdgeDst(e);
      auto& dstData = graph.getData(dst);
      __sync_fetch_and_add(&dstData, 1);
    }
  };

  // push operator with self synchronization in do_all
  initialize(graph);
  galois::do_all(
      galois::iterate(graph),
      incrementNeighborsAtomically
      , galois::loopname("do_all_self_sync")
      , galois::steal()
      , galois::chunk_size<32>()
  );
  verify(graph, N);

  // push operator with self synchronization in optimized for_each
  initialize(graph); 
  galois::for_each(
      galois::iterate(graph),
      [&] (GNode n, auto& ctx) { incrementNeighborsAtomically(n); }
      , galois::loopname("for_each_self_sync")
      , galois::no_conflicts()
      , galois::no_pushes()
  );
  verify(graph, N);

  return 0;
}
