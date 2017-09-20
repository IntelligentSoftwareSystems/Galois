/** Tutorial torus application -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
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
 *
 * @section Description
 *
 * Simple tutorial application. Creates a torus graph and each node increments
 * its neighbors data by one. This version improves {@link Torus.cpp} by
 * allocating the graph in parallel, which distributes the graph, and by
 * more carefully assigning work to threads.
 *
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#include "galois/Galois.h"
#include "galois/Timer.h"
#include "galois/graphs/Graph.h"
#include <iostream>

//! Graph has int node data, void edge data and is directed
typedef galois::graphs::FirstGraph<int,void,true> Graph;
//! Opaque pointer to graph node
typedef Graph::GraphNode GNode;

//! Increments node value of each neighbor by 1
struct IncrementNeighbors {
  Graph& g;
  IncrementNeighbors(Graph& g): g(g) { }

  //! Operator. Context parameter is unused in this example.
  void operator()(GNode n, galois::UserContext<GNode>& ctx) {
    // For each outgoing edge (n, dst)
    //for (auto ii : g.edges(n)) {
    for (Graph::edge_iterator ii = g.edge_begin(n), ei = g.edge_end(n); ii != ei; ++ii) {
      GNode dst = g.getEdgeDst(ii);
      int& data = g.getData(dst);
      // Increment node data by 1
      data += 1;
    }
  }
};

//! Returns true if node value equals v
struct ValueEqual {
  Graph& g;
  int v;
  ValueEqual(Graph& g, int v): g(g), v(v) { }
  bool operator()(GNode n) {
    return g.getData(n) == v;
  }
};

class Point2D {
  int v[2];
public:
  Point2D(): v { 0, 0 } { }
  Point2D(int x, int y): v { x, y } { }

  const int& at(int i) const { return v[i]; }
  const int& x() const { return v[0]; }
  const int& y() const { return v[1]; }
  int dim() const { return 2; }
};

/**
 * Sort pairs according to Morton Z-Order.
 *
 * From http://en.wikipedia.org/wiki/Z-order_%28curve%29
 */
struct ZOrderCompare {
  bool operator()(const Point2D& p1, const Point2D& p2) const {
    int index = 0;
    int x = 0;
    for (int k = 0; k < p1.dim(); ++k) {
      int y = p1.at(k) ^ p2.at(k);
      if (lessMsb(x, y)) {
        index = k;
        x = y;
      }
    }
    return p1.at(index) - p2.at(index) < 0;
  }

  bool lessMsb(int a, int b) const {
    return a < b && a < (a ^ b);
  }
};

struct CreateNodes {
  Graph& g;
  std::vector<GNode>& nodes;
  int height;
  CreateNodes(Graph& g, std::vector<GNode>& n, int h): g(g), nodes(n), height(h) { }

  void operator()(const Point2D& p) const {
    GNode n = g.createNode(0);
    g.addNode(n);
    nodes[p.x() * height + p.y()] = n;
  }
};

//! Construct a simple torus graph
void constructTorus(Graph& g, int height, int width) {
  // Construct set of nodes
  int numNodes = height * width;
  std::vector<Point2D> points(numNodes);
  for (int x = 0; x < width; ++x) {
    for (int y = 0; y < height; ++y) {
      points[x*height + y] = Point2D(x, y);
    }
  }
  // Sort in a space-filling way
  std::sort(points.begin(), points.end(), ZOrderCompare());

  // Using space-filling order, assign nodes and create (and allocate) them in parallel
  std::vector<GNode> nodes(numNodes);
  galois::do_all(points.begin(), points.end(), CreateNodes(g, nodes, height));

  // Add edges
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
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "<num threads> <sqrt grid size>\n";
    return 1;
  }
  unsigned int numThreads = atoi(argv[1]);
  int n = atoi(argv[2]);

  GALOIS_ASSERT(n > 2);

  numThreads = galois::setActiveThreads(numThreads);
  std::cout << "Using " << numThreads << " threads and " << n << " x " << n << " torus\n";

  Graph graph;
  constructTorus(graph, n, n);

  galois::Timer T;
  T.start();
  // Unlike galois::for_each, galois::for_each_local initially assigns work
  // based on which thread created each node (galois::for_each uses a simple
  // blocking of the iterator range to initialize work, but the iterator order
  // of a Graph is implementation-defined). 
  galois::for_each_local(graph, IncrementNeighbors(graph));
  T.stop();

  std::cout << "Elapsed time: " << T.get() << " milliseconds\n";

  // Verify
  int count = std::count_if(graph.begin(), graph.end(), ValueEqual(graph, 4));
  if (count != n * n) {
    std::cerr << "Expected " << n * n << " nodes with value = 4 but found " << count << " instead.\n";
    return 1;
  } else {
    std::cout << "Correct!\n";
  }

  return 0;
}
