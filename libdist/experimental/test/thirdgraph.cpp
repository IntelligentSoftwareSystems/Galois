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

#include "galois/Galois.h"
#include "galois/Timer.h"
#include "galois/graphs/Graph3.h"

#include <boost/iterator/counting_iterator.hpp>
#include <iostream>

using namespace galois::graphs;
typedef ThirdGraph<int, int, EdgeDirection::Out> Graph;
typedef ThirdGraph<int, void, EdgeDirection::Un> UndirectedGraph;

struct AddSelfLoop {
  template <typename T, typename Context>
  void operator()(const T& node, const Context&) {
    galois::runtime::acquire(node, galois::MethodFlag::ALL);
    GALOIS_ASSERT(&*node);
    node->createEdge(node, node, node->getData());
  }
};

void testSerialAdd(int N) {
  Graph::pointer g = Graph::allocate();

  for (int x = 0; x < N; ++x)
    g->addNode(g->createNode(x));

  galois::for_each(g->begin(), g->end(), AddSelfLoop());

  GALOIS_ASSERT(std::distance(g->begin(), g->end()) == N);
  for (auto nn : *g) {
    GALOIS_ASSERT(std::distance(g->edge_begin(nn), g->edge_end(nn)) == 1);
    for (auto jj = g->edge_begin(nn), ej = g->edge_end(nn); jj != ej; ++jj)
      GALOIS_ASSERT(g->getData(nn) == g->getEdgeData(jj));
  }

  Graph::deallocate(g);
}

struct AddNode {
  typedef int tt_is_copyable;

  Graph::pointer graph;

  AddNode(Graph::pointer g) : graph(g) {}
  AddNode() {}

  template <typename Context>
  void operator()(const int& x, const Context&) {
    Graph::NodeHandle node = graph->createNode(x);
    node->createEdge(node, node, node->getData());
    graph->addNode(node);
  }
};

void testParallelAdd(int N) {
  Graph::pointer g = Graph::allocate();

  galois::for_each(boost::counting_iterator<int>(0),
                   boost::counting_iterator<int>(N), AddNode(g));

  GALOIS_ASSERT(std::distance(g->begin(), g->end()) == N);
  for (auto nn : *g) {
    GALOIS_ASSERT(std::distance(g->edge_begin(nn), g->edge_end(nn)) == 1);
    for (auto jj = g->edge_begin(nn), ej = g->edge_end(nn); jj != ej; ++jj)
      GALOIS_ASSERT(g->getData(nn) == g->getEdgeData(jj));
  }

  Graph::deallocate(g);
}

struct AddRemoveNode {
  typedef int tt_is_copyable;

  UndirectedGraph::pointer graph;

  AddRemoveNode(UndirectedGraph::pointer g) : graph(g) {}
  AddRemoveNode() {}

  template <typename Context>
  void operator()(const int& x, const Context&) {
    UndirectedGraph::NodeHandle node1 = graph->createNode(x * 2);
    UndirectedGraph::NodeHandle node2 = graph->createNode((x * 2) + 1);
    graph->addNode(node1);
    graph->addNode(node2);
    graph->addEdge(node1, node2);
    if (x & 1) {
      graph->removeNode(node1);
      GALOIS_ASSERT(!graph->containsNode(node1),
                    "Node: ", graph->getData(node1), " not removed");
      GALOIS_ASSERT(
          std::distance(graph->edge_begin(node2), graph->edge_end(node2)) == 0);
    }
  }
};

void testAddRemove(int N) {
  UndirectedGraph::pointer g = UndirectedGraph::allocate();

  galois::for_each(boost::counting_iterator<int>(0),
                   boost::counting_iterator<int>(N), AddRemoveNode(g));

  ptrdiff_t dist = std::distance(g->begin(), g->end());
  int expected   = ((N + 1) / 2) * 2 + (N / 2);
  GALOIS_ASSERT(dist == expected, ": ", dist, " != ", expected);

  for (auto nn : *g) {
    ptrdiff_t dist = std::distance(g->edge_begin(nn), g->edge_end(nn));
    if ((g->getData(nn) / 2) & 1)
      GALOIS_ASSERT(dist == 0, ": node(", g->getData(nn), ") ", dist, " != 0");
    else
      GALOIS_ASSERT(dist == 1, ": node(", g->getData(nn), ") ", dist, " != 1");
  }
  UndirectedGraph::deallocate(g);
}

int main(int argc, char** argv) {
  galois::StatManager statManager;
  int threads = 2;
  if (argc > 1)
    threads = atoi(argv[1]);
  int N = 40;
  if (argc > 2)
    N = atoi(argv[2]);

  galois::setActiveThreads(threads);
  auto& net = galois::runtime::getSystemNetworkInterface();
  net.start();

  testSerialAdd(N);
  testParallelAdd(N);
  testAddRemove(N);

  net.terminate();

  return 0;
}
