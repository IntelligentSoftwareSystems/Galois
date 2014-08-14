#include "Galois/Galois.h"
#include "Galois/Statistic.h"
#include "Galois/Graphs/Graph3.h"

#include <boost/iterator/counting_iterator.hpp>
#include <iostream>

using namespace Galois::Graph;
typedef ThirdGraph<int, int, EdgeDirection::Out> Graph;
typedef ThirdGraph<int, void, EdgeDirection::Un> UndirectedGraph;

struct AddSelfLoop {
  template<typename T, typename Context>
  void operator()(const T& node, const Context&) {
    node->createEdge(node, node, node->getData());
  }
};

struct AddNode {
  typedef int tt_is_copyable;

  Graph::pointer graph;

  AddNode(Graph::pointer g): graph(g) { }
  AddNode() { }

  template<typename Context>
  void operator()(const int& x, const Context&) {
    Graph::NodeHandle node = graph->createNode(x);
    node->createEdge(node, node, node->getData());
    graph->addNode(node);
  }
};

struct AddRemoveNode {
  typedef int tt_is_copyable;

  UndirectedGraph::pointer graph;

  AddRemoveNode(UndirectedGraph::pointer g): graph(g) {}
  AddRemoveNode() {}

  template<typename Context>
  void operator()(const int& x, const Context&) {
    UndirectedGraph::NodeHandle node1 = graph->createNode(x*2);
    UndirectedGraph::NodeHandle node2 = graph->createNode((x*2)+1);
    graph->addNode(node1);
    graph->addNode(node2);
    graph->addEdge(node1, node2);
    if (x & 1) {
      graph->removeNode(node1);
      GALOIS_ASSERT(!graph->containsNode(node1),
        "Node: ", graph->getData(node1), " not removed");
      GALOIS_ASSERT(std::distance(graph->edge_begin(node2), graph->edge_end(node2)) == 0);
    }
  }
};

void testSerialAdd(int N) {
  Graph::pointer g = Graph::allocate();

  for (int x = 0; x < N; ++x)
    g->addNode(g->createNode(x));

  Galois::for_each(g->begin(), g->end(), AddSelfLoop());

  GALOIS_ASSERT(std::distance(g->begin(), g->end()) == N);
  for (auto nn : *g) {
    GALOIS_ASSERT(std::distance(nn->begin(), nn->end()) == 1);
    for (auto jj = nn->begin(), ej = nn->end(); jj != ej; ++jj)
      GALOIS_ASSERT(nn->getData() == jj->getValue());
  }

  Graph::deallocate(g);
}

void testParallelAdd(int N) {
  Graph::pointer g = Graph::allocate();

  Galois::for_each(boost::counting_iterator<int>(0), boost::counting_iterator<int>(N), AddNode(g));

  GALOIS_ASSERT(std::distance(g->begin(), g->end()) == N);
  for (auto nn : *g) {
    GALOIS_ASSERT(std::distance(g->edge_begin(nn), g->edge_end(nn)) == 1);
    for (auto jj = g->edge_begin(nn), ej = g->edge_end(nn); jj != ej; ++jj)
      GALOIS_ASSERT(g->getData(nn) == g->getEdgeData(jj));
  }

  Graph::deallocate(g);
}

void testAddRemove(int N) {
  UndirectedGraph::pointer g = UndirectedGraph::allocate();

  Galois::for_each(boost::counting_iterator<int>(0), boost::counting_iterator<int>(N), AddRemoveNode(g));

  std::cout << std::distance(g->begin(), g->end()) << "\n";
  for (auto nn : *g) {
    ptrdiff_t dist = std::distance(g->edge_begin(nn), g->edge_end(nn));
    std::cout << g->getData(nn) << " " << (g->getData(nn) / 2) << " " << dist << "\n";
  }

  GALOIS_ASSERT(std::distance(g->begin(), g->end()) == (N / 2) * 2 + (N / 2));
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
  Galois::StatManager statManager;
  int threads = 2;
  if (argc > 1)
    threads = atoi(argv[1]);
  int N = 40;
  if (argc > 2)
    N = atoi(argv[2]);

  Galois::setActiveThreads(threads);
  auto& net = Galois::Runtime::getSystemNetworkInterface();
  net.start();
  
  testSerialAdd(N);
  testParallelAdd(N);
  testAddRemove(N);

  net.terminate();

  return 0;
}
