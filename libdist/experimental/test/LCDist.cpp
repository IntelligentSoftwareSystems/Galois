#include "galois/Galois.h"
#include "galois/Graphs/LC_Dist_Graph.h"

#include <boost/iterator/counting_iterator.hpp>
#include <vector>
#include <iostream>

typedef galois::graphs::LC_Dist<std::pair<int, int>, int> Graph;
typedef Graph::GraphNode GNode;

struct AddSelfLoop {
  typedef int tt_is_copyable;

  Graph::pointer graph;

  AddSelfLoop(Graph::pointer g): graph(g) { }
  AddSelfLoop() { }

  template<typename T, typename Context>
  void operator()(const T& n, const Context&) {
    galois::runtime::acquire(n, galois::MethodFlag::ALL);
    graph->addEdge(n, n, graph->at(n).first);
  }
};

void testSerialAdd(int N) {
  int numEdges = 1;
  std::vector<unsigned> counts(N, numEdges);
  Graph::pointer g = Graph::allocate(counts);

  for (int i = 0; i < N; ++i) {
    Graph::GraphNode n = g->begin()[i];
    g->at(n) = std::make_pair(i, i);
  }

  galois::for_each_local(g, AddSelfLoop(g));

  GALOIS_ASSERT(std::distance(g->begin(), g->end()) == N);
  for (auto nn : *g) {
    GALOIS_ASSERT(std::distance(g->edge_begin(nn), g->edge_end(nn)) == 1);
    for (auto jj = g->edge_begin(nn), ej = g->edge_end(nn); jj != ej; ++jj)
      GALOIS_ASSERT(g->at(nn).first == g->at(jj));
  }

  Graph::deallocate(g);
}

struct AddNode {
  typedef int tt_is_copyable;

  Graph::pointer graph;

  AddNode(Graph::pointer g): graph(g) { }
  AddNode() { }

  void operator()(const int& i, const galois::UserContext<int>&) {
    Graph::GraphNode n = graph->begin()[i];
    graph->at(n) = std::make_pair(i, i);
    graph->addEdge(n, n, graph->at(n).first);
  }
};

void testParallelAdd(int N) {
  int numEdges = 1;
  std::vector<unsigned> counts(N, numEdges);
  Graph::pointer g = Graph::allocate(counts);

  galois::for_each(boost::counting_iterator<int>(0), boost::counting_iterator<int>(N), AddNode(g));

  GALOIS_ASSERT(std::distance(g->begin(), g->end()) == N);
  for (auto nn : *g) {
    GALOIS_ASSERT(std::distance(g->edge_begin(nn), g->edge_end(nn)) == 1);
    for (auto jj = g->edge_begin(nn), ej = g->edge_end(nn); jj != ej; ++jj)
      GALOIS_ASSERT(g->at(nn).first == g->at(jj));
  }

  Graph::deallocate(g);
}

struct Grid {
  typedef int tt_is_copyable;

  Graph::pointer graph;
  int N;

  Grid(Graph::pointer g, int N): graph(g), N(N) { }
  Grid() { }

  Graph::GraphNode getNode(int i, int j) {
    if (i == N) i = 0;
    if (j == N) j = 0;
    if (i == -1) i = N - 1;
    if (j == -1) j = N - 1;

    return graph->begin()[i*N+j];
  }

  void operator()(const Graph::GraphNode& n, galois::UserContext<Graph::GraphNode>& ctx) {
    int x = graph->at(n).first;
    int i = x / N;
    int j = x % N;

    Graph::GraphNode c = getNode(i, j);
    GALOIS_ASSERT(c == n);
    Graph::GraphNode c1 = getNode(i, j+1);
    Graph::GraphNode c2 = getNode(i, j-1);
    Graph::GraphNode c3 = getNode(i+1, j);
    Graph::GraphNode c4 = getNode(i-1, j);
    if (std::distance(graph->edge_begin(c), graph->edge_end(c)) == 1) {
      graph->addEdge(c, c1, graph->at(c).first);
      graph->addEdge(c, c2, graph->at(c).first);
      graph->addEdge(c, c3, graph->at(c).first);
      graph->addEdge(c, c4, graph->at(c).first);
    }
    int m = std::numeric_limits<int>::max();
    for (auto vv = graph->edge_begin(c), ev = graph->edge_end(c); vv != ev; ++vv) {
      Graph::GraphNode dd = graph->dst(vv);
      m = std::min(m, graph->at(dd).second);
    }
    for (auto vv = graph->edge_begin(c), ev = graph->edge_end(c); vv != ev; ++vv) {
      Graph::GraphNode dd = graph->dst(vv);
      if (graph->at(dd).second == m)
        continue;

      graph->at(dd).second = m;
      ctx.push(dd);
    }
  }
};

void testGrid(int N) {
  static const bool printGraph = false;
  int numEdges = 5;
  std::vector<unsigned> counts(N*N, numEdges);
  Graph::pointer g = Graph::allocate(counts);

  GALOIS_ASSERT(N > 0);

  galois::for_each(boost::counting_iterator<int>(0), boost::counting_iterator<int>(N*N), AddNode(g));
  galois::for_each_local(g, Grid(g, N));
  
  if (!printGraph)
    GALOIS_ASSERT(std::distance(g->begin(), g->end()) == N * N);
  for (auto nn : *g) {
    if (printGraph) {
      std::cout << std::distance(g->edge_begin(nn), g->edge_end(nn)) << " ";
      std::cout << "(" << g->at(nn).first << " " << g->at(nn).second << ") ";
    } else {
      GALOIS_ASSERT(std::distance(g->edge_begin(nn), g->edge_end(nn)) == numEdges);
      GALOIS_ASSERT(g->at(nn).second == 0);
    }
    for (auto jj = g->edge_begin(nn), ej = g->edge_end(nn); jj != ej; ++jj) {
      if (printGraph) {
        std::cout << "(" << g->at(g->dst(jj)).first << " " << g->at(g->dst(jj)).second << " " << g->at(jj) << ") ";
      } else {
        GALOIS_ASSERT(g->at(nn).first == g->at(jj));
        GALOIS_ASSERT(g->at(g->dst(jj)).second == 0);
      }
    }
    if (printGraph)
      std::cout << "\n";
  }

  Graph::deallocate(g);
}

int main(int argc, char *argv[])
{
  galois::StatManager M;

  int threads = 2;
  if (argc > 1)
    threads = atoi(argv[1]);
  int N = 10;
  if (argc > 2)
    N = atoi(argv[2]);
  
  galois::setActiveThreads(threads);
  auto& net = galois::runtime::getSystemNetworkInterface();
  net.start();

  testSerialAdd(N);
  testParallelAdd(N);
  testGrid(N);

  net.terminate();
  return 0;
};
