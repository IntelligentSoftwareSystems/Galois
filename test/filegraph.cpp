#include "Galois/Graphs/FileGraph.h"
#include "Galois/gIO.h"

typedef galois::graphs::FileGraph Graph;

void checkGraph(Graph& g) {
  auto numNodes = g.size();
  auto numEdges = g.sizeEdges();
  for (auto n : g) {
    numNodes -= 1;
    numEdges -= std::distance(g.edge_begin(n), g.edge_end(n));
  }
  GALOIS_ASSERT(numNodes == 0);
  GALOIS_ASSERT(numEdges == 0);
}

template<typename Fn>
void testBasic(Graph&& graph, const std::string& filename, Fn fn) {
  Graph g = std::move(graph);
  fn(g, filename);
  checkGraph(g);
}

void testPart(const std::string& filename, int numParts) {
  //! [Reading part of graph]
  typedef galois::graphs::FileGraph Graph;
  Graph g;
  size_t nodeSize = 1; // in bytes
  size_t edgeSize = 1; // in bytes
  g.fromFile(filename);

  std::vector<Graph> partsByNode(numParts);
  Graph::iterator nlast = g.begin();
  for (int i = 0; i < numParts; ++i) {
    auto ranges = g.divideByNode(nodeSize, edgeSize, i, numParts);
    GALOIS_ASSERT(nlast == ranges.first.first, "non-consecutive ranges");
    partsByNode[i].partFromFile(filename, ranges.first, ranges.second);
    checkGraph(partsByNode[i]);   
    nlast = ranges.first.second;
  }
  GALOIS_ASSERT(nlast == g.end(), "sum of partitions != original graph");

  std::vector<Graph> partsByEdge(numParts);
  Graph::edge_iterator elast;
  if (g.begin() != g.end())
    elast = g.edge_begin(*g.begin());
  for (int i = 0; i < numParts; ++i) {
    auto ranges = g.divideByEdge(nodeSize, edgeSize, i, numParts);
    GALOIS_ASSERT(elast == ranges.second.first, "non-consecutive ranges");
    partsByEdge[i].partFromFile(filename, ranges.first, ranges.second);
    checkGraph(partsByEdge[i]);
    elast = ranges.second.second;
  }
  if (g.begin() != g.end())
    GALOIS_ASSERT(elast == g.edge_end(*(g.end() - 1)), "sum of partitions != original graph");

  //! [Reading part of graph]
}

int main(int argc, char** argv) {
  GALOIS_ASSERT(argc > 1);
  testBasic(Graph(), argv[1], [](Graph& g, std::string f) { g.fromFile(f); });
  testBasic(Graph(), argv[1], [](Graph& g, std::string f) { g.fromFileInterleaved<void>(f); });
  testPart(argv[1], 7);

  return 0;
}
