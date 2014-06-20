#include "Galois/Graph/Graph.h"

void useGraph() {
  //! [Using a graph]
  typedef Galois::Graph::LC_CSR_Graph<int,int> Graph;
  
  // Create graph
  Graph g;
  Galois::Graph::readGraph(g, inputfile);
  
  // Traverse graph
  for (Graph::iterator ii = g.begin(), ei = g.end(); ii != ei; ++ii) {
    Graph::GraphNode src = *ii;
    for (Graph::edge_iterator jj = g.edge_begin(src), ej = g.edge_end(src); jj != ej; ++jj) {
      Graph::GraphNode dst = g.getEdgeDst(jj);
      int edgeData = g.getEdgeData(jj);
      int nodeData = g.getData(dst);
    }
  }
  //! [Using a graph]
}


void useGraphCxx11() {
  //! [Using a graph cxx11] 
  typedef Galois::Graph::LC_CSR_Graph<int,int> Graph;
  
  // Create graph
  Graph g;
  Galois::Graph::readGraph(g, inputfile);
  
  // Traverse graph
  for (Graph::GraphNode src : g) {
    for (Graph::edge_iterator edge : g.out_edges(src)) {
      Graph::GraphNode dst = g.getEdgeDst(edge);
      int edgeData = g.getEdgeData(edge);
      int nodeData = g.getData(dst);
    }
  }
  //! [Using a graph cxx11] 
}


int main() {
  useGraph();
  useGraphCxx11();
  return 0;
}

