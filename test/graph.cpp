#include "Galois/Graphs/Graph.h"
#include <string>

int useGraph(std::string inputfile) {
  //! [Using a graph]
  typedef galois::graphs::LC_CSR_Graph<int,int> Graph;
  
  // Create graph
  Graph g;
  galois::graphs::readGraph(g, inputfile);
  
  int sum = 0;

  // Traverse graph
  for (Graph::iterator ii = g.begin(), ei = g.end(); ii != ei; ++ii) {
    Graph::GraphNode src = *ii;
    for (Graph::edge_iterator jj = g.edge_begin(src), ej = g.edge_end(src); jj != ej; ++jj) {
      Graph::GraphNode dst = g.getEdgeDst(jj);
      int edgeData = g.getEdgeData(jj);
      int nodeData = g.getData(dst);
      sum += edgeData * nodeData;
    }
  }
  //! [Using a graph]

  return sum;
}


int useGraphCxx11(std::string inputfile) {
  //! [Using a graph cxx11] 
  typedef galois::graphs::LC_CSR_Graph<int,int> Graph;
  
  // Create graph
  Graph g;
  galois::graphs::readGraph(g, inputfile);
  
  int sum = 0;

  // Traverse graph
  for (Graph::GraphNode src : g) {
    for (Graph::edge_iterator edge : g.out_edges(src)) {
      Graph::GraphNode dst = g.getEdgeDst(edge);
      int edgeData = g.getEdgeData(edge);
      int nodeData = g.getData(dst);
      sum += edgeData * nodeData;
    }
  }
  //! [Using a graph cxx11] 

  return sum;
}


int main(int argc, char** argv) {
  if (argc > 1) {
    useGraph(argv[1]);
    useGraphCxx11(argv[1]);
  }
  return 0;
}
