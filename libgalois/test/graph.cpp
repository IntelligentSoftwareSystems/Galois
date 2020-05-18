/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
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

#include "galois/graphs/Graph.h"
#include <string>

int useGraph(std::string inputfile) {
  //! [Using a graph]
  typedef galois::graphs::LC_CSR_Graph<int, int> Graph;

  // Create graph
  Graph g;
  galois::graphs::readGraph(g, inputfile);

  int sum = 0;

  // Traverse graph
  for (Graph::iterator ii = g.begin(), ei = g.end(); ii != ei; ++ii) {
    Graph::GraphNode src = *ii;
    for (Graph::edge_iterator jj = g.edge_begin(src), ej = g.edge_end(src);
         jj != ej; ++jj) {
      Graph::GraphNode dst = g.getEdgeDst(jj);
      int edgeData         = g.getEdgeData(jj);
      int nodeData         = g.getData(dst);
      sum += edgeData * nodeData;
    }
  }
  //! [Using a graph]

  return sum;
}

int useGraphCxx11(std::string inputfile) {
  //! [Using a graph cxx11]
  typedef galois::graphs::LC_CSR_Graph<int, int> Graph;

  // Create graph
  Graph g;
  galois::graphs::readGraph(g, inputfile);

  int sum = 0;

  // Traverse graph
  for (Graph::GraphNode src : g) {
    for (Graph::edge_iterator edge : g.out_edges(src)) {
      Graph::GraphNode dst = g.getEdgeDst(edge);
      int edgeData         = g.getEdgeData(edge);
      int nodeData         = g.getData(dst);
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
