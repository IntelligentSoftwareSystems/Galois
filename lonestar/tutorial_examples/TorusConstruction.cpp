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

// This example shows how to manipulate MorphGraph to change graph topology
// 1. createNode
// 2. addNode
// 3. addEdge
#include "galois/Galois.h"
#include "galois/graphs/Graph.h"
#include <iostream>

//! [Define a MorphGraph]
// Graph has int node data, void edge data and is directed
using Graph = galois::graphs::MorphGraph<int, void, true>;
// Opaque pointer to graph node
using GNode = Graph::GraphNode;
//! [Define a MorphGraph]

//! [Construct torus]
void constructTorus(Graph& g, int height, int width) {
  // Construct set of nodes
  int numNodes = height * width;
  std::vector<GNode> nodes(numNodes);
  for (int i = 0; i < numNodes; ++i) {
    GNode n = g.createNode(
        0);       // allocate node data and initialize the node data with 0
    g.addNode(n); // add n to g. from now on n can be located from g
    nodes[i] = n;
  }

  // Add edges
  for (int x = 0; x < width; ++x) {
    for (int y = 0; y < height; ++y) {
      GNode c = nodes[x * height + y];
      GNode n = nodes[x * height + ((y + 1) % height)];
      GNode s = nodes[x * height + ((y - 1 + height) % height)];
      GNode e = nodes[((x + 1) % width) * height + y];
      GNode w = nodes[((x - 1 + width) % width) * height + y];
      g.addEdge(c, n); // addEdge checks if the edge exists or not. nop if so.
      g.addEdge(c, s);
      g.addEdge(c, e);
      g.addEdge(c, w);
    }
  }
}
//! [Construct torus]

int main(int argc, char** argv) {
  galois::SharedMemSys G;

  if (argc < 2) {
    std::cerr << "<sqrt grid size>\n";
    return 1;
  }
  int N = atoi(argv[1]);

  Graph graph;
  constructTorus(graph, N, N);

  std::cout << "Constructed a " << N << " x " << N << " torus." << std::endl;

  return 0;
}
