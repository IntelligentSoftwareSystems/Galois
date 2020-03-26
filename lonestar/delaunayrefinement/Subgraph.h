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

#ifndef SUBGRAPH_H
#define SUBGRAPH_H

#include "Element.h"

#include "galois/Galois.h"
#include "galois/graphs/Graph.h"

#include <vector>
#include <algorithm>

typedef galois::graphs::MorphGraph<Element, void, false> Graph;
typedef Graph::GraphNode GNode;

struct EdgeTuple {
  GNode src;
  GNode dst;
  Edge data;
  EdgeTuple(GNode s, GNode d, const Edge& _d) : src(s), dst(d), data(_d) {}

  bool operator==(const EdgeTuple& rhs) const {
    return src == rhs.src && dst == rhs.dst && data == data;
  }
};

/**
 *  A sub-graph of the mesh. Used to store information about the original
 *  cavity
 */
class PreGraph {
  typedef std::vector<GNode, galois::PerIterAllocTy::rebind<GNode>::other>
      NodesTy;
  NodesTy nodes;

public:
  typedef NodesTy::iterator iterator;

  explicit PreGraph(galois::PerIterAllocTy& cnx) : nodes(cnx) {}

  bool containsNode(GNode N) {
    return std::find(nodes.begin(), nodes.end(), N) != nodes.end();
  }

  void addNode(GNode n) { return nodes.push_back(n); }
  void reset() { nodes.clear(); }
  iterator begin() { return nodes.begin(); }
  iterator end() { return nodes.end(); }
};

/**
 *  A sub-graph of the mesh. Used to store information about the original
 *  and updated cavity
 */
class PostGraph {
  struct TempEdge {
    size_t src;
    GNode dst;
    Edge edge;
    TempEdge(size_t s, GNode d, const Edge& e) : src(s), dst(d), edge(e) {}
  };

  typedef std::vector<GNode, galois::PerIterAllocTy::rebind<GNode>::other>
      NodesTy;
  typedef std::vector<EdgeTuple,
                      galois::PerIterAllocTy::rebind<EdgeTuple>::other>
      EdgesTy;

  //! the nodes in the graph before updating
  NodesTy nodes;
  //! the edges that connect the subgraph to the rest of the graph
  EdgesTy edges;

public:
  typedef NodesTy::iterator iterator;
  typedef EdgesTy::iterator edge_iterator;

  explicit PostGraph(galois::PerIterAllocTy& cnx) : nodes(cnx), edges(cnx) {}

  void addNode(GNode n) { nodes.push_back(n); }

  void addEdge(GNode src, GNode dst, const Edge& e) {
    edges.push_back(EdgeTuple(src, dst, e));
  }

  void reset() {
    nodes.clear();
    edges.clear();
  }

  iterator begin() { return nodes.begin(); }
  iterator end() { return nodes.end(); }
  edge_iterator edge_begin() { return edges.begin(); }
  edge_iterator edge_end() { return edges.end(); }
};

#endif
