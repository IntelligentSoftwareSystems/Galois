/** Useful classes and methods for implementing graphs  -*- C++ -*-
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
 * There are two main classes, ::FileGraph and ::LC_XXX_Graph. The former
 * represents the pure structure of a graph (i.e., whether an edge exists between
 * two nodes) and cannot be modified. The latter allows values to be stored on
 * nodes and edges, but the structure of the graph cannot be modified.
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#ifndef GALOIS_GRAPH_UTIL_H
#define GALOIS_GRAPH_UTIL_H

#include "Galois/LazyObject.h"
#include "Galois/NoDerefIterator.h"
#include "Galois/Threads.h"
#include "Galois/Graph/FileGraph.h"
#include "Galois/Runtime/MethodFlags.h"
#include "Galois/Runtime/mm/Mem.h"

#include <algorithm>

namespace Galois {
namespace Graph {

template<typename GraphTy>
void structureFromFile(GraphTy& g, const std::string& fname) {
  FileGraph graph;
  graph.structureFromFile(fname);
  g.structureFromGraph(graph);
}

template<typename EdgeContainerTy,typename CompTy>
struct EdgeSortCompWrapper {
  const CompTy& comp;

  EdgeSortCompWrapper(const CompTy& c): comp(c) { }
  bool operator()(const EdgeContainerTy& a, const EdgeContainerTy& b) const {
    return comp(a.get(), b.get());
  }
};

uint64_t inline localStart(uint64_t numNodes) {
  unsigned int id = Galois::Runtime::LL::getTID();
  unsigned int num = Galois::getActiveThreads();
  return (numNodes + num - 1) / num * id;
}

uint64_t inline localEnd(uint64_t numNodes) {
  unsigned int id = Galois::Runtime::LL::getTID();
  unsigned int num = Galois::getActiveThreads();
  uint64_t end = (numNodes + num - 1) / num * (id + 1);
  return std::min(end, numNodes);
}

//! Partial specializations for void node data
template<typename NodeTy>
class NodeInfoBase: public Galois::Runtime::Lockable {
  NodeTy data;
public:
  typedef NodeTy& reference;
  reference getData() { return data; } 
  void destruct() {
    (&data)->~NodeTy();
  }
  void construct() { 
    new (&data) NodeTy;
  }
};

template<>
struct NodeInfoBase<void>: public Galois::Runtime::Lockable {
  typedef void* reference;
  reference getData() { return 0; }
  void destruct() { }
  void construct() { }
};

//! Convenience wrapper around Graph.edge_begin and Graph.edge_end to allow
//! C++11 foreach iteration of edges
template<typename GraphTy>
class EdgesIterator {
  GraphTy& g;
  typename GraphTy::GraphNode n;
  MethodFlag flag;
public:
  typedef NoDerefIterator<typename GraphTy::edge_iterator> iterator;

  EdgesIterator(GraphTy& g, typename GraphTy::GraphNode n, MethodFlag f): g(g), n(n), flag(f) { }

  iterator begin() { return make_no_deref_iterator(g.edge_begin(n, flag)); }
  iterator end() { return make_no_deref_iterator(g.edge_end(n, flag)); }
};

template<typename NodeInfoTy,typename EdgeTy>
struct EdgeInfoBase: public LazyObject<EdgeTy> {
  typedef LazyObject<EdgeTy> Super;
  typedef typename Super::reference reference;
  typedef typename Super::value_type value_type;
  const static bool has_value = Super::has_value;

  NodeInfoTy* dst;
};

} // end namespace
} // end namespace

#endif
