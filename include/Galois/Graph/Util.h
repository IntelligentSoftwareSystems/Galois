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
#include "Galois/Runtime/Context.h"
#include "Galois/Runtime/MethodFlags.h"
#include "Galois/Runtime/mm/Mem.h"

#include <algorithm>

namespace Galois {
namespace Graph {

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

//! Proxy object for {@link EdgeSortIterator}
template<typename EdgeTy>
struct EdgeSortValue: public StrictObject<EdgeTy> {
  typedef StrictObject<EdgeTy> Super;
  typedef typename Super::value_type value_type;

  uint32_t dst;
  
  EdgeSortValue(uint32_t d, const value_type& v): Super(v), dst(d) { }

  template<typename ER>
  EdgeSortValue(const ER& ref) {
    ref.initialize(*this);
  }
};

//! Proxy object for {@link EdgeSortIterator}
template<typename EdgeDst, typename EdgeData>
struct EdgeSortReference {
  typedef typename EdgeData::raw_value_type EdgeTy;
  uint64_t at;
  EdgeDst* edgeDst;
  EdgeData* edgeData;

  EdgeSortReference(uint64_t x, EdgeDst* dsts, EdgeData* data): at(x), edgeDst(dsts), edgeData(data) { }

  EdgeSortReference operator=(const EdgeSortValue<EdgeTy>& x) {
    edgeDst->at(at) = x.dst;
    edgeData->set(at, x.get());
    return *this;
  }

  EdgeSortReference operator=(const EdgeSortReference<EdgeDst,EdgeData>& x) {
    edgeDst->at(at) = edgeDst->at(x.at);
    edgeData->set(at, edgeData->at(x.at));
    return *this;
  }

  EdgeSortValue<EdgeTy> operator*() const {
    return EdgeSortValue<EdgeTy>(edgeDst->at(at), edgeData->at(at));
  }

  void initialize(EdgeSortValue<EdgeTy>& value) const {
    value = *(*this);
  }
};

/**
 * Converts comparison functions over EdgeTy to be over {@link EdgeSortValue}.
 */
template<typename EdgeSortValueTy,typename CompTy>
struct EdgeSortCompWrapper {
  const CompTy& comp;

  EdgeSortCompWrapper(const CompTy& c): comp(c) { }
  bool operator()(const EdgeSortValueTy& a, const EdgeSortValueTy& b) const {
    return comp(a.get(), b.get());
  }
};

/**
 * Iterator to facilitate sorting of CSR-like graphs. Converts random access operations
 * on iterator to appropriate computations on edge destinations and edge data.
 *
 * @tparam EdgeDst {@link LargeArray}-like container of edge destinations
 * @tparam EdgeData {@link LargeArray}-like container of edge data
 */
template<typename EdgeDst, typename EdgeData>
class EdgeSortIterator: public boost::iterator_facade<
                        EdgeSortIterator<EdgeDst,EdgeData>,
                        EdgeSortValue<typename EdgeData::raw_value_type>,
                        boost::random_access_traversal_tag,
                        EdgeSortReference<EdgeDst, EdgeData>
                        > {
  uint64_t at;
  EdgeDst* edgeDst;
  EdgeData* edgeData;
public:
  EdgeSortIterator(): at(~0) { }
  EdgeSortIterator(uint64_t x, EdgeDst* dsts, EdgeData* data):
    at(x), edgeDst(dsts), edgeData(data) { }
private:
  friend class boost::iterator_core_access;
  
  bool equal(const EdgeSortIterator<EdgeDst,EdgeData>& other) const { return at == other.at; }
  EdgeSortReference<EdgeDst,EdgeData>
    dereference() const { return EdgeSortReference<EdgeDst, EdgeData>(at, edgeDst, edgeData); }
  ptrdiff_t distance_to(const EdgeSortIterator<EdgeDst,EdgeData>& other) const { return other.at - (ptrdiff_t) at; }
  void increment() { ++at; }
  void decrement() { --at; }
  void advance(ptrdiff_t n) { at += n; }
};

//! Specializations for void node data
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

//! Edge specialization for void edge data
template<typename NodeInfoTy,typename EdgeTy>
struct EdgeInfoBase: public LazyObject<EdgeTy> {
  typedef LazyObject<EdgeTy> Super;
  typedef typename Super::reference reference;
  typedef typename Super::value_type value_type;
  const static bool has_value = Super::has_value;

  NodeInfoTy* dst;
};

/**
 * Convenience wrapper around Graph.edge_begin and Graph.edge_end to allow
 * C++11 foreach iteration of edges.
 */
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

template<typename GraphTy>
class EdgesWithNoFlagIterator {
  GraphTy& g;
  typename GraphTy::GraphNode n;
public:
  typedef NoDerefIterator<typename GraphTy::edge_iterator> iterator;

  EdgesWithNoFlagIterator(GraphTy& g, typename GraphTy::GraphNode n): g(g), n(n) { }

  iterator begin() { return make_no_deref_iterator(g.edge_begin(n)); }
  iterator end() { return make_no_deref_iterator(g.edge_end(n)); }
};

} // end namespace
} // end namespace

#endif
