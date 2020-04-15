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

#ifndef GALOIS_GRAPHS_LC_ADAPTOR_GRAPH_H
#define GALOIS_GRAPHS_LC_ADAPTOR_GRAPH_H

#include "galois/config.h"
#include "galois/graphs/Details.h"
#include "galois/LargeArray.h"

namespace galois {
namespace graphs {

template <typename NodeTy, typename EdgeTy, typename DerivedTy,
          typename GraphNodeTy, typename IteratorTy, typename EdgeIteratorTy,
          bool HasNoLockable = false>
class LC_Adaptor_Graph
    : private internal::OutOfLineLockableFeature<HasNoLockable>,
      private internal::LocalIteratorFeature<false> {
public:
  //! If true, do not use abstract locks in graph
  template <bool _has_no_lockable>
  struct with_no_lockable {
    typedef LC_Adaptor_Graph<NodeTy, EdgeTy, DerivedTy, GraphNodeTy, IteratorTy,
                             EdgeIteratorTy, _has_no_lockable>
        type;
  };

  typedef GraphNodeTy GraphNode;
  typedef EdgeTy edge_data_type;
  typedef NodeTy node_data_type;
  typedef typename internal::EdgeInfoBase<void*, EdgeTy>::reference
      edge_data_reference;
  typedef typename internal::NodeInfoBase<NodeTy, false>::reference
      node_data_reference;
  typedef EdgeIteratorTy edge_iterator;
  typedef IteratorTy iterator;
  typedef iterator const_iterator;
  typedef iterator local_iterator;

protected:
  template <bool _A1 = HasNoLockable>
  void acquireNode(GraphNode N, MethodFlag mflag,
                   typename std::enable_if<!_A1>::type* = 0) {
    this->outOfLineAcquire(getId(N), mflag);
  }

  template <bool _A1 = HasNoLockable>
  void acquireNode(GraphNode, MethodFlag,
                   typename std::enable_if<_A1>::type* = 0) {}

  const DerivedTy& derived() const {
    return *static_cast<const DerivedTy*>(this);
  }

  DerivedTy& derived() { return *static_cast<DerivedTy*>(this); }

  size_t getId(GraphNode n) { return derived().get_id(n); }

public:
  node_data_reference getData(GraphNode N,
                              MethodFlag mflag = MethodFlag::WRITE) {
    // galois::runtime::checkWrite(mflag, false);
    acquireNode(N, mflag);
    return derived().get_data(N);
  }

  edge_data_reference getEdgeData(edge_iterator ni,
    MethodFlag GALOIS_UNUSED(mflag) = MethodFlag::UNPROTECTED) {
    // galois::runtime::checkWrite(mflag, false);
    return derived().get_edge_data(ni);
  }

  GraphNode getEdgeDst(edge_iterator ni) { return derived().get_edge_dst(ni); }

  uint64_t size() const { return derived().get_size(); }
  uint64_t sizeEdges() const { return derived().get_size_edges(); }

  iterator begin() const { return derived().get_begin(); }
  iterator end() const { return derived().get_end(); }
  local_iterator local_begin() {
    return local_iterator(this->localBegin(size()));
  }
  local_iterator local_end() { return local_iterator(this->localEnd(size())); }

  edge_iterator edge_begin(GraphNode N, MethodFlag mflag = MethodFlag::WRITE) {
    acquireNode(N, mflag);
    if (galois::runtime::shouldLock(mflag)) {
      for (edge_iterator ii = derived().get_edge_begin(N),
                         ee = derived().get_edge_end(N);
           ii != ee; ++ii) {
        acquireNode(getEdgeDst(ii), mflag);
      }
    }
    return derived().get_edge_begin(N);
  }

  edge_iterator edge_end(GraphNode N, MethodFlag mflag = MethodFlag::WRITE) {
    acquireNode(N, mflag);
    return derived().get_edge_end(N);
  }

  internal::EdgesIterator<LC_Adaptor_Graph>
  out_edges(GraphNode N, MethodFlag mflag = MethodFlag::WRITE) {
    return internal::EdgesIterator<LC_Adaptor_Graph>(*this, N, mflag);
  }
};

} // namespace graphs
} // namespace galois

#endif
