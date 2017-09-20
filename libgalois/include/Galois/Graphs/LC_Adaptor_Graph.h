/** Local Computation graphs -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, version 2.1 of the
 * License.
 *
 * Galois is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Galois.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @section Copyright
 *
 * Copyright (C) 2015, The University of Texas at Austin. All rights
 * reserved.
 *
 * @section Description
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#ifndef GALOIS_GRAPH_LC_ADAPTOR_H
#define GALOIS_GRAPH_LC_ADAPTOR_H

#include "Galois/LargeArray.h"
#include "Galois/Graphs/Details.h"

namespace galois { namespace Graph {

template<typename NodeTy, typename EdgeTy, typename DerivedTy,
  typename GraphNodeTy, typename IteratorTy, typename EdgeIteratorTy,
  bool HasNoLockable = false>
class LC_Adaptor_Graph:
  private detail::OutOfLineLockableFeature<HasNoLockable>, 
  private detail::LocalIteratorFeature<false> {
public:
  //! If true, do not use abstract locks in graph
  template<bool _has_no_lockable>
  struct with_no_lockable { typedef LC_Adaptor_Graph<NodeTy,EdgeTy,DerivedTy,GraphNodeTy,IteratorTy,EdgeIteratorTy,_has_no_lockable> type; };

  typedef GraphNodeTy GraphNode;
  typedef EdgeTy edge_data_type;
  typedef NodeTy node_data_type;
  typedef typename detail::EdgeInfoBase<void*,EdgeTy>::reference edge_data_reference;
  typedef typename detail::NodeInfoBase<NodeTy,false>::reference node_data_reference;
  typedef EdgeIteratorTy edge_iterator;
  typedef IteratorTy iterator;
  typedef iterator const_iterator;
  typedef iterator local_iterator;

protected:
  template<bool _A1 = HasNoLockable>
  void acquireNode(GraphNode N, MethodFlag mflag, typename std::enable_if<!_A1>::type* = 0) {
    this->outOfLineAcquire(getId(N), mflag);
  }

  template<bool _A1 = HasNoLockable>
  void acquireNode(GraphNode N, MethodFlag mflag, typename std::enable_if<_A1>::type* = 0) { }

  const DerivedTy& derived() const { return *static_cast<const DerivedTy*>(this); }

  DerivedTy& derived() { return *static_cast<DerivedTy*>(this); }

  size_t getId(GraphNode n) {
    return derived().get_id(n);
  }
  
public:
  node_data_reference getData(GraphNode N, MethodFlag mflag = MethodFlag::WRITE) {
    // galois::Runtime::checkWrite(mflag, false);
    acquireNode(N, mflag);
    return derived().get_data(N);
  }

  edge_data_reference getEdgeData(edge_iterator ni, MethodFlag mflag = MethodFlag::UNPROTECTED) {
    // galois::Runtime::checkWrite(mflag, false);
    return derived().get_edge_data(ni);
  }

  GraphNode getEdgeDst(edge_iterator ni) {
    return derived().get_edge_dst(ni);
  }

  uint64_t size() const { return derived().get_size(); }
  uint64_t sizeEdges() const { return derived().get_size_edges(); }

  iterator begin() const { return derived().get_begin(); }
  iterator end() const { return derived().get_end(); }
  local_iterator local_begin() { return local_iterator(this->localBegin(size())); }
  local_iterator local_end() { return local_iterator(this->localEnd(size())); }

  edge_iterator edge_begin(GraphNode N, MethodFlag mflag = MethodFlag::WRITE) {
    acquireNode(N, mflag);
    if (galois::Runtime::shouldLock(mflag)) {
      for (edge_iterator ii = derived().get_edge_begin(N), ee = derived().get_edge_end(N); ii != ee; ++ii) {
        acquireNode(getEdgeDst(ii), mflag);
      }
    }
    return derived().get_edge_begin(N);
  }

  edge_iterator edge_end(GraphNode N, MethodFlag mflag = MethodFlag::WRITE) {
    acquireNode(N, mflag);
    return derived().get_edge_end(N);
  }

  detail::EdgesIterator<LC_Adaptor_Graph> out_edges(GraphNode N, MethodFlag mflag = MethodFlag::WRITE) {
    return detail::EdgesIterator<LC_Adaptor_Graph>(*this, N, mflag);
  }
};

} } // end namespace

#endif
