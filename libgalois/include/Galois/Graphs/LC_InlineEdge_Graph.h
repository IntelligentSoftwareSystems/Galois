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
 * License
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
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */

#ifndef GALOIS_GRAPH_LC_INLINEEDGE_GRAPH_H
#define GALOIS_GRAPH_LC_INLINEEDGE_GRAPH_H

#include "Galois/LargeArray.h"
#include "Galois/Graphs/FileGraph.h"
#include "Galois/Graphs/Details.h"

#include <boost/mpl/if.hpp>
#include <type_traits>

namespace galois {
namespace Graph {

/**
 * Local computation graph (i.e., graph structure does not change). The data representation
 * is a modification of {@link LC_CSR_Graph} where the edge data is stored inline with the
 * adjacency information. 
 *
 * The position of template parameters may change between Galois releases; the
 * most robust way to specify them is through the with_XXX nested templates.
 */
template<typename NodeTy, typename EdgeTy,
  bool HasNoLockable=false,
  bool UseNumaAlloc=false,
  bool HasOutOfLineLockable=false,
  bool HasCompressedNodePtr=false,
  typename FileEdgeTy=EdgeTy>
class LC_InlineEdge_Graph:
    private boost::noncopyable,
    private detail::LocalIteratorFeature<UseNumaAlloc>,
    private detail::OutOfLineLockableFeature<HasOutOfLineLockable && !HasNoLockable> {
  template<typename Graph> friend class LC_InOut_Graph;

public:
  template<bool _has_id>
  struct with_id { typedef LC_InlineEdge_Graph type; };

  template<typename _node_data>
  struct with_node_data { typedef LC_InlineEdge_Graph<_node_data,EdgeTy,HasNoLockable,UseNumaAlloc,HasOutOfLineLockable,HasCompressedNodePtr,FileEdgeTy> type; };

  template<typename _edge_data>
  struct with_edge_data { typedef LC_InlineEdge_Graph<NodeTy,_edge_data,HasNoLockable,UseNumaAlloc,HasOutOfLineLockable,HasCompressedNodePtr,FileEdgeTy> type; };

  template<typename _file_edge_data>
  struct with_file_edge_data { typedef LC_InlineEdge_Graph<NodeTy,EdgeTy,HasNoLockable,UseNumaAlloc,HasOutOfLineLockable,HasCompressedNodePtr,_file_edge_data> type; };

  template<bool _has_no_lockable>
  struct with_no_lockable { typedef LC_InlineEdge_Graph<NodeTy,EdgeTy,_has_no_lockable,UseNumaAlloc,HasOutOfLineLockable,HasCompressedNodePtr,FileEdgeTy> type; };

  template<bool _use_numa_alloc>
  struct with_numa_alloc { typedef LC_InlineEdge_Graph<NodeTy,EdgeTy,HasNoLockable,_use_numa_alloc,HasOutOfLineLockable,HasCompressedNodePtr,FileEdgeTy> type; };

  template<bool _has_out_of_line_lockable>
  struct with_out_of_line_lockable { typedef LC_InlineEdge_Graph<NodeTy,EdgeTy,HasNoLockable,UseNumaAlloc,_has_out_of_line_lockable,HasCompressedNodePtr,FileEdgeTy> type; };

  /**
   * Compress representation of graph at the expense of one level of indirection on accessing
   * neighbors of a node
   */
  template<bool _has_compressed_node_ptr>
  struct with_compressed_node_ptr { typedef  LC_InlineEdge_Graph<NodeTy,EdgeTy,HasNoLockable,UseNumaAlloc,HasOutOfLineLockable,_has_compressed_node_ptr,FileEdgeTy> type; };

  typedef read_default_graph_tag read_tag;

protected:
  class NodeInfo;
  typedef detail::EdgeInfoBase<typename boost::mpl::if_c<HasCompressedNodePtr,uint32_t,NodeInfo*>::type,EdgeTy> EdgeInfo;
  typedef LargeArray<EdgeInfo> EdgeData;
  typedef LargeArray<NodeInfo> NodeData;
  typedef detail::NodeInfoBaseTypes<NodeTy,!HasNoLockable && !HasOutOfLineLockable> NodeInfoTypes;

  class NodeInfo: public detail::NodeInfoBase<NodeTy,!HasNoLockable && !HasOutOfLineLockable> {
    EdgeInfo* m_edgeBegin;
    EdgeInfo* m_edgeEnd;
  public:
    EdgeInfo*& edgeBegin() { return m_edgeBegin; }
    EdgeInfo*& edgeEnd() { return m_edgeEnd; }
  };

public:
  typedef NodeInfo* GraphNode;
  typedef EdgeTy edge_data_type;
  typedef FileEdgeTy file_edge_data_type;
  typedef NodeTy node_data_type;
  typedef typename EdgeInfo::reference edge_data_reference;
  typedef typename NodeInfoTypes::reference node_data_reference;
  typedef EdgeInfo* edge_iterator;
  typedef galois::NoDerefIterator<NodeInfo*> iterator;
  typedef galois::NoDerefIterator<const NodeInfo*> const_iterator;
  typedef iterator local_iterator;
  typedef const_iterator const_local_iterator;

protected:
  NodeData nodeData;
  EdgeData edgeData;
  uint64_t numNodes;
  uint64_t numEdges;

  template<bool C_b = HasCompressedNodePtr>
  NodeInfo* getDst(edge_iterator ii, typename std::enable_if<C_b>::type* x = 0) const {
    return const_cast<NodeInfo*>(&nodeData[ii->dst]);
  }

  template<bool C_b = HasCompressedNodePtr>
  NodeInfo* getDst(edge_iterator ii, typename std::enable_if<!C_b>::type* x = 0) const {
    return ii->dst;
  }

  template<typename Container,typename Index, bool C_b = HasCompressedNodePtr>
  void setEdgeDst(Container& c, edge_iterator edge, Index idx, typename std::enable_if<C_b>::type* = 0) {
    edge->dst = idx;
  }

  template<typename Container,typename Index, bool C_b = HasCompressedNodePtr>
  void setEdgeDst(Container& c, edge_iterator edge, Index idx, typename std::enable_if<!C_b>::type* = 0) {
    edge->dst = &c[idx];
  }

  template<bool _A1 = HasNoLockable, bool _A2 = HasOutOfLineLockable>
  void acquireNode(GraphNode N, MethodFlag mflag, typename std::enable_if<!_A1 && !_A2>::type* = 0) {
    galois::Runtime::acquire(N, mflag);
  }

  template<bool _A1 = HasOutOfLineLockable, bool _A2 = HasNoLockable>
  void acquireNode(GraphNode N, MethodFlag mflag, typename std::enable_if<_A1 && !_A2>::type* = 0) {
    this->outOfLineAcquire(getId(N), mflag);
  }

  template<bool _A1 = HasOutOfLineLockable, bool _A2 = HasNoLockable>
  void acquireNode(GraphNode N, MethodFlag mflag, typename std::enable_if<_A2>::type* = 0) { }

  edge_iterator raw_begin(GraphNode N) {
    return nodeData[getId(N)].edgeBegin();
  }

  edge_iterator raw_end(GraphNode N) {
    return nodeData[getId(N)].edgeEnd();
  }

  template<bool _A1 = EdgeInfo::has_value, bool _A2 = LargeArray<FileEdgeTy>::has_value>
  void constructEdgeValue(FileGraph& graph, typename FileGraph::edge_iterator nn,
      EdgeInfo* edge, typename std::enable_if<!_A1 || _A2>::type* = 0) {
    typedef LargeArray<FileEdgeTy> FED;
    if (EdgeInfo::has_value)
      edge->construct(graph.getEdgeData<typename FED::value_type>(nn));
  }

  template<bool _A1 = EdgeInfo::has_value, bool _A2 = LargeArray<FileEdgeTy>::has_value>
  void constructEdgeValue(FileGraph& graph, typename FileGraph::edge_iterator nn,
      EdgeInfo* edge, typename std::enable_if<_A1 && !_A2>::type* = 0) {
    edge->construct();
  }

  size_t getId(GraphNode N) {
    return std::distance(this->nodeData.data(), N);
  }

  GraphNode getNode(size_t n) {
    return &nodeData[n];
  }

public:
  ~LC_InlineEdge_Graph() {
    if (!EdgeInfo::has_value) return;
    if (numNodes == 0) return;

    for (edge_iterator ii = nodeData[0].edgeBegin(), ei = nodeData[numNodes-1].edgeEnd(); ii != ei; ++ii) {
      ii->destroy();
    }
  }

  node_data_reference getData(GraphNode N, MethodFlag mflag = MethodFlag::WRITE) {
    // galois::Runtime::checkWrite(mflag, false);
    acquireNode(N, mflag);
    return N->getData();
  }

  edge_data_reference getEdgeData(edge_iterator ni, MethodFlag mflag = MethodFlag::UNPROTECTED) const {
    // galois::Runtime::checkWrite(mflag, false);
    return ni->get();
   }

  GraphNode getEdgeDst(edge_iterator ni) const {
    return getDst(ni);
  }

  size_t size() const { return numNodes; }
  size_t sizeEdges() const { return numEdges; }

  const_iterator begin() const { return const_iterator(nodeData.begin()); }
  const_iterator end() const { return const_iterator(nodeData.end()); }
  iterator begin() { return iterator(nodeData.data()); }
  iterator end() { return iterator(nodeData.end()); }

  local_iterator local_begin() { return local_iterator(&nodeData[this->localBegin(numNodes)]); }
  local_iterator local_end() { return local_iterator(&nodeData[this->localEnd(numNodes)]); }
  const_local_iterator local_begin() const { return const_local_iterator(&nodeData[this->localBegin(numNodes)]); }
  const_local_iterator local_end() const { return const_local_iterator(&nodeData[this->localEnd(numNodes)]); }

  edge_iterator edge_begin(GraphNode N, MethodFlag mflag = MethodFlag::WRITE) {
    acquireNode(N, mflag);
    if (galois::Runtime::shouldLock(mflag)) {
      for (edge_iterator ii = N->edgeBegin(), ee = N->edgeEnd(); ii != ee; ++ii) {
        acquireNode(getDst(ii), mflag);
      }
    }
    return N->edgeBegin();
  }

  edge_iterator edge_end(GraphNode N, MethodFlag mflag = MethodFlag::WRITE) {
    acquireNode(N, mflag);
    return N->edgeEnd();
  }

  Runtime::iterable<NoDerefIterator<edge_iterator>> edges(GraphNode N, MethodFlag mflag = MethodFlag::WRITE) {
    return detail::make_no_deref_range(edge_begin(N, mflag), edge_end(N, mflag));
  }

  Runtime::iterable<NoDerefIterator<edge_iterator>> out_edges(GraphNode N, MethodFlag mflag = MethodFlag::WRITE) {
    return edges(N, mflag);
  }

#if 0
  /**
   * Sorts outgoing edges of a node. Comparison function is over EdgeTy.
   */
  template<typename CompTy>
  void sortEdgesByEdgeData(GraphNode N, const CompTy& comp = std::less<EdgeTy>(), MethodFlag mflag = MethodFlag::WRITE) {
    galois::Runtime::acquire(N, mflag);
    std::sort(edge_sort_begin(N), edge_sort_end(N), EdgeSortCompWrapper<EdgeSortValue<GraphNode,EdgeTy>,CompTy>(comp));
  }

  /**
   * Sorts outgoing edges of a node. Comparison function is over <code>EdgeSortValue<EdgeTy></code>.
   */
  template<typename CompTy>
  void sortEdges(GraphNode N, const CompTy& comp, MethodFlag mflag = MethodFlag::WRITE) {
    galois::Runtime::acquire(N, mflag);
    std::sort(edge_sort_begin(N), edge_sort_end(N), comp);
  }
#endif

  void allocateFrom(FileGraph& graph) {
    numNodes = graph.size();
    numEdges = graph.sizeEdges();

    if (UseNumaAlloc) {
      nodeData.allocateBlocked(numNodes);
      edgeData.allocateBlocked(numEdges);
      this->outOfLineAllocateBlocked(numNodes);
    } else {
      nodeData.allocateInterleaved(numNodes);
      edgeData.allocateInterleaved(numEdges);
      this->outOfLineAllocateInterleaved(numNodes);
    }
  }

  void constructFrom(FileGraph& graph, unsigned tid, unsigned total) {
    typedef typename EdgeInfo::value_type EDV;
    auto r = graph.divideByNode(
        NodeData::size_of::value + LC_InlineEdge_Graph::size_of_out_of_line::value,
        EdgeData::size_of::value,
        tid, total).first;

    EdgeInfo* curEdge = edgeData.data() + *graph.edge_begin(*r.first);

    this->setLocalRange(*r.first, *r.second);

    for (FileGraph::iterator ii = r.first, ei = r.second; ii != ei; ++ii) {
      nodeData.constructAt(*ii);
      this->outOfLineConstructAt(*ii);
      nodeData[*ii].edgeBegin() = curEdge;
      for (FileGraph::edge_iterator nn = graph.edge_begin(*ii), en = graph.edge_end(*ii); nn != en; ++nn) {
        constructEdgeValue(graph, nn, curEdge);
        setEdgeDst(nodeData, curEdge, graph.getEdgeDst(nn));
        ++curEdge;
      }
      nodeData[*ii].edgeEnd() = curEdge;
    }
  }
};

} // end namespace
} // end namespace

#endif
