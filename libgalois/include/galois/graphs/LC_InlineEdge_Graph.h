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

#ifndef GALOIS_GRAPHS_LC_INLINEEDGE_GRAPH_H
#define GALOIS_GRAPHS_LC_INLINEEDGE_GRAPH_H

#include <type_traits>

#include "galois/config.h"
#include "galois/LargeArray.h"
#include "galois/graphs/FileGraph.h"
#include "galois/graphs/Details.h"

namespace galois {
namespace graphs {

/**
 * Local computation graph (i.e., graph structure does not change). The data
 * representation is a modification of {@link LC_CSR_Graph} where the edge data
 * is stored inline with the adjacency information.
 *
 * The position of template parameters may change between Galois releases; the
 * most robust way to specify them is through the with_XXX nested templates.
 */
template <typename NodeTy, typename EdgeTy, bool HasNoLockable = false,
          bool UseNumaAlloc = false, bool HasOutOfLineLockable = false,
          bool HasCompressedNodePtr = false, typename FileEdgeTy = EdgeTy>
class LC_InlineEdge_Graph
    : private boost::noncopyable,
      private internal::LocalIteratorFeature<UseNumaAlloc>,
      private internal::OutOfLineLockableFeature<HasOutOfLineLockable &&
                                                 !HasNoLockable> {
  template <typename Graph>
  friend class LC_InOut_Graph;

public:
  template <bool _has_id>
  struct with_id {
    typedef LC_InlineEdge_Graph type;
  };

  template <typename _node_data>
  struct with_node_data {
    typedef LC_InlineEdge_Graph<_node_data, EdgeTy, HasNoLockable, UseNumaAlloc,
                                HasOutOfLineLockable, HasCompressedNodePtr,
                                FileEdgeTy>
        type;
  };

  template <typename _edge_data>
  struct with_edge_data {
    typedef LC_InlineEdge_Graph<NodeTy, _edge_data, HasNoLockable, UseNumaAlloc,
                                HasOutOfLineLockable, HasCompressedNodePtr,
                                FileEdgeTy>
        type;
  };

  template <typename _file_edge_data>
  struct with_file_edge_data {
    typedef LC_InlineEdge_Graph<NodeTy, EdgeTy, HasNoLockable, UseNumaAlloc,
                                HasOutOfLineLockable, HasCompressedNodePtr,
                                _file_edge_data>
        type;
  };

  template <bool _has_no_lockable>
  struct with_no_lockable {
    typedef LC_InlineEdge_Graph<NodeTy, EdgeTy, _has_no_lockable, UseNumaAlloc,
                                HasOutOfLineLockable, HasCompressedNodePtr,
                                FileEdgeTy>
        type;
  };

  template <bool _use_numa_alloc>
  struct with_numa_alloc {
    typedef LC_InlineEdge_Graph<NodeTy, EdgeTy, HasNoLockable, _use_numa_alloc,
                                HasOutOfLineLockable, HasCompressedNodePtr,
                                FileEdgeTy>
        type;
  };

  template <bool _has_out_of_line_lockable>
  struct with_out_of_line_lockable {
    typedef LC_InlineEdge_Graph<NodeTy, EdgeTy, HasNoLockable, UseNumaAlloc,
                                _has_out_of_line_lockable, HasCompressedNodePtr,
                                FileEdgeTy>
        type;
  };

  /**
   * Compress representation of graph at the expense of one level of indirection
   * on accessing neighbors of a node
   */
  template <bool _has_compressed_node_ptr>
  struct with_compressed_node_ptr {
    typedef LC_InlineEdge_Graph<NodeTy, EdgeTy, HasNoLockable, UseNumaAlloc,
                                HasOutOfLineLockable, _has_compressed_node_ptr,
                                FileEdgeTy>
        type;
  };

  typedef read_default_graph_tag read_tag;

protected:
  class NodeInfo;
  typedef internal::EdgeInfoBase<
      typename std::conditional<HasCompressedNodePtr, uint32_t,
                                NodeInfo*>::type,
      EdgeTy>
      EdgeInfo;
  typedef LargeArray<EdgeInfo> EdgeData;
  typedef LargeArray<NodeInfo> NodeData;
  typedef internal::NodeInfoBaseTypes<NodeTy,
                                      !HasNoLockable && !HasOutOfLineLockable>
      NodeInfoTypes;

  class NodeInfo
      : public internal::NodeInfoBase<NodeTy,
                                      !HasNoLockable && !HasOutOfLineLockable> {
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

  template <bool C_b = HasCompressedNodePtr>
  NodeInfo* getDst(edge_iterator ii,
                   typename std::enable_if<C_b>::type* = 0) const {
    return const_cast<NodeInfo*>(&nodeData[ii->dst]);
  }

  template <bool C_b = HasCompressedNodePtr>
  NodeInfo* getDst(edge_iterator ii,
                   typename std::enable_if<!C_b>::type* = 0) const {
    return ii->dst;
  }

  template <typename Container, typename Index, bool C_b = HasCompressedNodePtr>
  void setEdgeDst(Container&, edge_iterator edge, Index idx,
                  typename std::enable_if<C_b>::type* = 0) {
    edge->dst = idx;
  }

  template <typename Container, typename Index, bool C_b = HasCompressedNodePtr>
  void setEdgeDst(Container& c, edge_iterator edge, Index idx,
                  typename std::enable_if<!C_b>::type* = 0) {
    edge->dst = &c[idx];
  }

  template <bool _A1 = HasNoLockable, bool _A2 = HasOutOfLineLockable>
  void acquireNode(GraphNode N, MethodFlag mflag,
                   typename std::enable_if<!_A1&& !_A2>::type* = 0) {
    galois::runtime::acquire(N, mflag);
  }

  template <bool _A1 = HasOutOfLineLockable, bool _A2 = HasNoLockable>
  void acquireNode(GraphNode N, MethodFlag mflag,
                   typename std::enable_if<_A1&& !_A2>::type* = 0) {
    this->outOfLineAcquire(getId(N), mflag);
  }

  template <bool _A1 = HasOutOfLineLockable, bool _A2 = HasNoLockable>
  void acquireNode(GraphNode, MethodFlag,
                   typename std::enable_if<_A2>::type* = 0) {}

  edge_iterator raw_begin(GraphNode N) {
    return nodeData[getId(N)].edgeBegin();
  }

  edge_iterator raw_end(GraphNode N) { return nodeData[getId(N)].edgeEnd(); }

  template <bool _A1 = EdgeInfo::has_value,
            bool _A2 = LargeArray<FileEdgeTy>::has_value>
  void constructEdgeValue(FileGraph& graph,
                          typename FileGraph::edge_iterator nn, EdgeInfo* edge,
                          typename std::enable_if<!_A1 || _A2>::type* = 0) {
    typedef LargeArray<FileEdgeTy> FED;
    if (EdgeInfo::has_value)
      edge->construct(graph.getEdgeData<typename FED::value_type>(nn));
  }

  template <bool _A1 = EdgeInfo::has_value,
            bool _A2 = LargeArray<FileEdgeTy>::has_value>
  void constructEdgeValue(FileGraph&, typename FileGraph::edge_iterator,
                          EdgeInfo* edge,
                          typename std::enable_if<_A1&& !_A2>::type* = 0) {
    edge->construct();
  }

  size_t getId(GraphNode N) { return std::distance(this->nodeData.data(), N); }

  GraphNode getNode(size_t n) { return &nodeData[n]; }

public:
  ~LC_InlineEdge_Graph() {
    if (!EdgeInfo::has_value)
      return;
    if (numNodes == 0)
      return;

    for (edge_iterator ii = nodeData[0].edgeBegin(),
                       ei = nodeData[numNodes - 1].edgeEnd();
         ii != ei; ++ii) {
      ii->destroy();
    }
  }

  node_data_reference getData(GraphNode N,
                              MethodFlag mflag = MethodFlag::WRITE) {
    // galois::runtime::checkWrite(mflag, false);
    acquireNode(N, mflag);
    return N->getData();
  }

  edge_data_reference
  getEdgeData(edge_iterator ni,
              MethodFlag GALOIS_UNUSED(mflag) = MethodFlag::UNPROTECTED) const {
    // galois::runtime::checkWrite(mflag, false);
    return ni->get();
  }

  GraphNode getEdgeDst(edge_iterator ni) const { return getDst(ni); }

  size_t size() const { return numNodes; }
  size_t sizeEdges() const { return numEdges; }

  const_iterator begin() const { return const_iterator(nodeData.begin()); }
  const_iterator end() const { return const_iterator(nodeData.end()); }
  iterator begin() { return iterator(nodeData.data()); }
  iterator end() { return iterator(nodeData.end()); }

  local_iterator local_begin() {
    return local_iterator(&nodeData[this->localBegin(numNodes)]);
  }
  local_iterator local_end() {
    return local_iterator(&nodeData[this->localEnd(numNodes)]);
  }
  const_local_iterator local_begin() const {
    return const_local_iterator(&nodeData[this->localBegin(numNodes)]);
  }
  const_local_iterator local_end() const {
    return const_local_iterator(&nodeData[this->localEnd(numNodes)]);
  }

  edge_iterator edge_begin(GraphNode N, MethodFlag mflag = MethodFlag::WRITE) {
    acquireNode(N, mflag);
    if (galois::runtime::shouldLock(mflag)) {
      for (edge_iterator ii = N->edgeBegin(), ee = N->edgeEnd(); ii != ee;
           ++ii) {
        acquireNode(getDst(ii), mflag);
      }
    }
    return N->edgeBegin();
  }

  edge_iterator edge_end(GraphNode N, MethodFlag mflag = MethodFlag::WRITE) {
    acquireNode(N, mflag);
    return N->edgeEnd();
  }

  runtime::iterable<NoDerefIterator<edge_iterator>>
  edges(GraphNode N, MethodFlag mflag = MethodFlag::WRITE) {
    return internal::make_no_deref_range(edge_begin(N, mflag),
                                         edge_end(N, mflag));
  }

  runtime::iterable<NoDerefIterator<edge_iterator>>
  out_edges(GraphNode N, MethodFlag mflag = MethodFlag::WRITE) {
    return edges(N, mflag);
  }

#if 0
  /**
   * Sorts outgoing edges of a node. Comparison function is over EdgeTy.
   */
  template<typename CompTy>
  void sortEdgesByEdgeData(GraphNode N, const CompTy& comp = std::less<EdgeTy>(), MethodFlag mflag = MethodFlag::WRITE) {
    galois::runtime::acquire(N, mflag);
    std::sort(edge_sort_begin(N), edge_sort_end(N), EdgeSortCompWrapper<EdgeSortValue<GraphNode,EdgeTy>,CompTy>(comp));
  }

  /**
   * Sorts outgoing edges of a node. Comparison function is over <code>EdgeSortValue<EdgeTy></code>.
   */
  template<typename CompTy>
  void sortEdges(GraphNode N, const CompTy& comp, MethodFlag mflag = MethodFlag::WRITE) {
    galois::runtime::acquire(N, mflag);
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
    auto r =
        graph
            .divideByNode(NodeData::size_of::value +
                              LC_InlineEdge_Graph::size_of_out_of_line::value,
                          EdgeData::size_of::value, tid, total)
            .first;

    EdgeInfo* curEdge = edgeData.data() + *graph.edge_begin(*r.first);

    this->setLocalRange(*r.first, *r.second);

    for (FileGraph::iterator ii = r.first, ei = r.second; ii != ei; ++ii) {
      nodeData.constructAt(*ii);
      this->outOfLineConstructAt(*ii);
      nodeData[*ii].edgeBegin() = curEdge;
      for (FileGraph::edge_iterator nn = graph.edge_begin(*ii),
                                    en = graph.edge_end(*ii);
           nn != en; ++nn) {
        constructEdgeValue(graph, nn, curEdge);
        setEdgeDst(nodeData, curEdge, graph.getEdgeDst(nn));
        ++curEdge;
      }
      nodeData[*ii].edgeEnd() = curEdge;
    }
  }
};

} // namespace graphs
} // namespace galois

#endif
