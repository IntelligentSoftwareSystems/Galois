/**
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of XYZ License (a copy is located in
 * LICENSE.txt at the top-level directory).
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

#ifndef LC_PARTITIONEDINLINEEDGE_GRAPH_H
#define LC_PARTITIONEDINLINEEDGE_GRAPH_H

#include "galois/LargeArray.h"
#include "galois/graphs/FileGraph.h"
#include "galois/graphs/Details.h"

#include <boost/mpl/if.hpp>
#include <type_traits>

namespace galois {
namespace graphs {

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
class LC_PartitionedInlineEdge_Graph:
    boost::noncopyable,
    detail::LocalIteratorFeature<UseNumaAlloc>,
    detail::OutOfLineLockableFeature<HasOutOfLineLockable && !HasNoLockable> {
  template<typename Graph> friend class LC_InOut_Graph;

public:
  template<bool _has_id>
  using with_id = LC_PartitionedInlineEdge_Graph;

  template<typename _node_data>
  using with_node_data =
    LC_PartitionedInlineEdge_Graph<_node_data,EdgeTy,HasNoLockable,UseNumaAlloc,HasOutOfLineLockable,HasCompressedNodePtr,FileEdgeTy>;

  template<typename _edge_data>
  using with_edge_data =
    LC_PartitionedInlineEdge_Graph<NodeTy,_edge_data,HasNoLockable,UseNumaAlloc,HasOutOfLineLockable,HasCompressedNodePtr,FileEdgeTy>;

  template<typename _file_edge_data>
  using with_file_edge_data =
    LC_PartitionedInlineEdge_Graph<NodeTy,EdgeTy,HasNoLockable,UseNumaAlloc,HasOutOfLineLockable,HasCompressedNodePtr,_file_edge_data>;

  template<bool _has_no_lockable>
  using with_no_lockable =
    LC_PartitionedInlineEdge_Graph<NodeTy,EdgeTy,_has_no_lockable,UseNumaAlloc,HasOutOfLineLockable,HasCompressedNodePtr,FileEdgeTy>;

  template<bool _use_numa_alloc>
  using with_numa_alloc =
    LC_PartitionedInlineEdge_Graph<NodeTy,EdgeTy,HasNoLockable,_use_numa_alloc,HasOutOfLineLockable,HasCompressedNodePtr,FileEdgeTy>;

  template<bool _has_out_of_line_lockable>
  using with_out_of_line_lockable =
    LC_PartitionedInlineEdge_Graph<NodeTy,EdgeTy,HasNoLockable,UseNumaAlloc,_has_out_of_line_lockable,HasCompressedNodePtr,FileEdgeTy>;

  /**
   * Compress representation of graph at the expense of one level of indirection on accessing
   * neighbors of a node
   */
  template<bool _has_compressed_node_ptr>
  using with_compressed_node_ptr =
    LC_PartitionedInlineEdge_Graph<NodeTy,EdgeTy,HasNoLockable,UseNumaAlloc,HasOutOfLineLockable,_has_compressed_node_ptr,FileEdgeTy>;

  typedef read_default_graph_tag read_tag;

protected:
  class NodeInfo;
  typedef detail::EdgeInfoBase<typename boost::mpl::if_c<HasCompressedNodePtr,uint32_t,NodeInfo*>::type,EdgeTy> EdgeInfo;
  typedef LargeArray<EdgeInfo> EdgeData;

  class NodeInfo: public detail::NodeInfoBase<NodeTy,!HasNoLockable && !HasOutOfLineLockable> {
    EdgeInfo* m_edgeBegin;
    EdgeInfo* m_edgeEnd;
  public:
    EdgeInfo*& edgeBegin() { return m_edgeBegin; }
    EdgeInfo*& edgeEnd() { return m_edgeEnd; }
  };

  typedef LargeArray<NodeInfo> NodeData;

public:
  typedef NodeInfo* GraphNode;
  typedef EdgeTy edge_data_type;
  typedef FileEdgeTy file_edge_data_type;
  typedef NodeTy node_data_type;
  typedef typename EdgeInfo::reference edge_data_reference;
  typedef typename NodeInfo::reference node_data_reference;
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

  template<bool _C = HasCompressedNodePtr>
  NodeInfo* getDst(edge_iterator ii, typename std::enable_if<_C>::type* x = 0) const {
    return const_cast<NodeInfo*>(&nodeData[ii->dst]);
  }

  template<bool _C = HasCompressedNodePtr>
  NodeInfo* getDst(edge_iterator ii, typename std::enable_if<!_C>::type* x = 0) const {
    return ii->dst;
  }

  template<typename Container,typename Index, bool _C = HasCompressedNodePtr>
  void setEdgeDst(Container& c, edge_iterator edge, Index idx, typename std::enable_if<_C>::type* = 0) {
    edge->dst = idx;
  }

  template<typename Container,typename Index, bool _C = HasCompressedNodePtr>
  void setEdgeDst(Container& c, edge_iterator edge, Index idx, typename std::enable_if<!_C>::type* = 0) {
    edge->dst = &c[idx];
  }

  template<bool _A1 = HasNoLockable, bool _A2 = HasOutOfLineLockable>
  void acquireNode(GraphNode N, MethodFlag mflag, typename std::enable_if<!_A1 && !_A2>::type* = 0) {
    galois::runtime::acquire(N, mflag);
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
  ~LC_PartitionedInlineEdge_Graph() {
    if (!EdgeInfo::has_value) return;
    if (numNodes == 0) return;

    for (edge_iterator ii = nodeData[0].edgeBegin(), ei = nodeData[numNodes-1].edgeEnd(); ii != ei; ++ii) {
      ii->destroy();
    }
  }

  node_data_reference getData(GraphNode N, MethodFlag mflag = MethodFlag::WRITE) {
    // galois::runtime::checkWrite(mflag, false);
    acquireNode(N, mflag);
    return N->getData();
  }

  edge_data_reference getEdgeData(edge_iterator ni, MethodFlag mflag = MethodFlag::UNPROTECTED) const {
    // galois::runtime::checkWrite(mflag, false);
    return ni->get();
   }

  GraphNode getEdgeDst(edge_iterator ni) const {
    return getDst(ni);
  }

  uint64_t size() const { return numNodes; }
  uint64_t sizeEdges() const { return numEdges; }

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
    if (galois::runtime::shouldLock(mflag)) {
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

  detail::EdgesIterator<LC_PartitionedInlineEdge_Graph> out_edges(GraphNode N, MethodFlag mflag = MethodFlag::WRITE) {
    return detail::EdgesIterator<LC_PartitionedInlineEdge_Graph>(*this, N, mflag);
  }

  size_t idFromNode(GraphNode N) {
    return getId(N);
  }

  GraphNode nodeFromId(size_t N) {
    return getNode(N);
  }

  void allocateFrom(FileGraph& graph) {
    numNodes = graph.size();
    numEdges = graph.sizeEdges();

    if (UseNumaAlloc) {
      nodeData.allocateLocal(numNodes);
      edgeData.allocateLocal(numEdges);
      this->outOfLineAllocateLocal(numNodes);
    } else {
      nodeData.allocateInterleaved(numNodes);
      edgeData.allocateInterleaved(numEdges);
      this->outOfLineAllocateInterleaved(numNodes);
    }
  }

  void constructFrom(FileGraph& graph, unsigned tid, unsigned total) {
    // XXX
    auto r = graph.divideByNode(
        NodeData::size_of::value + LC_PartitionedInlineEdge_Graph::size_of_out_of_line::value,
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
