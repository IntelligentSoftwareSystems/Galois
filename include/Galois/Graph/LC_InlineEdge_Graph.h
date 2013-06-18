/** Local Computation graphs -*- C++ -*-
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
 * @author Andrew Lenharth <andrewl@lenharth.org>
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#ifndef GALOIS_GRAPH_LC_INLINEEDGE_GRAPH_H
#define GALOIS_GRAPH_LC_INLINEEDGE_GRAPH_H

#include "Galois/config.h"
#include "Galois/LargeArray.h"
#include "Galois/Graph/FileGraph.h"
#include "Galois/Graph/Details.h"
#include "Galois/Runtime/MethodFlags.h"

#include <boost/mpl/if.hpp>
#include GALOIS_CXX11_STD_HEADER(type_traits)

namespace Galois {
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
  bool HasCompressedNodePtr=false>
class LC_InlineEdge_Graph:
    private boost::noncopyable,
    private detail::LocalIteratorFeature<UseNumaAlloc>,
    private detail::OutOfLineLockableFeature<HasOutOfLineLockable && !HasNoLockable> {
  template<typename Graph> friend class LC_InOut_Graph;

public:
  template<bool _has_id>
  struct with_id { typedef LC_InlineEdge_Graph type; };

  template<typename _node_data>
  struct with_node_data { typedef LC_InlineEdge_Graph<_node_data,EdgeTy,HasNoLockable,UseNumaAlloc,HasOutOfLineLockable,HasCompressedNodePtr> type; };

  template<bool _has_no_lockable>
  struct with_no_lockable { typedef LC_InlineEdge_Graph<NodeTy,EdgeTy,_has_no_lockable,UseNumaAlloc,HasOutOfLineLockable,HasCompressedNodePtr> type; };

  template<bool _use_numa_alloc>
  struct with_numa_alloc { typedef LC_InlineEdge_Graph<NodeTy,EdgeTy,HasNoLockable,_use_numa_alloc,HasOutOfLineLockable,HasCompressedNodePtr> type; };

  template<bool _has_out_of_line_lockable>
  struct with_out_of_line_lockable { typedef LC_InlineEdge_Graph<NodeTy,EdgeTy,HasNoLockable,UseNumaAlloc,_has_out_of_line_lockable,HasCompressedNodePtr> type; };

  /**
   * Compress representation of graph at the expense of one level of indirection on accessing
   * neighbors of a node
   */
  template<bool _has_compressed_node_ptr>
  struct with_compressed_node_ptr { typedef  LC_InlineEdge_Graph<NodeTy,EdgeTy,HasNoLockable,UseNumaAlloc,HasOutOfLineLockable,_has_compressed_node_ptr> type; };

  typedef read_default_graph_tag read_tag;

protected:
  struct NodeInfo;
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
  typedef NodeTy node_data_type;
  typedef typename EdgeInfo::reference edge_data_reference;
  typedef typename NodeInfo::reference node_data_reference;
  typedef EdgeInfo* edge_iterator;
  typedef Galois::NoDerefIterator<NodeInfo*> iterator;
  typedef Galois::NoDerefIterator<const NodeInfo*> const_iterator;
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
    Galois::Runtime::acquire(N, mflag);
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

  node_data_reference getData(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    Galois::Runtime::checkWrite(mflag, false);
    acquireNode(N, mflag);
    return N->getData();
  }

  edge_data_reference getEdgeData(edge_iterator ni, MethodFlag mflag = MethodFlag::NONE) const {
    Galois::Runtime::checkWrite(mflag, false);
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

  edge_iterator edge_begin(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    acquireNode(N, mflag);
    if (Galois::Runtime::shouldLock(mflag)) {
      for (edge_iterator ii = N->edgeBegin(), ee = N->edgeEnd(); ii != ee; ++ii) {
        acquireNode(getDst(ii), mflag);
      }
    }
    return N->edgeBegin();
  }

  edge_iterator edge_end(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    acquireNode(N, mflag);
    return N->edgeEnd();
  }

  detail::EdgesIterator<LC_InlineEdge_Graph> out_edges(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    return detail::EdgesIterator<LC_InlineEdge_Graph>(*this, N, mflag);
  }

#if 0
  /**
   * Sorts outgoing edges of a node. Comparison function is over EdgeTy.
   */
  template<typename CompTy>
  void sortEdgesByEdgeData(GraphNode N, const CompTy& comp = std::less<EdgeTy>(), MethodFlag mflag = MethodFlag::ALL) {
    Galois::Runtime::acquire(N, mflag);
    std::sort(edge_sort_begin(N), edge_sort_end(N), EdgeSortCompWrapper<EdgeSortValue<GraphNode,EdgeTy>,CompTy>(comp));
  }

  /**
   * Sorts outgoing edges of a node. Comparison function is over <code>EdgeSortValue<EdgeTy></code>.
   */
  template<typename CompTy>
  void sortEdges(GraphNode N, const CompTy& comp, MethodFlag mflag = MethodFlag::ALL) {
    Galois::Runtime::acquire(N, mflag);
    std::sort(edge_sort_begin(N), edge_sort_end(N), comp);
  }
#endif

  void allocateFrom(FileGraph& graph) {
    numNodes = graph.size();
    numEdges = graph.sizeEdges();

    if (UseNumaAlloc) {
      nodeData.allocateLocal(numNodes, false);
      edgeData.allocateLocal(numEdges, false);
      this->outOfLineAllocateLocal(numNodes, false);
    } else {
      nodeData.allocateInterleaved(numNodes);
      edgeData.allocateInterleaved(numEdges);
      this->outOfLineAllocateInterleaved(numNodes);
    }
  }

  void constructFrom(FileGraph& graph, unsigned tid, unsigned total) {
    typedef typename EdgeInfo::value_type EDV;
    auto r = graph.divideBy(
        NodeData::sizeof_value + LC_InlineEdge_Graph::sizeof_out_of_line_value,
        EdgeData::sizeof_value,
        tid, total);

    EdgeInfo* curEdge = edgeData.data() + *graph.edge_begin(*r.first);

    this->setLocalRange(*r.first, *r.second);

    for (FileGraph::iterator ii = r.first, ei = r.second; ii != ei; ++ii) {
      nodeData.constructAt(*ii);
      this->outOfLineConstructAt(*ii);
      nodeData[*ii].edgeBegin() = curEdge;
      for (FileGraph::edge_iterator nn = graph.edge_begin(*ii), en = graph.edge_end(*ii); nn != en; ++nn) {
        if (EdgeInfo::has_value)
          curEdge->construct(graph.getEdgeData<EDV>(nn));
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
