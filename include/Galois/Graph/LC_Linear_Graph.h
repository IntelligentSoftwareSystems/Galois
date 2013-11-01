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
#ifndef GALOIS_GRAPH_LC_LINEAR_GRAPH_H
#define GALOIS_GRAPH_LC_LINEAR_GRAPH_H

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
 * Local computation graph (i.e., graph structure does not change). The data
 * representation is a modification of {@link LC_CSR_Graph} where the edge data
 * and node data is stored inline with the adjacency information. 
 *
 * The position of template parameters may change between Galois releases; the
 * most robust way to specify them is through the with_XXX nested templates.
 */
template<typename NodeTy, typename EdgeTy,
  bool HasNoLockable=false,
  bool UseNumaAlloc=false,
  bool HasOutOfLineLockable=false,
  bool HasId=false>
class LC_Linear_Graph:
    private boost::noncopyable,
    private detail::LocalIteratorFeature<UseNumaAlloc>,
    private detail::OutOfLineLockableFeature<HasOutOfLineLockable && !HasNoLockable> {
  template<typename Graph> friend class LC_InOut_Graph;

public:
  template<bool _has_id>
  struct with_id { typedef LC_Linear_Graph<NodeTy,EdgeTy,HasNoLockable,UseNumaAlloc,HasOutOfLineLockable,_has_id> type; };

  template<typename _node_data>
  struct with_node_data { typedef  LC_Linear_Graph<_node_data,EdgeTy,HasNoLockable,UseNumaAlloc,HasOutOfLineLockable,HasId> type; };

  template<bool _has_no_lockable>
  struct with_no_lockable { typedef LC_Linear_Graph<NodeTy,EdgeTy,_has_no_lockable,UseNumaAlloc,HasOutOfLineLockable,HasId> type; };

  template<bool _use_numa_alloc>
  struct with_numa_alloc { typedef LC_Linear_Graph<NodeTy,EdgeTy,HasNoLockable,_use_numa_alloc,HasOutOfLineLockable,HasId> type; };

  template<bool _has_out_of_line_lockable>
  struct with_out_of_line_lockable { typedef LC_Linear_Graph<NodeTy,EdgeTy,HasNoLockable,UseNumaAlloc,_has_out_of_line_lockable,_has_out_of_line_lockable||HasId> type; };

  typedef read_with_aux_graph_tag read_tag;

protected:
  class NodeInfo;
  typedef detail::EdgeInfoBase<NodeInfo*,EdgeTy> EdgeInfo;
  typedef LargeArray<NodeInfo*> Nodes;
  typedef detail::NodeInfoBaseTypes<NodeTy,!HasNoLockable && !HasOutOfLineLockable> NodeInfoTypes;

  class NodeInfo:
      public detail::NodeInfoBase<NodeTy,!HasNoLockable && !HasOutOfLineLockable>,
      public detail::IntrusiveId<typename boost::mpl::if_c<HasId,uint32_t,void>::type> {
    friend class LC_Linear_Graph;
    int numEdges;

    EdgeInfo* edgeBegin() {
      NodeInfo* n = this;
      ++n; //start of edges
      return reinterpret_cast<EdgeInfo*>(n);
    }

    EdgeInfo* edgeEnd() {
      EdgeInfo* ei = edgeBegin();
      ei += numEdges;
      return ei;
    }

    NodeInfo* next() {
      NodeInfo* ni = this;
      EdgeInfo* ei = edgeEnd();
      while (reinterpret_cast<char*>(ni) < reinterpret_cast<char*>(ei))
        ++ni;
      return ni;
    }
  };

public:
  typedef NodeInfo* GraphNode;
  typedef EdgeTy edge_data_type;
  typedef NodeTy node_data_type;
  typedef typename NodeInfoTypes::reference node_data_reference;
  typedef typename EdgeInfo::reference edge_data_reference;
  typedef EdgeInfo* edge_iterator;
  typedef NodeInfo** iterator;
  typedef NodeInfo*const * const_iterator;
  typedef iterator local_iterator;
  typedef const_iterator const_local_iterator;
  typedef int ReadGraphAuxData;

protected:
  LargeArray<char> data;
  uint64_t numNodes;
  uint64_t numEdges;
  Nodes nodes;

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
    return N->edgeBegin();
  }

  edge_iterator raw_end(GraphNode N) {
    return N->edgeEnd();
  }

  template<bool _Enable = HasId>
  size_t getId(GraphNode N, typename std::enable_if<_Enable>::type* = 0) {
    return N->getId();
  }

  template<bool _Enable = HasId>
  GraphNode getNode(size_t n, typename std::enable_if<_Enable>::type* = 0) {
    return nodes[n];
  }

public:
  ~LC_Linear_Graph() { 
    for (typename Nodes::iterator ii = nodes.begin(), ei = nodes.end(); ii != ei; ++ii) {
      NodeInfo* n = *ii;
      EdgeInfo* edgeBegin = n->edgeBegin();
      EdgeInfo* edgeEnd = n->edgeEnd();

      if (EdgeInfo::has_value) {
        while (edgeBegin != edgeEnd) {
          edgeBegin->destroy();
          ++edgeBegin;
        }
      }
      n->~NodeInfo();
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
    return ni->dst;
  }

  uint64_t size() const { return numNodes; }
  uint64_t sizeEdges() const { return numEdges; }
  iterator begin() { return &nodes[0]; }
  iterator end() { return &nodes[numNodes]; }
  const_iterator begin() const { return &nodes[0]; }
  const_iterator end() const { return &nodes[numNodes]; }

  local_iterator local_begin() { return &nodes[this->localBegin(numNodes)]; }
  local_iterator local_end() { return &nodes[this->localEnd(numNodes)]; }
  const_local_iterator local_begin() const { return &nodes[this->localBegin(numNodes)]; }
  const_local_iterator local_end() const { return &nodes[this->localEnd(numNodes)]; }

  edge_iterator edge_begin(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    acquireNode(N, mflag);
    if (Galois::Runtime::shouldLock(mflag)) {
      for (edge_iterator ii = N->edgeBegin(), ee = N->edgeEnd(); ii != ee; ++ii) {
        acquireNode(ii->dst, mflag);
      }
    }
    return N->edgeBegin();
  }

  edge_iterator edge_end(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    acquireNode(N, mflag);
    return N->edgeEnd();
  }

  detail::EdgesIterator<LC_Linear_Graph> out_edges(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    return detail::EdgesIterator<LC_Linear_Graph>(*this, N, mflag);
  }

  /**
   * Sorts outgoing edges of a node. Comparison function is over EdgeTy.
   */
  template<typename CompTy>
  void sortEdgesByEdgeData(GraphNode N, const CompTy& comp = std::less<EdgeTy>(), MethodFlag mflag = MethodFlag::ALL) {
    acquireNode(N, mflag);
    std::sort(N->edgeBegin(), N->edgeEnd(), detail::EdgeSortCompWrapper<EdgeInfo,CompTy>(comp));
  }

  /**
   * Sorts outgoing edges of a node. Comparison function is over <code>EdgeSortValue<EdgeTy></code>.
   */
  template<typename CompTy>
  void sortEdges(GraphNode N, const CompTy& comp, MethodFlag mflag = MethodFlag::ALL) {
    acquireNode(N, mflag);
    std::sort(N->edgeBegin(), N->edgeEnd(), comp);
  }

  void allocateFrom(FileGraph& graph, const ReadGraphAuxData&) {
    numNodes = graph.size();
    numEdges = graph.sizeEdges();
    if (UseNumaAlloc) {
      data.allocateLocal(sizeof(NodeInfo) * numNodes * 2 + sizeof(EdgeInfo) * numEdges, false);
      nodes.allocateLocal(numNodes, false);
      this->outOfLineAllocateLocal(numNodes, false);
    } else {
      data.allocateInterleaved(sizeof(NodeInfo) * numNodes * 2 + sizeof(EdgeInfo) * numEdges);
      nodes.allocateInterleaved(numNodes);
      this->outOfLineAllocateInterleaved(numNodes);
    }
  }

  void constructNodesFrom(FileGraph& graph, unsigned tid, unsigned total, const ReadGraphAuxData&) {
    auto r = graph.divideBy(
        Nodes::size_of::value + 2 * sizeof(NodeInfo) + LC_Linear_Graph::size_of_out_of_line::value,
        sizeof(EdgeInfo),
        tid, total);

    this->setLocalRange(*r.first, *r.second);
    NodeInfo* curNode = reinterpret_cast<NodeInfo*>(data.data());

    size_t id = *r.first;
    size_t edges = *graph.edge_begin(*r.first);
    size_t bytes = edges * sizeof(EdgeInfo) + 2 * (id + 1) * sizeof(NodeInfo);
    curNode += bytes / sizeof(NodeInfo);
    for (FileGraph::iterator ii = r.first, ei = r.second; ii != ei; ++ii, ++id) {
      nodes.constructAt(*ii);
      new (curNode) NodeInfo();
      //curNode->construct();
      curNode->setId(id);
      curNode->numEdges = std::distance(graph.edge_begin(*ii), graph.edge_end(*ii));
      nodes[*ii] = curNode;
      curNode = curNode->next();
    }
  }

  void constructEdgesFrom(FileGraph& graph, unsigned tid, unsigned total, const ReadGraphAuxData&) {
    typedef typename EdgeInfo::value_type EDV;
    auto r = graph.divideBy(
        Nodes::size_of::value + 2 * sizeof(NodeInfo) + LC_Linear_Graph::size_of_out_of_line::value,
        sizeof(EdgeInfo),
        tid, total);

    for (FileGraph::iterator ii = r.first, ei = r.second; ii != ei; ++ii) {
      EdgeInfo* edge = nodes[*ii]->edgeBegin();
      for (FileGraph::edge_iterator nn = graph.edge_begin(*ii), en = graph.edge_end(*ii); nn != en; ++nn) {
        if (EdgeInfo::has_value)
          edge->construct(graph.getEdgeData<EDV>(nn));
        edge->dst = nodes[graph.getEdgeDst(nn)];
        ++edge;
      }
    }
  }
};

} // end namespace
} // end namespace

#endif
