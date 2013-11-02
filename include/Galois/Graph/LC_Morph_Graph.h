/** Appendable semi-LC graphs -*- C++ -*-
 * @file
 * @section License
 *
 * Graph which is like other LC graphs but allows adding edges.
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
 * @author Nikunj Yadav nikunj@cs.utexas.edu
 */
#ifndef GALOIS_GRAPH_LC_MORPH_GRAPH_H
#define GALOIS_GRAPH_LC_MORPH_GRAPH_H

#include "Galois/config.h"
#include "Galois/Bag.h"
#include "Galois/LargeArray.h"
#include "Galois/Graph/FileGraph.h"
#include "Galois/Graph/Details.h"
#include "Galois/Runtime/MethodFlags.h"

#include <boost/mpl/if.hpp>
#include GALOIS_CXX11_STD_HEADER(type_traits)

namespace Galois {
namespace Graph {

//! Local computation graph (i.e., graph structure does not change)
template<typename NodeTy, typename EdgeTy,
  bool HasNoLockable=false,
  bool UseNumaAlloc=false,
  bool HasOutOfLineLockable=false,
  bool HasId=false>
class LC_Morph_Graph:
    private boost::noncopyable,
    private detail::OutOfLineLockableFeature<HasOutOfLineLockable && !HasNoLockable> {
  template<typename Graph> friend class LC_InOut_Graph;

public:
  template<bool _has_id>
  struct with_id { typedef LC_Morph_Graph<NodeTy,EdgeTy,HasNoLockable,UseNumaAlloc,HasOutOfLineLockable,_has_id> type; };

  template<typename _node_data>
  struct with_node_data { typedef  LC_Morph_Graph<_node_data,EdgeTy,HasNoLockable,UseNumaAlloc,HasOutOfLineLockable,HasId> type; };

  template<bool _has_no_lockable>
  struct with_no_lockable { typedef LC_Morph_Graph<NodeTy,EdgeTy,_has_no_lockable,UseNumaAlloc,HasOutOfLineLockable,HasId> type; };

  template<bool _use_numa_alloc>
  struct with_numa_alloc { typedef LC_Morph_Graph<NodeTy,EdgeTy,HasNoLockable,_use_numa_alloc,HasOutOfLineLockable,HasId> type; };

  template<bool _has_out_of_line_lockable>
  struct with_out_of_line_lockable { typedef LC_Morph_Graph<NodeTy,EdgeTy,HasNoLockable,UseNumaAlloc,_has_out_of_line_lockable,_has_out_of_line_lockable||HasId> type; };

  typedef read_with_aux_graph_tag read_tag;

protected:
  class NodeInfo;
  typedef detail::EdgeInfoBase<NodeInfo*, EdgeTy> EdgeInfo;
  typedef Galois::InsertBag<NodeInfo> Nodes;
  typedef detail::NodeInfoBaseTypes<NodeTy,!HasNoLockable && !HasOutOfLineLockable> NodeInfoTypes;
  
  struct EdgeHolder {
    EdgeInfo* begin;
    EdgeInfo* end;
    EdgeHolder* next;
  };

  class NodeInfo: public detail::NodeInfoBase<NodeTy,!HasNoLockable && !HasOutOfLineLockable> {
    typedef detail::NodeInfoBase<NodeTy,!HasNoLockable && !HasOutOfLineLockable> Super;
    friend class LC_Morph_Graph;

    EdgeInfo* edgeBegin;
    EdgeInfo* edgeEnd;

  public:
    template<typename... Args>
    NodeInfo(Args&&... args): Super(std::forward<Args>(args)...) { }
  };

  struct makeGraphNode: public std::unary_function<NodeInfo&, NodeInfo*> {
    NodeInfo* operator()(NodeInfo& data) const { return &data; }
  };
  
  struct dst_equals {
    NodeInfo* dst;
    dst_equals(NodeInfo* d): dst(d) { }
    bool operator()(const EdgeInfo& edge) { return edge.dst == dst; }
  };

public:
  typedef NodeInfo* GraphNode;
  typedef EdgeTy edge_data_type;
  typedef NodeTy node_data_type;
  typedef typename NodeInfoTypes::reference node_data_reference;
  typedef typename EdgeInfo::reference edge_data_reference;
  typedef EdgeInfo* edge_iterator;
  typedef boost::transform_iterator<makeGraphNode,typename Nodes::iterator> iterator;
  typedef boost::transform_iterator<makeGraphNode,typename Nodes::const_iterator> const_iterator;
  typedef iterator local_iterator;
  typedef const_iterator const_local_iterator;
  typedef LargeArray<GraphNode> ReadGraphAuxData;

protected:
  Nodes nodes;
  Galois::Runtime::PerThreadStorage<EdgeHolder*> edges;

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

  template<bool _Enable = HasId>
  size_t getId(GraphNode N, typename std::enable_if<_Enable>::type* = 0) {
    return N->getId();
  }

public:
  ~LC_Morph_Graph() {
    for (typename Nodes::iterator ii = nodes.begin(), ei = nodes.end(); ii != ei; ++ii) {
      NodeInfo& n = *ii;
      EdgeInfo* edgeBegin = n.edgeBegin;
      EdgeInfo* edgeEnd = n.edgeEnd;

      if (EdgeInfo::has_value) {
        while (edgeBegin != edgeEnd) {
          edgeBegin->destroy();
          ++edgeBegin;
        }
      }
    }
  }

  node_data_reference getData(const GraphNode& N, MethodFlag mflag = MethodFlag::ALL) {
    Galois::Runtime::checkWrite(mflag, false);
    acquireNode(N, mflag);
    return N->getData();
  }

  edge_data_reference getEdgeData(edge_iterator ni, MethodFlag mflag = MethodFlag::NONE) {
    Galois::Runtime::checkWrite(mflag, false);
    acquireNode(ni->dst, mflag);
    return ni->get();
  }

  GraphNode getEdgeDst(edge_iterator ni) {
    //Galois::Runtime::checkWrite(mflag, false);
    //acquireNode(ni->dst, mflag);
    return GraphNode(ni->dst);
  }

  /**
   * Returns an iterator to all the nodes in the graph. Not thread-safe.
   */
  iterator begin() {
    return boost::make_transform_iterator(nodes.begin(), makeGraphNode());
  }

  //! Returns the end of the node iterator. Not thread-safe.
  iterator end() {
    return boost::make_transform_iterator(nodes.end(), makeGraphNode());
  }

  local_iterator local_begin() {
    return boost::make_transform_iterator(nodes.local_begin(), makeGraphNode());
  }
  
  local_iterator local_end() {
    return boost::make_transform_iterator(nodes.local_end(), makeGraphNode());
  }

  edge_iterator edge_begin(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    acquireNode(N, mflag);
    if (Galois::Runtime::shouldLock(mflag)) {
      for (edge_iterator ii = N->edgeBegin, ee = N->edgeEnd; ii != ee; ++ii) {
        acquireNode(ii->dst, mflag);
      }
    }
    return N->edgeBegin;
  }

  edge_iterator edge_end(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    return N->edgeEnd;
  }
  
  template<typename... Args>
  GraphNode createNode(int nedges, Args&&... args) {
    Galois::Runtime::checkWrite(MethodFlag::ALL, true);
    NodeInfo* N = &nodes.emplace(std::forward<Args>(args)...);
    acquireNode(N, MethodFlag::ALL);
    EdgeHolder*& local_edges = *edges.getLocal();
    if (!local_edges || std::distance(local_edges->begin, local_edges->end) < nedges) {
      EdgeHolder* old = local_edges;
      char* newblock = (char*)Runtime::MM::pageAlloc();
      local_edges = (EdgeHolder*)newblock;
      local_edges->next = old;
      char* estart = newblock + sizeof(EdgeHolder);
      if ((uintptr_t)estart % sizeof(EdgeInfo)) // Not aligned
#ifdef HAVE_CXX11_ALIGNOF
        estart += sizeof(EdgeInfo) - ((uintptr_t)estart % alignof(EdgeInfo));
#else
        estart += sizeof(EdgeInfo) - ((uintptr_t)estart % 8);
#endif

      local_edges->begin = (EdgeInfo*)estart;
      char* eend = newblock + Runtime::MM::pageSize;
      eend -= (uintptr_t)eend % sizeof(EdgeInfo);
      local_edges->end = (EdgeInfo*)eend;
    }
    N->edgeBegin = N->edgeEnd = local_edges->begin;
    local_edges->begin += nedges;
    return GraphNode(N);
  }

  template<typename... Args>
  edge_iterator addEdge(GraphNode src, GraphNode dst, Galois::MethodFlag mflag, Args&&... args) {
    Galois::Runtime::checkWrite(mflag, true);
    acquireNode(src, mflag);
    auto it = std::find_if(src->edgeBegin, src->edgeEnd, dst_equals(dst));
    if (it == src->edgeEnd) {
      it->dst = dst;
      it->construct(std::forward<Args>(args)...);
      src->edgeEnd++;
    }
    return it;
  }

  template<typename... Args>
  edge_iterator addEdgeWithoutCheck(GraphNode src, GraphNode dst, Galois::MethodFlag mflag, Args&&... args) {
    Galois::Runtime::checkWrite(mflag, true);
    acquireNode(src, mflag);
    auto it = src->edgeEnd;
    it->dst = dst;
    it->construct(std::forward<Args>(args)...);
    src->edgeEnd++;
    return it;
  }
  
  edge_iterator findEdge(GraphNode src, GraphNode dst, Galois::MethodFlag mflag = MethodFlag::ALL) {
    Galois::Runtime::checkWrite(mflag, true);
    acquireNode(src, mflag);
    return std::find_if(src->edgeBegin, src->edgeEnd, dst_equals(dst)); 
  }
  
  void allocateFrom(FileGraph& graph, ReadGraphAuxData& aux) {
    size_t numNodes = graph.size();
    
    if (UseNumaAlloc) {
      aux.allocateLocal(numNodes, false);
      this->outOfLineAllocateLocal(numNodes, false);
    } else {
      aux.allocateInterleaved(numNodes);
      this->outOfLineAllocateInterleaved(numNodes);
    }
  }

  void constructNodesFrom(FileGraph& graph, unsigned tid, unsigned total, ReadGraphAuxData& aux) {
    auto r = graph.divideBy(
        sizeof(NodeInfo) + LC_Morph_Graph::size_of_out_of_line::value,
        sizeof(EdgeInfo),
        tid, total);

    size_t id = *r.first;
    for (FileGraph::iterator ii = r.first, ei = r.second; ii != ei; ++ii, ++id) {
      aux[id] = createNode(std::distance(graph.edge_begin(*ii), graph.edge_end(*ii)));
    }
  }
  
  void constructEdgesFrom(FileGraph& graph, unsigned tid, unsigned total, const ReadGraphAuxData& aux) {
    auto r = graph.divideBy(
        sizeof(NodeInfo) + LC_Morph_Graph::size_of_out_of_line::value,
        sizeof(EdgeInfo),
        tid, total);

    for (FileGraph::iterator ii = r.first, ei = r.second; ii != ei; ++ii) {
      for (FileGraph::edge_iterator nn = graph.edge_begin(*ii), en = graph.edge_end(*ii); nn != en; ++nn) {
        if (EdgeInfo::has_value) {
          addEdgeWithoutCheck(aux[*ii], aux[graph.getEdgeDst(nn)], Galois::MethodFlag::NONE, graph.getEdgeData<uint32_t>(nn));
        } else {
          addEdgeWithoutCheck(aux[*ii], aux[graph.getEdgeDst(nn)], Galois::MethodFlag::NONE);
        }
      }
    }
  }
};

} // end namespace
} // end namespace

#endif /* LC_MORPH_GRAPH_H_ */
