/** Appendable semi-LC graphs -*- C++ -*-
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
 * @author Nikunj Yadav nikunj@cs.utexas.edu
 */

#ifndef GALOIS_GRAPH_LC_MORPH_GRAPH_H
#define GALOIS_GRAPH_LC_MORPH_GRAPH_H

#include "Galois/Bag.h"
#include "Galois/LargeArray.h"
#include "Galois/Graphs/FileGraph.h"
#include "Galois/Graphs/Details.h"

#include <boost/mpl/if.hpp>
#include <type_traits>

namespace galois {
namespace Graph {

//! Local computation graph (i.e., graph structure does not change)
template<typename NodeTy, typename EdgeTy,
  bool HasNoLockable=false,
  bool UseNumaAlloc=false,
  bool HasOutOfLineLockable=false,
  bool HasId=false,
  typename FileEdgeTy=EdgeTy>
class LC_Morph_Graph:
    private boost::noncopyable,
    private detail::OutOfLineLockableFeature<HasOutOfLineLockable && !HasNoLockable> {
  template<typename Graph> friend class LC_InOut_Graph;

public:
  template<bool _has_id>
  struct with_id { typedef LC_Morph_Graph<NodeTy,EdgeTy,HasNoLockable,UseNumaAlloc,HasOutOfLineLockable,_has_id,FileEdgeTy> type; };

  template<typename _node_data>
  struct with_node_data { typedef  LC_Morph_Graph<_node_data,EdgeTy,HasNoLockable,UseNumaAlloc,HasOutOfLineLockable,HasId,FileEdgeTy> type; };

  template<typename _edge_data>
  struct with_edge_data { typedef  LC_Morph_Graph<NodeTy,_edge_data,HasNoLockable,UseNumaAlloc,HasOutOfLineLockable,HasId,FileEdgeTy> type; };

  template<typename _file_edge_data>
  struct with_file_edge_data { typedef  LC_Morph_Graph<NodeTy,EdgeTy,HasNoLockable,UseNumaAlloc,HasOutOfLineLockable,HasId,_file_edge_data> type; };

  template<bool _has_no_lockable>
  struct with_no_lockable { typedef LC_Morph_Graph<NodeTy,EdgeTy,_has_no_lockable,UseNumaAlloc,HasOutOfLineLockable,HasId,FileEdgeTy> type; };

  template<bool _use_numa_alloc>
  struct with_numa_alloc { typedef LC_Morph_Graph<NodeTy,EdgeTy,HasNoLockable,_use_numa_alloc,HasOutOfLineLockable,HasId,FileEdgeTy> type; };

  template<bool _has_out_of_line_lockable>
  struct with_out_of_line_lockable { typedef LC_Morph_Graph<NodeTy,EdgeTy,HasNoLockable,UseNumaAlloc,_has_out_of_line_lockable,_has_out_of_line_lockable||HasId,FileEdgeTy> type; };

  typedef read_with_aux_graph_tag read_tag;

protected:
  class NodeInfo;
  typedef detail::EdgeInfoBase<NodeInfo*, EdgeTy> EdgeInfo;
  typedef galois::InsertBag<NodeInfo> Nodes;
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
#ifndef NDEBUG
    EdgeInfo* trueEdgeEnd;
#endif

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
  typedef FileEdgeTy file_edge_data_type;
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
  galois::Substrate::PerThreadStorage<EdgeHolder*> edgesL;

  template<bool _A1 = HasNoLockable, bool _A2 = HasOutOfLineLockable>
  void acquireNode(GraphNode N, MethodFlag mflag, typename std::enable_if<!_A1 && !_A2>::type* = 0) {
    galois::runtime::acquire(N, mflag);
  }

  template<bool _A1 = HasOutOfLineLockable, bool _A2 = HasNoLockable>
  void acquireNode(GraphNode N, MethodFlag mflag, typename std::enable_if<_A1 && !_A2>::type* = 0) {
    this->outOfLineAcquire(getId(N), mflag);
  }

  template<bool _A1 = EdgeInfo::has_value, bool _A2 = LargeArray<FileEdgeTy>::has_value>
  void constructEdgeValue(FileGraph& graph, typename FileGraph::edge_iterator nn,
      GraphNode src, GraphNode dst, typename std::enable_if<!_A1 || _A2>::type* = 0) {
    typedef typename LargeArray<FileEdgeTy>::value_type FEDV;
    if (EdgeInfo::has_value) {
      addMultiEdge(src, dst, galois::MethodFlag::UNPROTECTED, graph.getEdgeData<FEDV>(nn));
    } else {
      addMultiEdge(src, dst, galois::MethodFlag::UNPROTECTED);
    }
  }

  template<bool _A1 = EdgeInfo::has_value, bool _A2 = LargeArray<FileEdgeTy>::has_value>
  void constructEdgeValue(FileGraph& graph, typename FileGraph::edge_iterator nn,
      GraphNode src, GraphNode dst, typename std::enable_if<_A1 && !_A2>::type* = 0) {
    addMultiEdge(src, dst, galois::MethodFlag::UNPROTECTED);
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

  node_data_reference getData(const GraphNode& N, MethodFlag mflag = MethodFlag::WRITE) {
    // galois::runtime::checkWrite(mflag, false);
    acquireNode(N, mflag);
    return N->getData();
  }

  edge_data_reference getEdgeData(edge_iterator ni, MethodFlag mflag = MethodFlag::UNPROTECTED) {
    // galois::runtime::checkWrite(mflag, false);
    acquireNode(ni->dst, mflag);
    return ni->get();
  }

  GraphNode getEdgeDst(edge_iterator ni) {
    //galois::runtime::checkWrite(mflag, false);
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

  edge_iterator edge_begin(GraphNode N, MethodFlag mflag = MethodFlag::WRITE) {
    acquireNode(N, mflag);
    if (galois::runtime::shouldLock(mflag)) {
      for (edge_iterator ii = N->edgeBegin, ee = N->edgeEnd; ii != ee; ++ii) {
        acquireNode(ii->dst, mflag);
      }
    }
    return N->edgeBegin;
  }

  edge_iterator edge_end(GraphNode N, MethodFlag mflag = MethodFlag::WRITE) {
    return N->edgeEnd;
  }

  runtime::iterable<NoDerefIterator<edge_iterator>> edges(GraphNode N, MethodFlag mflag = MethodFlag::WRITE) {
    return detail::make_no_deref_range(edge_begin(N, mflag), edge_end(N, mflag));
  }

  /**
   * An object with begin() and end() methods to iterate over the outgoing
   * edges of N.
   */
  detail::EdgesIterator<LC_Morph_Graph> out_edges(GraphNode N, MethodFlag mflag = MethodFlag::WRITE) {
    return detail::EdgesIterator<LC_Morph_Graph>(*this, N, mflag);
  }
  
  template<typename... Args>
  GraphNode createNode(int nedges, Args&&... args) {
    // galois::runtime::checkWrite(MethodFlag::WRITE, true);
    NodeInfo* N = &nodes.emplace(std::forward<Args>(args)...);
    acquireNode(N, MethodFlag::WRITE);
    EdgeHolder*& local_edges = *edgesL.getLocal();
    if (!local_edges || std::distance(local_edges->begin, local_edges->end) < nedges) {
      EdgeHolder* old = local_edges;
      //FIXME: this seems to leak
      char* newblock = (char*)runtime::pagePoolAlloc();
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
      char* eend = newblock + runtime::pagePoolSize();
      eend -= (uintptr_t)eend % sizeof(EdgeInfo);
      local_edges->end = (EdgeInfo*)eend;
    }
    N->edgeBegin = N->edgeEnd = local_edges->begin;
    local_edges->begin += nedges;
#ifndef NDEBUG
    N->trueEdgeEnd = local_edges->begin;
#endif
    return GraphNode(N);
  }

  template<typename... Args>
  edge_iterator addEdge(GraphNode src, GraphNode dst, galois::MethodFlag mflag, Args&&... args) {
    // galois::runtime::checkWrite(mflag, true);
    acquireNode(src, mflag);
    auto it = std::find_if(src->edgeBegin, src->edgeEnd, dst_equals(dst));
    if (it == src->edgeEnd) {
      it->dst = dst;
      it->construct(std::forward<Args>(args)...);
      src->edgeEnd++;
      assert(src->edgeEnd <= src->trueEdgeEnd);
    }
    return it;
  }

  template<typename... Args>
  edge_iterator addMultiEdge(GraphNode src, GraphNode dst, galois::MethodFlag mflag, Args&&... args) {
    // galois::runtime::checkWrite(mflag, true);
    acquireNode(src, mflag);
    auto it = src->edgeEnd;
    it->dst = dst;
    it->construct(std::forward<Args>(args)...);
    src->edgeEnd++;
    assert(src->edgeEnd <= src->trueEdgeEnd);
    return it;
  }

  /**
   * Remove an edge from the graph.
   *
   * Invalidates edge iterator.
   */
  void removeEdge(GraphNode src, edge_iterator dst, galois::MethodFlag mflag = MethodFlag::WRITE) {
    // galois::runtime::checkWrite(mflag, true);
    acquireNode(src, mflag);
    src->edgeEnd--;
    assert(src->edgeBegin <= src->edgeEnd);
    std::swap(*dst, *src->edgeEnd);
    src->edgeEnd->destroy();
  }
  
  edge_iterator findEdge(GraphNode src, GraphNode dst, galois::MethodFlag mflag = MethodFlag::WRITE) {
    // galois::runtime::checkWrite(mflag, true); // TODO: double check 'true' here
    acquireNode(src, mflag);
    return std::find_if(src->edgeBegin, src->edgeEnd, dst_equals(dst)); 
  }
  
  void allocateFrom(FileGraph& graph, ReadGraphAuxData& aux) {
    size_t numNodes = graph.size();
    
    if (UseNumaAlloc) {
      aux.allocateLocal(numNodes);
      this->outOfLineAllocateLocal(numNodes);
    } else {
      aux.allocateInterleaved(numNodes);
      this->outOfLineAllocateInterleaved(numNodes);
    }
  }

  void constructNodesFrom(FileGraph& graph, unsigned tid, unsigned total, ReadGraphAuxData& aux) {
    auto r = graph.divideByNode(
        sizeof(NodeInfo) + LC_Morph_Graph::size_of_out_of_line::value,
        sizeof(EdgeInfo),
        tid, total).first;

    size_t id = *r.first;
    for (FileGraph::iterator ii = r.first, ei = r.second; ii != ei; ++ii, ++id) {
      aux[id] = createNode(std::distance(graph.edge_begin(*ii), graph.edge_end(*ii)));
    }
  }
  
  void constructEdgesFrom(FileGraph& graph, unsigned tid, unsigned total, const ReadGraphAuxData& aux) {
    typedef typename EdgeInfo::value_type value_type;
    typedef LargeArray<FileEdgeTy> FED;
    auto r = graph.divideByNode(
        sizeof(NodeInfo) + LC_Morph_Graph::size_of_out_of_line::value,
        sizeof(EdgeInfo),
        tid, total).first;

    for (FileGraph::iterator ii = r.first, ei = r.second; ii != ei; ++ii) {
      for (FileGraph::edge_iterator nn = graph.edge_begin(*ii), en = graph.edge_end(*ii); nn != en; ++nn) {
        constructEdgeValue(graph, nn, aux[*ii], aux[graph.getEdgeDst(nn)]);
      }
    }
  }
};

} // end namespace
} // end namespace

#endif /* LC_MORPH_GRAPH_H_ */
