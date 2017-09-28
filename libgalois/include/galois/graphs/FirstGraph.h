/** Basic morph graphs -*- C++ -*-
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

#ifndef GALOIS_GRAPH_FIRSTGRAPH_H
#define GALOIS_GRAPH_FIRSTGRAPH_H

//#define AUX_MAP

#include "galois/Bag.h"
#include "galois/gstl.h"
#ifdef AUX_MAP
  #include "galois/PerThreadContainer.h"
#else
  #include "galois/substrate/CacheLineStorage.h"
  #include "galois/substrate/SimpleLock.h"
#endif
#include "galois/graphs/FileGraph.h"
#include "galois/graphs/Details.h"
#include "galois/Galois.h"

#include "llvm/ADT/SmallVector.h"

#include <boost/functional.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/filter_iterator.hpp>

#include <algorithm>
#include <map>
#include <set>
#include <type_traits>
#include <vector>

namespace galois {
//! Parallel graph data structures.
namespace graphs {

namespace internal {
/**
 * Wrapper class to have a valid type on void edges
 */
template<typename NTy, typename ETy, bool DirectedButNotInOut>
struct UEdgeInfoBase;

template<typename NTy, typename ETy>
struct UEdgeInfoBase<NTy, ETy, true> {
  typedef ETy& reference;

  NTy* N;
  ETy Ea;

  inline NTy*       first()       { assert(N); return N; }
  inline NTy* const first() const { assert(N); return N; }
  inline ETy*       second()       { return &Ea; }
  inline const ETy* second() const { return &Ea; }

  template<typename... Args>
  UEdgeInfoBase(NTy* n, ETy* v, bool f, Args&&... args) : N(n), Ea(std::forward<Args>(args)...) {}

  template<typename... Args>
  UEdgeInfoBase(NTy* n, ETy &v, bool f, Args&&... args): N(n) { Ea = v; }

  static size_t sizeOfSecond()     { return sizeof(ETy); }
  bool isInEdge() const { return false; }
};

template<typename NTy, typename ETy>
struct UEdgeInfoBase<NTy, ETy, false> {
  typedef ETy& reference;
  
  NTy* N;
  ETy* Ea;

  inline NTy*       first()       { assert(N); return (NTy*)((uintptr_t)N&~1); }
  inline NTy* const first() const { assert(N); return (NTy*)((uintptr_t)N&~1); }
  inline ETy*       second()       { return Ea; }
  inline const ETy* second() const { return Ea; }
  template<typename... Args>
  UEdgeInfoBase(NTy* n, ETy* v, bool f, Args&&... args) : N((NTy*)((uintptr_t)n|f)), Ea(v) {}
  static size_t sizeOfSecond()     { return sizeof(ETy); }
  bool isInEdge() const { return (uintptr_t)N&1; }
};

template<typename NTy>
struct UEdgeInfoBase<NTy, void, true> {
  typedef char& reference;

  NTy* N;
  inline NTy*       first()        { return N; }
  inline NTy* const first()  const { return N; }
  inline char*       second() const { return static_cast<char*>(NULL); }
  inline char*       addr()   const { return second(); }
  template<typename... Args>
  UEdgeInfoBase(NTy* n, void*, bool f, Args&&... args) : N(n) {}
  static size_t sizeOfSecond()      { return 0; }
  bool isInEdge() const { return false; }
};

template<typename NTy>
struct UEdgeInfoBase<NTy, void, false> {
  typedef char& reference;

  NTy* N;
  inline NTy*       first()        { return (NTy*)((uintptr_t)N&~1); }
  inline NTy* const first()  const { return (NTy*)((uintptr_t)N&~1); }
  inline char*       second() const { return static_cast<char*>(NULL); }
  inline char*       addr()   const { return second(); }
  template<typename... Args>
  UEdgeInfoBase(NTy* n, void*, bool f, Args&&... args) : N((NTy*)((uintptr_t)n|f)) {}
  static size_t sizeOfSecond()      { return 0; }
  bool isInEdge() const { return (uintptr_t)N&1; }
};

template<typename ETy>
struct EdgeFactory {
  galois::runtime::FixedSizeAllocator<ETy> mem;
  template<typename... Args>
  ETy* mkEdge(Args&&... args) {
    ETy* e = mem.allocate(1);
    mem.construct(e, std::forward<Args>(args)...);
    return e;
  }
  void delEdge(ETy* e) {
    mem.destroy(e);
    mem.deallocate(e, 1);
  }
  bool mustDel() const { return true; }
};

template<>
struct EdgeFactory<void> {
  template<typename... Args>
  void* mkEdge(Args&&... args) { return static_cast<void*>(NULL); }
  void delEdge(void*) {}
  bool mustDel() const { return false; }
};

} // end namespace impl

/**
 * A Graph.
 *
 * An example of use:
 * 
 * \code
 * struct Node {
 *   ... // Definition of node data
 * };
 *
 * typedef galois::graphs::FirstGraph<Node,int,true> Graph;
 * 
 * // Create graph
 * Graph g;
 * Node n1, n2;
 * Graph::GraphNode a, b;
 * a = g.createNode(n1);
 * g.addNode(a);
 * b = g.createNode(n2);
 * g.addNode(b);
 * g.getEdgeData(g.addEdge(a, b)) = 5;
 *
 * // Traverse graph
 * for (Graph::iterator ii = g.begin(), ei = g.end(); ii != ei; ++ii) {
 *   Graph::GraphNode src = *ii;
 *   for (Graph::edge_iterator jj = g.edge_begin(src), ej = g.edge_end(src); ++jj) {
 *     Graph::GraphNode dst = graph.getEdgeDst(jj);
 *     int edgeData = g.getEdgeData(jj);
 *     assert(edgeData == 5);
 *   }
 * }
 * \endcode
 *
 * And in C++11:
 *
 * \code
 * // Traverse graph
 * for (Graph::GraphNode src : g) {
 *   for (Graph::edge_iterator edge : g.out_edges(src)) {
 *     Graph::GraphNode dst = g.getEdgeDst(edge);
 *     int edgeData = g.getEdgeData(edge);
 *     assert(edgeData == 5);
 *   }
 * }
 * \endcode
 *
 * @tparam NodeTy Type of node data
 * @tparam EdgeTy Type of edge data
 * @tparam Directional true if graph is directed
 * @tparam InOut true if directed graph tracks in-edges
 * @tparam SortedNeighbors Keep neighbors sorted (for faster findEdge)
 */
template<typename NodeTy, typename EdgeTy, bool Directional, bool InOut = false,
  bool HasNoLockable=false,
  bool SortedNeighbors=false,
  typename FileEdgeTy=EdgeTy
  >
class FirstGraph : private boost::noncopyable {
public:
  //! If true, do not use abstract locks in graph
  template<bool _has_no_lockable>
  struct with_no_lockable { typedef FirstGraph<NodeTy,EdgeTy,Directional,InOut,_has_no_lockable,SortedNeighbors,FileEdgeTy> type; };

  template<typename _node_data>
  struct with_node_data { typedef FirstGraph<_node_data,EdgeTy,Directional,InOut,HasNoLockable,SortedNeighbors,FileEdgeTy> type; };

  template<typename _edge_data>
  struct with_edge_data { typedef FirstGraph<NodeTy,_edge_data,Directional,InOut,HasNoLockable,SortedNeighbors,FileEdgeTy> type; };

  template<typename _file_edge_data>
  struct with_file_edge_data { typedef FirstGraph<NodeTy,EdgeTy,Directional,InOut,HasNoLockable,SortedNeighbors,_file_edge_data> type; };

  template<bool _directional>
  struct with_directional { typedef FirstGraph<NodeTy,EdgeTy,_directional,InOut,HasNoLockable,SortedNeighbors,FileEdgeTy> type; };

  template<bool _sorted_neighbors>
  struct with_sorted_neighbors { typedef FirstGraph<NodeTy,EdgeTy,Directional,InOut,HasNoLockable,_sorted_neighbors,FileEdgeTy> type; };

  typedef read_with_aux_first_graph_tag read_tag;

private:
  template<typename T>
  struct first_eq_and_valid {
    T N2;
    first_eq_and_valid(T& n) :N2(n) {}
    template <typename T2>
    bool operator()(const T2& ii) const { 
      return ii.first() == N2 && ii.first() && ii.first()->active;
    }
  };

  struct first_not_valid {
    template <typename T2>
    bool operator()(const T2& ii) const { return !ii.first() || !ii.first()->active; }
  };

  template<typename T>
  struct first_lt {
    template <typename T2>
    bool operator()(const T& N2, const T2& ii) const {
      assert(ii.first() && "UNEXPECTED: invalid item in edgelist");
      return N2 < ii.first();
    }
    template <typename T2>
    bool operator()(const T2& ii, const T& N2) const {
      assert(ii.first() && "UNEXPECTED: invalid item in edgelist");
      return ii.first() < N2;
    }
  };
  
  class gNode;
  struct gNodeTypes: public internal::NodeInfoBaseTypes<NodeTy, !HasNoLockable> {
    //! The storage type for an edge
    typedef internal::UEdgeInfoBase<gNode, EdgeTy, Directional&!InOut> EdgeInfo;
    
    //! The storage type for edges
    typedef llvm::SmallVector<EdgeInfo, 3> EdgesTy;
    
    typedef typename EdgesTy::iterator iterator;
  };

  class gNode:
    public internal::NodeInfoBase<NodeTy, !HasNoLockable>,
    public gNodeTypes
  {
    friend class FirstGraph;
    typedef internal::NodeInfoBase<NodeTy, !HasNoLockable> NodeInfo;
    typename gNode::EdgesTy edges;
    typedef typename gNode::iterator iterator;
    typedef typename gNode::EdgeInfo EdgeInfo;

    bool active;
    
    iterator begin() { return edges.begin(); }
    iterator end() { return edges.end();  }
    
    void erase(iterator ii) {
      if (SortedNeighbors) {
        // For sorted case remove the element, moving following
        // elements back to fill the space.
        edges.erase(ii);
      } else {
        // We don't need to preserve the order, so move the last edge
        // into this place and then remove last edge.
        *ii = edges.back();
        edges.pop_back();
      }
    }

    void erase(gNode* N, bool inEdge = false) { 
      iterator ii = find(N, inEdge);
      if (ii != end())
        edges.erase(ii); 
    }

    iterator find(gNode* N, bool inEdge = false) {
      iterator ii, ei = edges.end();
      if (SortedNeighbors) {
        assert(std::is_sorted(edges.begin(), edges.end(), 
                              [=] (const EdgeInfo& e1, const EdgeInfo& e2) 
                                  { return e1.first() < e2.first(); }
                             )
              );
        ii = std::lower_bound(edges.begin(), edges.end(), N,
                              first_lt<gNode*>());
      } else {
        ii = edges.begin();
      }

      first_eq_and_valid<gNode*> checker(N);
      ii = std::find_if(ii, ei, checker);
      while(ii != ei && ii->isInEdge() != inEdge) {
        ++ii;
        ii = std::find_if(ii, ei, checker);
      };
      return ii;
    }

    void resizeEdges(size_t size) {
      edges.resize(size, EdgeInfo(new gNode(), 0));
    }

    template<typename... Args>
    iterator createEdge(gNode* N, EdgeTy* v, bool inEdge, Args&&... args) {
      iterator ii;
      if (SortedNeighbors) {
        // If neighbors are sorted, find appropriate insertion point.
        // Insert before first neighbor that is too far.
        ii = std::upper_bound(edges.begin(), edges.end(), N,
                              first_lt<gNode*>());
      }
      else
        ii = edges.end();
      return edges.insert(ii, EdgeInfo(N, v, inEdge, std::forward<Args>(args)...));
    }

    template<typename... Args>
    iterator createEdgeWithReuse(gNode* N, EdgeTy* v, bool inEdge, Args&&... args) {
      // First check for holes
      iterator ii, ei;
      if (SortedNeighbors) {
        // If neighbors are sorted, find acceptable range for insertion.
        ii = std::lower_bound(edges.begin(), edges.end(), N,
                              first_lt<gNode*>());
        ei = std::upper_bound(ii, edges.end(), N,
                              first_lt<gNode*>());
      }
      else {
        // If not sorted, we can insert anywhere in the list.
        ii = edges.begin();
        ei = edges.end();
      }
      ii = std::find_if(ii, ei, first_not_valid());
      if (ii != ei) {
        // FIXME: We could move elements around (short distances).
	*ii = EdgeInfo(N, v, inEdge, std::forward<Args>(args)...);
	return ii;
      }
      return edges.insert(ei, EdgeInfo(N, v, inEdge, std::forward<Args>(args)...));
    }

    template<bool _A1 = HasNoLockable>
    void acquire(MethodFlag mflag, typename std::enable_if<!_A1>::type* = 0) {
      galois::runtime::acquire(this, mflag);
    }

    template<bool _A1 = HasNoLockable>
    void acquire(MethodFlag mflag, typename std::enable_if<_A1>::type* = 0) { }

  public:
    template<typename... Args>
    gNode(Args&&... args): NodeInfo(std::forward<Args>(args)...), active(false) { }
  };

  //The graph manages the lifetimes of the data in the nodes and edges
  typedef galois::InsertBag<gNode> NodeListTy;
  NodeListTy nodes;

  internal::EdgeFactory<EdgeTy> edgesF;

  //Helpers for iterator classes
  struct is_node : public std::unary_function<gNode&, bool>{
    bool operator() (const gNode& g) const { return g.active; }
  };
  struct is_edge : public std::unary_function<typename gNodeTypes::EdgeInfo&, bool> {
    bool operator()(typename gNodeTypes::EdgeInfo& e) const { return e.first()->active; }
  };
  struct is_in_edge : public std::unary_function<typename gNodeTypes::EdgeInfo&, bool> {
    bool operator()(typename gNodeTypes::EdgeInfo& e) const { return e.first()->active && e.isInEdge(); }
  };
  struct is_out_edge : public std::unary_function<typename gNodeTypes::EdgeInfo&, bool> {
    bool operator()(typename gNodeTypes::EdgeInfo& e) const { return e.first()->active && !e.isInEdge(); }
  };
  struct makeGraphNode: public std::unary_function<gNode&, gNode*> {
    gNode* operator()(gNode& data) const { return &data; }
  };

public:
  //! Graph node handle
  typedef gNode* GraphNode;
  //! Edge data type
  typedef EdgeTy edge_data_type;
  //! Edge data type of file we are loading this graph from
  typedef FileEdgeTy file_edge_data_type;
  //! Node data type
  typedef NodeTy node_data_type;
  //! (Out or Undirected) Edge iterator
  typedef typename boost::filter_iterator<is_out_edge, typename gNodeTypes::iterator> edge_iterator;
  //! In Edge iterator
  typedef typename boost::filter_iterator<is_in_edge, typename gNodeTypes::iterator> in_edge_iterator;
  //! Reference to edge data
  typedef typename gNodeTypes::EdgeInfo::reference edge_data_reference;
  //! Reference to node data
  typedef typename gNodeTypes::reference node_data_reference;
  //! Node iterator
  typedef boost::transform_iterator<makeGraphNode,
          boost::filter_iterator<is_node,
                   typename NodeListTy::iterator> > iterator;
#ifdef AUX_MAP
  struct ReadGraphAuxData {
    LargeArray<GraphNode> nodes;
    galois::PerThreadMap<FileGraph::GraphNode, galois::gstl::Vector<GraphNode> > inNghs;
  };
#else
  struct AuxNode {
    galois::substrate::SimpleLock lock;
    GraphNode n;
    galois::gstl::Vector<GraphNode> inNghs;
  };
  using AuxNodePadded = typename galois::substrate::CacheLineStorage<AuxNode>;
  using ReadGraphAuxData = LargeArray<AuxNodePadded> ReadGraphAuxData;
#endif

private:
  template<typename... Args>
  edge_iterator createEdgeWithReuse(GraphNode src, GraphNode dst, galois::MethodFlag mflag, Args&&... args) {
    assert(src);
    assert(dst);
    // galois::runtime::checkWrite(mflag, true);
    src->acquire(mflag);
    typename gNode::iterator ii = src->find(dst);
    if (ii == src->end()) {
      if (Directional && !InOut) {
	ii = src->createEdgeWithReuse(dst, 0, false, std::forward<Args>(args)...);
      } else {
        dst->acquire(mflag);
	EdgeTy* e = edgesF.mkEdge(std::forward<Args>(args)...);
	ii = dst->createEdgeWithReuse(src, e, Directional ? true : false,  std::forward<Args>(args)...);
	ii = src->createEdgeWithReuse(dst, e, false, std::forward<Args>(args)...);
      }
    }
    return boost::make_filter_iterator(is_out_edge(), ii, src->end());
  }

  template<typename... Args>
  edge_iterator createEdge(GraphNode src, GraphNode dst, galois::MethodFlag mflag, Args&&... args) {
    assert(src);
    assert(dst);
    // galois::runtime::checkWrite(mflag, true);
    src->acquire(mflag);
    typename gNode::iterator ii = src->end();
    if (ii == src->end()) {
      if (Directional && !InOut) {
	ii = src->createEdge(dst, 0, false, std::forward<Args>(args)...);
      } else {
        dst->acquire(mflag);
	EdgeTy* e = edgesF.mkEdge(std::forward<Args>(args)...);
	ii = dst->createEdge(src, e, Directional ? true : false, std::forward<Args>(args)...);
	ii = src->createEdge(dst, e, false, std::forward<Args>(args)...);
      }
    }
    return boost::make_filter_iterator(is_out_edge(), ii, src->end());
  }

  /**
   * Creates an outgoing edge at src for the edge from src to dst.
   * Only called by constructOutEdgeValue.
   */
  template<typename... Args>
  edge_iterator createOutEdge(GraphNode src, GraphNode dst, galois::MethodFlag mflag, Args&&... args) {
    assert(src);
    assert(dst);

    src->acquire(mflag);
    typename gNode::iterator ii = src->end();
    if (ii == src->end()) {
      dst->acquire(mflag);
      EdgeTy* e = edgesF.mkEdge(std::forward<Args>(args)...);
      ii = src->createEdge(dst, e, false, std::forward<Args>(args)...);
    }
    return boost::make_filter_iterator(is_out_edge(), ii, src->end());
  }

  /**
   * Creates an incoming edge at dst for the edge from src to dst.
   * Only called by constructInEdgeValue.
   * Reuse data from the corresponding outgoing edge.
   */
  template<typename... Args>
  in_edge_iterator createInEdge(GraphNode src, GraphNode dst, galois::MethodFlag mflag, Args&&... args) {
    assert(src);
    assert(dst);

    dst->acquire(mflag);
    typename gNode::iterator ii = dst->end();
    if (ii == dst->end()) {
      src->acquire(mflag);
      EdgeTy* e = src->find(dst)->second();
      ii = dst->createEdge(src, e, Directional ? true : false, std::forward<Args>(args)...);
    }
    return boost::make_filter_iterator(is_in_edge(), ii, dst->end());
  }

  template<bool _A1 = LargeArray<EdgeTy>::has_value, bool _A2 = LargeArray<FileEdgeTy>::has_value>
  void constructOutEdgeValue(FileGraph& graph, typename FileGraph::edge_iterator nn,
      GraphNode src, GraphNode dst, typename std::enable_if<!_A1 || _A2>::type* = 0) {
    typedef typename LargeArray<FileEdgeTy>::value_type FEDV;
    typedef LargeArray<EdgeTy> ED;
    if (ED::has_value) {
      createOutEdge(src, dst, galois::MethodFlag::UNPROTECTED, graph.getEdgeData<FEDV>(nn));
    } else {
      createOutEdge(src, dst, galois::MethodFlag::UNPROTECTED);
    }
  }

  template<bool _A1 = LargeArray<EdgeTy>::has_value, bool _A2 = LargeArray<FileEdgeTy>::has_value>
  void constructOutEdgeValue(FileGraph& graph, typename FileGraph::edge_iterator nn,
      GraphNode src, GraphNode dst, typename std::enable_if<_A1 && !_A2>::type* = 0) {
    createOutEdge(src, dst, galois::MethodFlag::UNPROTECTED);
  }

  // will reuse edge data from outgoing edges
  void constructInEdgeValue(FileGraph& graph, GraphNode src, GraphNode dst) {
    createInEdge(src, dst, galois::MethodFlag::UNPROTECTED);
  }

public:
  /**
   * Creates a new node holding the indicated data. Usually you should call
   * {@link addNode()} afterwards.
   */
  template<typename... Args>
  GraphNode createNode(Args&&... args) {
    gNode* N = &(nodes.emplace(std::forward<Args>(args)...));
    N->active = false;
    return GraphNode(N);
  }

  /**
   * Adds a node to the graph.
   */
  void addNode(const GraphNode& n, galois::MethodFlag mflag = MethodFlag::WRITE) {
    // galois::runtime::checkWrite(mflag, true);
    n->acquire(mflag);
    n->active = true;
  }

  //! Gets the node data for a node.
  node_data_reference getData(const GraphNode& n, galois::MethodFlag mflag = MethodFlag::WRITE) const {
    assert(n);
    // galois::runtime::checkWrite(mflag, false);
    n->acquire(mflag);
    return n->getData();
  }

  //! Checks if a node is in the graph
  bool containsNode(const GraphNode& n, galois::MethodFlag mflag = MethodFlag::WRITE) const {
    assert(n);
    n->acquire(mflag);
    return n->active;
  }

  /**
   * Removes a node from the graph along with all its outgoing/incoming edges
   * for undirected graphs or outgoing edges for directed graphs.
   */
  //FIXME: handle edge memory
  void removeNode(GraphNode n, galois::MethodFlag mflag = MethodFlag::WRITE) {
    assert(n);
    // galois::runtime::checkWrite(mflag, true);
    n->acquire(mflag);
    gNode* N = n;
    if (N->active) {
      N->active = false;
      if ((!Directional || InOut) && edgesF.mustDel()) {
	for (auto ii : edges(n, MethodFlag::UNPROTECTED))
	  edgesF.delEdge(ii->second());
        for (auto ii : in_edges(n, MethodFlag::UNPROTECTED))
          edgesF.delEdge(ii->second());
      }
      N->edges.clear();
    }
  }

  /**
   * Resize the edges of the node. For best performance, should be done serially.
   */
  void resizeEdges(GraphNode src, size_t size, galois::MethodFlag mflag = MethodFlag::WRITE) {
    assert(src);
    // galois::runtime::checkWrite(mflag, false);
    src->acquire(mflag);
    src->resizeEdges(size);
   }

  /** 
   * Adds an edge to graph, replacing existing value if edge already exists. 
   *
   * Ignore the edge data, let the caller use the returned iterator to set the
   * value if desired.  This frees us from dealing with the void edge data
   * problem in this API
   */
  edge_iterator addEdge(GraphNode src, GraphNode dst, galois::MethodFlag mflag = MethodFlag::WRITE) {
    return createEdgeWithReuse(src, dst, mflag);
  }

  //! Adds and initializes an edge to graph but does not check for duplicate edges
  template<typename... Args>
  edge_iterator addMultiEdge(GraphNode src, GraphNode dst, galois::MethodFlag mflag, Args&&... args) {
    return createEdge(src, dst, mflag, std::forward<Args>(args)...);
  }

  //! Removes an edge from the graph
  void removeEdge(GraphNode src, edge_iterator dst, galois::MethodFlag mflag = MethodFlag::WRITE) {
    assert(src);
    // galois::runtime::checkWrite(mflag, true);
    src->acquire(mflag);
    if (Directional && !InOut) {
      src->erase(dst.base());
    } else {
      dst->first()->acquire(mflag);
      EdgeTy* e = dst->second();
      edgesF.delEdge(e);
      dst->first()->erase(src, Directional ? true : false); // erase incoming/symmetric edge
      src->erase(dst.base());
    }
  }

  //! Finds if an edge between src and dst exists
  edge_iterator findEdge(GraphNode src, GraphNode dst, galois::MethodFlag mflag = MethodFlag::WRITE) {
    assert(src);
    assert(dst);
    src->acquire(mflag);
    typename gNodeTypes::iterator ii = src->find(dst), ei = src->end();
    is_out_edge edge_predicate;
    if ( ii != ei && edge_predicate(*ii) ) {
      // After finding edge, lock dst and verify still active
      dst->acquire(mflag);
      if ( !edge_predicate(*ii) )
        // I think we need this too, else we'll return some random iterator.
        ii = ei;
    }
    else
      ii = ei;
    return boost::make_filter_iterator(edge_predicate, ii, ei);
  }

  edge_iterator findEdgeSortedByDst(GraphNode src, GraphNode dst, galois::MethodFlag mflag = MethodFlag::WRITE) {
    assert(src);
    assert(dst);
    src->acquire(mflag);
    assert(std::is_sorted(src->begin(), src->end(), 
                          [=] (const typename gNode::EdgeInfo& e1, const typename gNode::EdgeInfo& e2) 
                              { return e1.first() < e2.first(); }
                         )
          );

    auto ei = src->end();
    auto ii = std::lower_bound(src->begin(), src->end(), dst,
                               first_lt<gNode*>());

    first_eq_and_valid<gNode*> checker(dst);
    ii = std::find_if(ii, ei, checker); // bug if ei set to upper_bound
    while(ii != ei && ii->isInEdge()) {
      ++ii;
      ii = std::find_if(ii, ei, checker);
    };

    is_out_edge edge_predicate;
    if(ii != ei) {
      dst->acquire(mflag);
      if(!edge_predicate(*ii)) {
        ii = ei;
      }
    }
    return boost::make_filter_iterator(edge_predicate, ii, ei);
  }

  template<bool _Undirected = !Directional>
  edge_iterator findInEdge(GraphNode src, GraphNode dst, galois::MethodFlag mflag = MethodFlag::WRITE, 
                           typename std::enable_if<_Undirected>::type* = 0)
  {
    // incoming neighbors are the same as outgoing neighbors in undirected graphs
    return findEdge(src, dst, mflag);
  }

  // Find if an incoming edge between src and dst exists for directed in-out graphs
  template<bool _DirectedInOut = (Directional && InOut)>
  in_edge_iterator findInEdge(GraphNode src, GraphNode dst, galois::MethodFlag mflag = MethodFlag::WRITE, 
                              typename std::enable_if<_DirectedInOut>::type* = 0) 
  {
    assert(src);
    assert(dst);
    src->acquire(mflag);
    typename gNodeTypes::iterator ii = src->find(dst, true), ei = src->end();
    is_in_edge edge_predicate;
    if ( ii != ei && edge_predicate(*ii) ) {
      // After finding edges, lock dst and verify still active
      dst->acquire(mflag);
      if ( !edge_predicate(*ii) )
        // need this to avoid returning a random iterator
        ii = ei;
    }
    else
      ii = ei;
    return boost::make_filter_iterator(edge_predicate, ii, ei);
  }

  /**
   * Returns the edge data associated with the edge. It is an error to
   * get the edge data for a non-existent edge.  It is an error to get
   * edge data for inactive edges. By default, the mflag is galois::MethodFlag::UNPROTECTED
   * because edge_begin() dominates this call and should perform the
   * appropriate locking.
   */
  edge_data_reference getEdgeData(edge_iterator ii, galois::MethodFlag mflag = MethodFlag::UNPROTECTED) const {
    assert(ii->first()->active);
    // galois::runtime::checkWrite(mflag, false);
    ii->first()->acquire(mflag);
    return *ii->second();
  }

  edge_data_reference getEdgeData(in_edge_iterator ii, galois::MethodFlag mflag = MethodFlag::UNPROTECTED) const {
    assert(ii->first()->active);
    // galois::runtime::checkWrite(mflag, false);
    ii->first()->acquire(mflag);
    return *ii->second();
  }

  //! Returns the destination of an edge
  GraphNode getEdgeDst(edge_iterator ii) {
    assert(ii->first()->active);
    return GraphNode(ii->first());
  }

  GraphNode getEdgeDst(in_edge_iterator ii) {
    assert(ii->first()->active);
    return GraphNode(ii->first());
  }

  void sortEdgesByDst(GraphNode N, galois::MethodFlag mflag = MethodFlag::WRITE) { 
    acquire(N, mflag);
    typedef typename gNode::EdgeInfo EdgeInfo;
    std::sort(N->begin(), N->end(), [=] (const EdgeInfo& e1, const EdgeInfo& e2) { return e1.first() < e2.first(); } );
  }

  void sortAllEdgesByDst(MethodFlag mflag = MethodFlag::WRITE) {
    galois::do_all(galois::iterate(*this), [=] (GraphNode N) {this->sortEdgesByDst(N, mflag);}, galois::steal<true>());
  }

  //// General Things ////

  //! Returns an iterator to the neighbors of a node 
  edge_iterator edge_begin(GraphNode N, galois::MethodFlag mflag = MethodFlag::WRITE) {
    assert(N);
    N->acquire(mflag);

    if (galois::runtime::shouldLock(mflag)) {
      for (typename gNode::iterator ii = N->begin(), ee = N->end(); ii != ee; ++ii) {
	if (ii->first()->active && !ii->isInEdge())
	  ii->first()->acquire(mflag);
      }
    }
    return boost::make_filter_iterator(is_out_edge(), N->begin(), N->end());
  }

  template<bool _Undirected = !Directional>
  in_edge_iterator in_edge_begin(GraphNode N, galois::MethodFlag mflag = MethodFlag::WRITE, 
                                 typename std::enable_if<!_Undirected>::type* = 0) 
  {
    assert(N);
    N->acquire(mflag);

    if (galois::runtime::shouldLock(mflag)) {
      for (typename gNode::iterator ii = N->begin(), ee = N->end(); ii != ee; ++ii) {
	if (ii->first()->active && ii->isInEdge())
	  ii->first()->acquire(mflag);
      }
    }
    return boost::make_filter_iterator(is_in_edge(), N->begin(), N->end());
  }

  template<bool _Undirected = !Directional>
  edge_iterator in_edge_begin(GraphNode N, galois::MethodFlag mflag = MethodFlag::WRITE, 
                              typename std::enable_if<_Undirected>::type* = 0) 
  {
    return edge_begin(N, mflag);
  }

  //! Returns the end of the neighbor iterator 
  edge_iterator edge_end(GraphNode N, galois::MethodFlag mflag = MethodFlag::WRITE) {
    assert(N);
    // Acquiring lock is not necessary: no valid use for an end pointer should
    // ever require it
    // N->acquire(mflag);
    return boost::make_filter_iterator(is_out_edge(), N->end(), N->end());
  }

  template<bool _Undirected = !Directional>
  in_edge_iterator in_edge_end(GraphNode N, galois::MethodFlag mflag = MethodFlag::WRITE,
                               typename std::enable_if<!_Undirected>::type* = 0)
  {
    assert(N);
    // Acquiring lock is not necessary: no valid use for an end pointer should
    // ever require it
    // N->acquire(mflag);
    return boost::make_filter_iterator(is_in_edge(), N->end(), N->end());
  }

  template<bool _Undirected = !Directional>
  edge_iterator in_edge_end(GraphNode N, galois::MethodFlag mflag = MethodFlag::WRITE,
                            typename std::enable_if<_Undirected>::type* = 0)
  {
    return edge_end(N, mflag);
  } 

  runtime::iterable<NoDerefIterator<edge_iterator>> edges(GraphNode N, galois::MethodFlag mflag = MethodFlag::WRITE) {
    return internal::make_no_deref_range(edge_begin(N, mflag), edge_end(N, mflag));
  }

  template<bool _Undirected = !Directional>
  runtime::iterable<NoDerefIterator<in_edge_iterator>> in_edges(GraphNode N, galois::MethodFlag mflag = MethodFlag::WRITE, 
                                                                typename std::enable_if<!_Undirected>::type* = 0) 
  {
    return internal::make_no_deref_range(in_edge_begin(N, mflag), in_edge_end(N, mflag));
  }

  template<bool _Undirected = !Directional>
  runtime::iterable<NoDerefIterator<edge_iterator>> in_edges(GraphNode N, galois::MethodFlag mflag = MethodFlag::WRITE, 
                                                                typename std::enable_if<_Undirected>::type* = 0) 
  {
    return edges(N, mflag);
  }

  /**
   * An object with begin() and end() methods to iterate over the outgoing
   * edges of N.
   */
  internal::EdgesIterator<FirstGraph> out_edges(GraphNode N, MethodFlag mflag = MethodFlag::WRITE) {
    return internal::EdgesIterator<FirstGraph>(*this, N, mflag);
  }

  /**
   * Returns an iterator to all the nodes in the graph. Not thread-safe.
   */
  iterator begin() {
    return boost::make_transform_iterator(
           boost::make_filter_iterator(is_node(),
				       nodes.begin(), nodes.end()),
	   makeGraphNode());
  }

  //! Returns the end of the node iterator. Not thread-safe.
  iterator end() {
    return boost::make_transform_iterator(
           boost::make_filter_iterator(is_node(),
				       nodes.end(), nodes.end()), 
	   makeGraphNode());
  }

  typedef iterator local_iterator;

  local_iterator local_begin() {
    return boost::make_transform_iterator(
           boost::make_filter_iterator(is_node(),
				       nodes.local_begin(), nodes.local_end()),
	   makeGraphNode());
  }

  local_iterator local_end() {
    return boost::make_transform_iterator(
           boost::make_filter_iterator(is_node(),
				       nodes.local_end(), nodes.local_end()), 
	   makeGraphNode());
  }

  /**
   * Returns the number of nodes in the graph. Not thread-safe.
   */
  unsigned int size() {
    return std::distance(begin(), end());
  }

  //! Returns the size of edge data.
  size_t sizeOfEdgeData() const {
    return gNode::EdgeInfo::sizeOfSecond();
  }

#ifdef AUX_MAP
  void allocateFrom(FileGraph& graph, ReadGraphAuxData& aux) { 
    size_t numNodes = graph.size();
    aux.nodes.allocateInterleaved(numNodes);
  }

  void constructNodesFrom(FileGraph& graph, unsigned tid, unsigned total, ReadGraphAuxData& aux) {
    auto r = graph.divideByNode(sizeof(gNode), sizeof(typename gNode::EdgeInfo), tid, total).first;
    for (FileGraph::iterator ii = r.first, ei = r.second; ii != ei; ++ii) {
      aux.nodes[*ii] = createNode();
      addNode(aux.nodes[*ii], galois::MethodFlag::UNPROTECTED);
    }
  }

  void constructOutEdgesFrom(FileGraph& graph, unsigned tid, unsigned total, ReadGraphAuxData& aux) {
    typedef typename std::decay<typename gNode::EdgeInfo::reference>::type value_type;
    auto r = graph.divideByNode(sizeof(gNode), sizeof(typename gNode::EdgeInfo), tid, total).first;
    auto& map = aux.inNghs.get();

    for (FileGraph::iterator ii = r.first, ei = r.second; ii != ei; ++ii) {
      for (FileGraph::edge_iterator nn = graph.edge_begin(*ii), en = graph.edge_end(*ii); nn != en; ++nn) {
        auto dstID = graph.getEdgeDst(nn);
        auto src = aux.nodes[*ii], dst = aux.nodes[dstID];
        constructOutEdgeValue(graph, nn, src, dst);
        if (!Directional || InOut) {
          map[dstID].push_back(src);
        }
      }
    }
  }

  void constructInEdgesFrom(FileGraph& graph, unsigned tid, unsigned total, const ReadGraphAuxData& aux) {
    typedef typename std::decay<typename gNode::EdgeInfo::reference>::type value_type;

    if (!Directional || InOut) {
      auto r = graph.divideByNode(sizeof(gNode), sizeof(typename gNode::EdgeInfo), tid, total).first;

      for (auto i = 0; i < aux.inNghs.numRows(); ++i) {
        auto& map = aux.inNghs.get(i);
        auto ii = map.lower_bound(*(r.first)), ei = map.upper_bound(*(r.second));
        for ( ; ii != ei; ++ii) {
          auto dst = aux.nodes[ii->first];
          for (auto src: ii->second) {
            constructInEdgeValue(graph, src, dst);
          }
        }
      }
    }
  }
#else
  void allocateFrom(FileGraph& graph, ReadGraphAuxData& aux) {
    size_t numNodes = graph.size();
    aux.allocateInterleaved(numNodes);

    galois::do_all(aux.begin(), aux.end(),
        [=] (AuxNodePadded& n) {
           // in-place new
           new (&n) AuxNodePadded();
        }
    );
  }

  void constructNodesFrom(FileGraph& graph, unsigned tid, unsigned total, ReadGraphAuxData& aux) {
    auto r = graph.divideByNode(sizeof(gNode), sizeof(typename gNode::EdgeInfo), tid, total).first;
    for (FileGraph::iterator ii = r.first, ei = r.second; ii != ei; ++ii) {
      auto& auxNode = aux[*ii].get();
      auxNode.n = createNode();
      addNode(auxNode.n, galois::MethodFlag::UNPROTECTED);
    }
  }

  void constructOutEdgesFrom(FileGraph& graph, unsigned tid, unsigned total, ReadGraphAuxData& aux) {
    typedef typename std::decay<typename gNode::EdgeInfo::reference>::type value_type;
    auto r = graph.divideByNode(sizeof(gNode), sizeof(typename gNode::EdgeInfo), tid, total).first;

    for (FileGraph::iterator ii = r.first, ei = r.second; ii != ei; ++ii) {
      for (FileGraph::edge_iterator nn = graph.edge_begin(*ii), en = graph.edge_end(*ii); nn != en; ++nn) {
        auto src = aux[*ii].get().n;
        auto& dstAux = aux[graph.getEdgeDst(nn)].get();
        constructOutEdgeValue(graph, nn, src, dstAux.n);
        if (!Directional || InOut) {
          dstAux.lock.lock();
          dstAux.inNghs.push_back(src);
          dstAux.lock.unlock();
        }
      }
    }
  }

  void constructInEdgesFrom(FileGraph& graph, unsigned tid, unsigned total, ReadGraphAuxData& aux) {
    typedef typename std::decay<typename gNode::EdgeInfo::reference>::type value_type;

    if (!Directional || InOut) {
      auto r = graph.divideByNode(sizeof(gNode), sizeof(typename gNode::EdgeInfo), tid, total).first;

      for (FileGraph::iterator ii = r.first, ei = r.second; ii != ei; ++ii) {
        auto& auxNode = aux[*ii].get();
        for (auto src: auxNode.inNghs) {
          constructInEdgeValue(graph, src, auxNode.n);
        }
      }
    }
  }
#endif
};

}
}
#endif
