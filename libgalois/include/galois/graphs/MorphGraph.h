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

/**
 * @file MorphGraph.h
 *
 * Contains MorphGraph and associated helpers.
 */

#ifndef GALOIS_GRAPHS_MORPHGRAPH_H
#define GALOIS_GRAPHS_MORPHGRAPH_H

#include <algorithm>
#include <map>
#include <set>
#include <type_traits>
#include <vector>

#include <boost/container/small_vector.hpp>
#include <boost/functional.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/filter_iterator.hpp>
#include <boost/container/small_vector.hpp>

#include "galois/Bag.h"
#include "galois/config.h"
#include "galois/Galois.h"
#include "galois/graphs/FileGraph.h"
#include "galois/graphs/Details.h"
#include "galois/gstl.h"

#ifdef AUX_MAP
#include "galois/PerThreadContainer.h"
#else
#include "galois/substrate/CacheLineStorage.h"
#include "galois/substrate/SimpleLock.h"
#endif

namespace galois {
namespace graphs {

namespace internal {
/**
 * Wrapper class to have a valid type on void edges
 */
template <typename NTy, typename ETy, bool DirectedButNotInOut>
struct UEdgeInfoBase;

template <typename NTy, typename ETy>
struct UEdgeInfoBase<NTy, ETy, true> {
  typedef ETy& reference;

  NTy* N;
  ETy Ea;

  inline NTy* first() {
    assert(N);
    return N;
  }
  inline const NTy* first() const {
    assert(N);
    return N;
  }
  inline ETy* second() { return &Ea; }
  inline const ETy* second() const { return &Ea; }

  template <typename... Args>
  UEdgeInfoBase(NTy* n, ETy*, bool, Args&&... args)
      : N(n), Ea(std::forward<Args>(args)...) {}

  template <typename... Args>
  UEdgeInfoBase(NTy* n, ETy& v, bool, Args&&...) : N(n) {
    Ea = v;
  }

  static size_t sizeOfSecond() { return sizeof(ETy); }
  bool isInEdge() const { return false; }
};

template <typename NTy, typename ETy>
struct UEdgeInfoBase<NTy, ETy, false> {
  typedef ETy& reference;

  NTy* N;
  ETy* Ea;

  inline NTy* first() {
    assert(N);
    return (NTy*)((uintptr_t)N & ~1);
  }
  inline const NTy* first() const {
    assert(N);
    return (NTy*)((uintptr_t)N & ~1);
  }
  inline ETy* second() { return Ea; }
  inline const ETy* second() const { return Ea; }
  template <typename... Args>
  UEdgeInfoBase(NTy* n, ETy* v, bool f, Args&&...)
      : N((NTy*)((uintptr_t)n | f)), Ea(v) {}
  static size_t sizeOfSecond() { return sizeof(ETy); }
  bool isInEdge() const { return (uintptr_t)N & 1; }
};

template <typename NTy>
struct UEdgeInfoBase<NTy, void, true> {
  typedef char& reference;

  NTy* N;
  inline NTy* first() { return N; }
  inline const NTy* first() const { return N; }
  inline char* second() const { return static_cast<char*>(NULL); }
  inline char* addr() const { return second(); }
  template <typename... Args>
  UEdgeInfoBase(NTy* n, void*, bool, Args&&...) : N(n) {}
  static size_t sizeOfSecond() { return 0; }
  bool isInEdge() const { return false; }
};

template <typename NTy>
struct UEdgeInfoBase<NTy, void, false> {
  typedef char& reference;

  NTy* N;
  inline NTy* first() { return (NTy*)((uintptr_t)N & ~1); }
  inline const NTy* first() const { return (NTy*)((uintptr_t)N & ~1); }
  inline char* second() const { return static_cast<char*>(NULL); }
  inline char* addr() const { return second(); }
  template <typename... Args>
  UEdgeInfoBase(NTy* n, void*, bool f, Args&&...)
      : N((NTy*)((uintptr_t)n | f)) {}
  static size_t sizeOfSecond() { return 0; }
  bool isInEdge() const { return (uintptr_t)N & 1; }
};

/*
 * Only graphs w/ in-out/symmetric edges and non-void edge data,
 * i.e. ETy != void and DirectedNotInOut = false,
 * need to allocate memory for edge data
 */
template <typename ETy, bool DirectedNotInOut>
struct EdgeFactory {
  galois::InsertBag<ETy> mem;
  template <typename... Args>
  ETy* mkEdge(Args&&... args) {
    return &mem.emplace(std::forward<Args>(args)...);
  }
  void delEdge(ETy*) {}
  bool mustDel() const { return false; }
};

template <typename ETy>
struct EdgeFactory<ETy, true> {
  template <typename... Args>
  ETy* mkEdge(Args&&...) {
    return nullptr;
  }
  void delEdge(ETy*) {}
  bool mustDel() const { return false; }
};

template <>
struct EdgeFactory<void, false> {
  template <typename... Args>
  void* mkEdge(Args&&...) {
    return static_cast<void*>(NULL);
  }
  void delEdge(void*) {}
  bool mustDel() const { return false; }
};

} // namespace internal

/**
 * A graph that can have new nodes and edges added to it.
 *
 * An example of use:
 *
 * \code
 * struct Node {
 *   ... // Definition of node data
 * };
 *
 * typedef galois::graphs::MorphGraph<Node,int,true> Graph;
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
 *   for (Graph::edge_iterator jj = g.edge_begin(src), ej = g.edge_end(src);
 *        jj != ej;
 *        ++jj) {
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
 * @tparam HasNoLockable if true, use no abstract locks in the graph
 * @tparam SortedNeighbors Keep neighbors sorted (for faster findEdge)
 * @tparam FileEdgeTy type of edges on file to be read from
 */
template <typename NodeTy, typename EdgeTy, bool Directional,
          bool InOut = false, bool HasNoLockable = false,
          bool SortedNeighbors = false, typename FileEdgeTy = EdgeTy>
class MorphGraph : private boost::noncopyable {
public:
  /**
   * Struct used to define the HasNoLockable template parameter as a type
   * in the struct.
   */
  template <bool _has_no_lockable>
  struct with_no_lockable {
    //! Type with Lockable parameter set according to struct template arg
    using type = MorphGraph<NodeTy, EdgeTy, Directional, InOut,
                            _has_no_lockable, SortedNeighbors, FileEdgeTy>;
  };

  /**
   * Struct used to define the type of node data in the graph.
   */
  template <typename _node_data>
  struct with_node_data {
    //! Type with node data parameter set according to struct template arg
    using type = MorphGraph<_node_data, EdgeTy, Directional, InOut,
                            HasNoLockable, SortedNeighbors, FileEdgeTy>;
  };

  /**
   * Struct used to define the type of edge data in the graph.
   */
  template <typename _edge_data>
  struct with_edge_data {
    //! Type with edge data parameter set according to struct template arg
    using type = MorphGraph<NodeTy, _edge_data, Directional, InOut,
                            HasNoLockable, SortedNeighbors, FileEdgeTy>;
  };

  /**
   * Struct used to define the type of file edge data in the graph.
   */
  template <typename _file_edge_data>
  struct with_file_edge_data {
    //! Type with file edge data parameter set according to struct template arg
    using type = MorphGraph<NodeTy, EdgeTy, Directional, InOut, HasNoLockable,
                            SortedNeighbors, _file_edge_data>;
  };

  /**
   * Struct used to define directionality of the graph.
   */
  template <bool _directional>
  struct with_directional {
    //! Type with directional parameter set according to struct template arg
    using type = MorphGraph<NodeTy, EdgeTy, _directional, InOut, HasNoLockable,
                            SortedNeighbors, FileEdgeTy>;
  };

  /**
   * Struct used to define if neighbors are sorted or not in the graph.
   */
  template <bool _sorted_neighbors>
  struct with_sorted_neighbors {
    //! Type with sort neighbor parameter set according to struct template arg
    using type = MorphGraph<NodeTy, EdgeTy, Directional, InOut, HasNoLockable,
                            _sorted_neighbors, FileEdgeTy>;
  };

  //! Tag that defines to graph reader how to read a graph into this class
  using read_tag = read_with_aux_first_graph_tag;

private: ///////////////////////////////////////////////////////////////////////
  template <typename T>
  struct first_eq_and_valid {
    T N2;
    first_eq_and_valid(T& n) : N2(n) {}
    template <typename T2>
    bool operator()(const T2& ii) const {
      return ii.first() == N2 && ii.first() && ii.first()->active;
    }
  };

  struct first_not_valid {
    template <typename T2>
    bool operator()(const T2& ii) const {
      return !ii.first() || !ii.first()->active;
    }
  };

  template <typename T>
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

  // forward declaration for graph node type
  class gNode;
  struct gNodeTypes
      : public internal::NodeInfoBaseTypes<NodeTy, !HasNoLockable> {
    //! The storage type for an edge
    using EdgeInfo =
        internal::UEdgeInfoBase<gNode, EdgeTy, Directional & !InOut>;

    //! The storage type for edges
    // typedef galois::gstl::Vector<EdgeInfo> EdgesTy;
    using EdgesTy = boost::container::small_vector<
        EdgeInfo, 3, galois::runtime::Pow_2_BlockAllocator<EdgeInfo>>;

    using iterator = typename EdgesTy::iterator;
  };

  class gNode : public internal::NodeInfoBase<NodeTy, !HasNoLockable>,
                public gNodeTypes {
    //! friend of MorphGraph since MorphGraph contains gNodes
    friend class MorphGraph;
    //! Storage type for node
    using NodeInfo = internal::NodeInfoBase<NodeTy, !HasNoLockable>;
    //! iterator over edges (taken from gNodeTypes)
    using iterator = typename gNode::iterator;
    //! Storage type of a single edge (taken from gNodeTypes)
    using EdgeInfo = typename gNode::EdgeInfo;

    //! edges on this node
    typename gNodeTypes::EdgesTy edges;

    //! Tracks if this node is considered as "in" the graph
    bool active;

    //! Return iterator to first edge
    iterator begin() { return edges.begin(); }
    //! Return iterator to end of edges
    iterator end() { return edges.end(); }

    //! Remove the provided edge from this node
    //! @param ii iterator to edge to remove
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

    /**
     * Erase an edge with a provided destination.
     */
    void erase(gNode* N, bool inEdge = false) {
      iterator ii = find(N, inEdge);
      if (ii != end())
        edges.erase(ii);
    }

    /**
     * Find an edge with a particular destination node.
     */
    iterator find(gNode* N, bool inEdge = false) {
      iterator ii, ei = edges.end();
      // find starting point to start search
      if (SortedNeighbors) {
        assert(std::is_sorted(edges.begin(), edges.end(),
                              [=](const EdgeInfo& e1, const EdgeInfo& e2) {
                                return e1.first() < e2.first();
                              }));
        ii =
            std::lower_bound(edges.begin(), edges.end(), N, first_lt<gNode*>());
      } else {
        ii = edges.begin();
      }

      first_eq_and_valid<gNode*> checker(N);
      ii = std::find_if(ii, ei, checker);
      while (ii != ei && ii->isInEdge() != inEdge) {
        ++ii;
        ii = std::find_if(ii, ei, checker);
      };
      return ii;
    }

    /**
     * Make space for more edges stored by this node
     */
    void resizeEdges(size_t size) {
      edges.resize(size, EdgeInfo(new gNode(), 0));
    }

    /**
     * Add a new edge to this node
     */
    template <typename... Args>
    iterator createEdge(gNode* N, EdgeTy* v, bool inEdge, Args&&... args) {
      iterator ii;
      if (SortedNeighbors) {
        // If neighbors are sorted, find appropriate insertion point.
        // Insert before first neighbor that is too far.
        ii =
            std::upper_bound(edges.begin(), edges.end(), N, first_lt<gNode*>());
      } else {
        ii = edges.end();
      }

      return edges.insert(ii,
                          EdgeInfo(N, v, inEdge, std::forward<Args>(args)...));
    }

    /**
     * Add an edge to this node; if space exists to add it in, then reuse that
     * space.
     */
    template <typename... Args>
    iterator createEdgeWithReuse(gNode* N, EdgeTy* v, bool inEdge,
                                 Args&&... args) {
      // First check for holes
      iterator ii, ei;
      if (SortedNeighbors) {
        // If neighbors are sorted, find acceptable range for insertion.
        ii =
            std::lower_bound(edges.begin(), edges.end(), N, first_lt<gNode*>());
        ei = std::upper_bound(ii, edges.end(), N, first_lt<gNode*>());
      } else {
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
      return edges.insert(ei,
                          EdgeInfo(N, v, inEdge, std::forward<Args>(args)...));
    }

    template <bool _A1 = HasNoLockable>
    void acquire(MethodFlag mflag, typename std::enable_if<!_A1>::type* = 0) {
      galois::runtime::acquire(this, mflag);
    }

    template <bool _A1 = HasNoLockable>
    void acquire(MethodFlag, typename std::enable_if<_A1>::type* = 0) {}

  public:
    template <typename... Args>
    gNode(Args&&... args)
        : NodeInfo(std::forward<Args>(args)...), active(false) {}
  };

  // The graph manages the lifetimes of the data in the nodes and edges
  //! Container for nodes
  using NodeListTy = galois::InsertBag<gNode>;
  //! nodes in this graph
  NodeListTy nodes;

  internal::EdgeFactory<EdgeTy, Directional && !InOut> edgesF;

  // Helpers for iterator classes
  struct is_node {
    bool operator()(const gNode& g) const { return g.active; }
  };
  struct is_edge {
    bool operator()(typename gNodeTypes::EdgeInfo& e) const {
      return e.first()->active;
    }
  };
  struct is_in_edge {
    bool operator()(typename gNodeTypes::EdgeInfo& e) const {
      return e.first()->active && e.isInEdge();
    }
  };
  struct is_out_edge {
    bool operator()(typename gNodeTypes::EdgeInfo& e) const {
      return e.first()->active && !e.isInEdge();
    }
  };
  struct makeGraphNode {
    gNode* operator()(gNode& data) const { return &data; }
  };

public: ////////////////////////////////////////////////////////////////////////
  //! Graph node handle
  using GraphNode = gNode*;
  //! Edge data type
  using edge_data_type = EdgeTy;
  //! Edge data type of file we are loading this graph from
  using file_edge_data_type = FileEdgeTy;
  //! Node data type
  using node_data_type = NodeTy;
  //! (Out or Undirected) Edge iterator
  using edge_iterator =
      typename boost::filter_iterator<is_out_edge,
                                      typename gNodeTypes::iterator>;
  //! In Edge iterator
  using in_edge_iterator =
      typename boost::filter_iterator<is_in_edge,
                                      typename gNodeTypes::iterator>;

  //! Reference to edge data
  using edge_data_reference = typename gNodeTypes::EdgeInfo::reference;
  //! Reference to node data
  using node_data_reference = typename gNodeTypes::reference;
  //! Node iterator
  using iterator = boost::transform_iterator<
      makeGraphNode,
      boost::filter_iterator<is_node, typename NodeListTy::iterator>>;

#ifdef AUX_MAP
  //! Auxiliary data for nodes that stores in neighbors in per thread storage
  //! accessed through a map
  struct ReadGraphAuxData {
    LargeArray<GraphNode> nodes;
    //! stores in neighbors
    galois::PerThreadMap<FileGraph::GraphNode,
                         galois::gstl::Vector<std::pair<GraphNode, EdgeTy*>>>
        inNghs;
  };
#else
  //! Wrapper around a graph node that provides a lock for it as well as
  //! in-neighbor tracking
  struct AuxNode {
    //! lock for wrapped graph node
    galois::substrate::SimpleLock lock;
    //! single graph node wrapped by this struct
    GraphNode n;
    //! stores in neighbors
    galois::gstl::Vector<std::pair<GraphNode, EdgeTy*>> inNghs;
  };
  //! Padded version of AuxNode
  using AuxNodePadded = typename galois::substrate::CacheLineStorage<AuxNode>;

  //! True if a node is both directional and not storing both in and out
  //! edges
  constexpr static const bool DirectedNotInOut = (Directional && !InOut);
  //! Large array that contains auxiliary data for each node (AuxNodes)
  using ReadGraphAuxData =
      typename std::conditional<DirectedNotInOut, LargeArray<GraphNode>,
                                LargeArray<AuxNodePadded>>::type;
#endif

private: ///////////////////////////////////////////////////////////////////////
  template <typename... Args>
  edge_iterator createEdgeWithReuse(GraphNode src, GraphNode dst,
                                    galois::MethodFlag mflag, Args&&... args) {
    assert(src);
    assert(dst);
    // galois::runtime::checkWrite(mflag, true);
    src->acquire(mflag);
    typename gNode::iterator ii = src->find(dst);
    // add edge only if it doesn't already exist
    if (ii == src->end()) {
      if (Directional && !InOut) {
        ii = src->createEdgeWithReuse(dst, 0, false,
                                      std::forward<Args>(args)...);
      } else {
        dst->acquire(mflag);
        EdgeTy* e = edgesF.mkEdge(std::forward<Args>(args)...);
        ii        = dst->createEdgeWithReuse(src, e, Directional ? true : false,
                                             std::forward<Args>(args)...);
        ii        = src->createEdgeWithReuse(dst, e, false,
                                             std::forward<Args>(args)...);
      }
    }
    return boost::make_filter_iterator(is_out_edge(), ii, src->end());
  }

  template <typename... Args>
  edge_iterator createEdge(GraphNode src, GraphNode dst,
                           galois::MethodFlag mflag, Args&&... args) {
    assert(src);
    assert(dst);
    // galois::runtime::checkWrite(mflag, true);
    src->acquire(mflag);
    typename gNode::iterator ii = src->end();
    // add edge only if it doesn't already exist
    if (ii == src->end()) {
      if (Directional && !InOut) {
        ii = src->createEdge(dst, 0, false, std::forward<Args>(args)...);
      } else {
        dst->acquire(mflag);
        EdgeTy* e = edgesF.mkEdge(std::forward<Args>(args)...);
        ii        = dst->createEdge(src, e, Directional ? true : false,
                                    std::forward<Args>(args)...);
        ii        = src->createEdge(dst, e, false, std::forward<Args>(args)...);
      }
    }
    return boost::make_filter_iterator(is_out_edge(), ii, src->end());
  }

  /**
   * Creates an outgoing edge at src for the edge from src to dst.
   * Only called by constructOutEdgeValue.
   */
  template <typename... Args>
  EdgeTy* createOutEdge(GraphNode src, GraphNode dst, galois::MethodFlag mflag,
                        Args&&... args) {
    assert(src);
    assert(dst);

    src->acquire(mflag);
    typename gNode::iterator ii = src->end();
    if (ii == src->end()) {
      dst->acquire(mflag);
      EdgeTy* e = edgesF.mkEdge(std::forward<Args>(args)...);
      ii        = src->createEdge(dst, e, false, std::forward<Args>(args)...);
      return e;
    }
    return nullptr;
  }

  /**
   * Creates an incoming edge at dst for the edge from src to dst.
   * Only called by constructInEdgeValue.
   * Reuse data from the corresponding outgoing edge.
   */
  template <typename... Args>
  void createInEdge(GraphNode src, GraphNode dst, EdgeTy* e,
                    galois::MethodFlag mflag, Args&&... args) {
    assert(src);
    assert(dst);

    dst->acquire(mflag);
    typename gNode::iterator ii = dst->end();
    if (ii == dst->end()) {
      src->acquire(mflag);
      ii = dst->createEdge(src, e, Directional ? true : false,
                           std::forward<Args>(args)...);
    }
  }

  template <bool _A1 = LargeArray<EdgeTy>::has_value,
            bool _A2 = LargeArray<FileEdgeTy>::has_value>
  EdgeTy*
  constructOutEdgeValue(FileGraph& graph, typename FileGraph::edge_iterator nn,
                        GraphNode src, GraphNode dst,
                        typename std::enable_if<!_A1 || _A2>::type* = 0) {
    typedef typename LargeArray<FileEdgeTy>::value_type FEDV;
    typedef LargeArray<EdgeTy> ED;
    if (ED::has_value) {
      return createOutEdge(src, dst, galois::MethodFlag::UNPROTECTED,
                           graph.getEdgeData<FEDV>(nn));
    } else {
      return createOutEdge(src, dst, galois::MethodFlag::UNPROTECTED);
    }
  }

  template <bool _A1 = LargeArray<EdgeTy>::has_value,
            bool _A2 = LargeArray<FileEdgeTy>::has_value>
  EdgeTy*
  constructOutEdgeValue(FileGraph&, typename FileGraph::edge_iterator,
                        GraphNode src, GraphNode dst,
                        typename std::enable_if<_A1 && !_A2>::type* = 0) {
    return createOutEdge(src, dst, galois::MethodFlag::UNPROTECTED);
  }

  // will reuse edge data from outgoing edges
  void constructInEdgeValue(FileGraph&, EdgeTy* e, GraphNode src,
                            GraphNode dst) {
    createInEdge(src, dst, e, galois::MethodFlag::UNPROTECTED);
  }

public
    : /////////////////////////////////////////////////////////////////////////
  /**
   * Creates a new node holding the indicated data. Usually you should call
   * {@link addNode()} afterwards.
   *
   * @param[in] args constructor arguments for node data
   * @returns newly created graph node
   */
  template <typename... Args>
  GraphNode createNode(Args&&... args) {
    gNode* N  = &(nodes.emplace(std::forward<Args>(args)...));
    N->active = false;
    return GraphNode(N);
  }

  /**
   * Adds a node to the graph.
   */
  void addNode(const GraphNode& n,
               galois::MethodFlag mflag = MethodFlag::WRITE) {
    // galois::runtime::checkWrite(mflag, true);
    n->acquire(mflag);
    n->active = true;
  }

  //! Gets the node data for a node.
  node_data_reference
  getData(const GraphNode& n,
          galois::MethodFlag mflag = MethodFlag::WRITE) const {
    assert(n);
    // galois::runtime::checkWrite(mflag, false);
    n->acquire(mflag);
    return n->getData();
  }

  GraphNode& getNode(uint64_t n) { return std::advance(this->begin(), n); }

  //! Checks if a node is in the graph
  //! @returns true if a node has is in the graph
  bool containsNode(const GraphNode& n,
                    galois::MethodFlag mflag = MethodFlag::WRITE) const {
    assert(n);
    n->acquire(mflag);
    return n->active;
  }

  /**
   * Removes a node from the graph along with all its outgoing/incoming edges
   * for undirected graphs or outgoing edges for directed graphs.
   *
   * @todo handle edge memory
   */
  void removeNode(GraphNode n, galois::MethodFlag mflag = MethodFlag::WRITE) {
    assert(n);
    // galois::runtime::checkWrite(mflag, true);
    n->acquire(mflag);
    gNode* N = n;
    if (N->active) {
      N->active = false;
      N->edges.clear();
    }
  }

  /**
   * Resize the edges of the node. For best performance, should be done
   * serially.
   */
  void resizeEdges(GraphNode src, size_t size,
                   galois::MethodFlag mflag = MethodFlag::WRITE) {
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
  edge_iterator addEdge(uint64_t src, uint64_t dst,
                        galois::MethodFlag mflag = MethodFlag::WRITE) {
    auto s = this->begin();
    std::advance(s, src);
    auto d = this->begin();
    std::advance(d, dst);
    return createEdgeWithReuse(*s, *d, mflag);
  }

  /**
   * Adds an edge to graph, replacing existing value if edge already exists.
   *
   * Ignore the edge data, let the caller use the returned iterator to set the
   * value if desired.  This frees us from dealing with the void edge data
   * problem in this API
   */
  edge_iterator addEdge(GraphNode src, GraphNode dst,
                        galois::MethodFlag mflag = MethodFlag::WRITE) {
    return createEdgeWithReuse(src, dst, mflag);
  }

  //! Adds and initializes an edge to graph but does not check for duplicate
  //! edges
  template <typename... Args>
  edge_iterator addMultiEdge(GraphNode src, GraphNode dst,
                             galois::MethodFlag mflag, Args&&... args) {
    return createEdge(src, dst, mflag, std::forward<Args>(args)...);
  }

  //! Removes an edge from the graph
  void removeEdge(GraphNode src, edge_iterator dst,
                  galois::MethodFlag mflag = MethodFlag::WRITE) {
    assert(src);
    // galois::runtime::checkWrite(mflag, true);
    src->acquire(mflag);
    if (Directional && !InOut) {
      src->erase(dst.base());
    } else {
      dst->first()->acquire(mflag);
      // EdgeTy* e = dst->second();
      dst->first()->erase(
          src, Directional ? true : false); // erase incoming/symmetric edge
      src->erase(dst.base());
    }
  }

  //! Finds if an edge between src and dst exists
  edge_iterator findEdge(GraphNode src, GraphNode dst,
                         galois::MethodFlag mflag = MethodFlag::WRITE) {
    assert(src);
    assert(dst);
    src->acquire(mflag);
    typename gNodeTypes::iterator ii = src->find(dst), ei = src->end();
    is_out_edge edge_predicate;
    if (ii != ei && edge_predicate(*ii)) {
      // After finding edge, lock dst and verify still active
      dst->acquire(mflag);
      if (!edge_predicate(*ii))
        // I think we need this too, else we'll return some random iterator.
        ii = ei;
    } else {
      ii = ei;
    }
    return boost::make_filter_iterator(edge_predicate, ii, ei);
  }

  //! Find/return edge between src/dst if it exists; assumes that edges
  //! are sorted by destination
  edge_iterator
  findEdgeSortedByDst(GraphNode src, GraphNode dst,
                      galois::MethodFlag mflag = MethodFlag::WRITE) {
    assert(src);
    assert(dst);
    src->acquire(mflag);
    assert(std::is_sorted(src->begin(), src->end(),
                          [=](const typename gNode::EdgeInfo& e1,
                              const typename gNode::EdgeInfo& e2) {
                            return e1.first() < e2.first();
                          }));

    auto ei = src->end();

    // jump directly to edges with destination we are looking for
    auto ii =
        std::lower_bound(src->begin(), src->end(), dst, first_lt<gNode*>());

    first_eq_and_valid<gNode*> checker(dst);
    ii = std::find_if(ii, ei, checker); // bug if ei set to upper_bound
    // ignore in edges
    while (ii != ei && ii->isInEdge()) {
      ++ii;
      ii = std::find_if(ii, ei, checker);
    };

    // make sure destination node is active else return end iterator
    is_out_edge edge_predicate;
    if (ii != ei) {
      dst->acquire(mflag);
      if (!edge_predicate(*ii)) {
        ii = ei;
      }
    }
    return boost::make_filter_iterator(edge_predicate, ii, ei);
  }

  //! Find a particular in-edge: note this function activates for the undirected
  //! graph case, so it just calls the regular out-edge finding function
  template <bool _Undirected = !Directional>
  edge_iterator findInEdge(GraphNode src, GraphNode dst,
                           galois::MethodFlag mflag = MethodFlag::WRITE,
                           typename std::enable_if<_Undirected>::type* = 0) {
    // incoming neighbors are the same as outgoing neighbors in undirected
    // graphs
    return findEdge(src, dst, mflag);
  }

  //! Find if an incoming edge between src and dst exists for directed in-out
  //! graphs
  template <bool _DirectedInOut = (Directional && InOut)>
  in_edge_iterator
  findInEdge(GraphNode src, GraphNode dst,
             galois::MethodFlag mflag                       = MethodFlag::WRITE,
             typename std::enable_if<_DirectedInOut>::type* = 0) {
    assert(src);
    assert(dst);
    src->acquire(mflag);
    typename gNodeTypes::iterator ii = src->find(dst, true), ei = src->end();
    is_in_edge edge_predicate;
    if (ii != ei && edge_predicate(*ii)) {
      // After finding edges, lock dst and verify still active
      dst->acquire(mflag);
      if (!edge_predicate(*ii))
        // need this to avoid returning a random iterator
        ii = ei;
    } else
      ii = ei;
    return boost::make_filter_iterator(edge_predicate, ii, ei);
  }

  /**
   * Returns the edge data associated with the edge. It is an error to
   * get the edge data for a non-existent edge.  It is an error to get
   * edge data for inactive edges. By default, the mflag is
   * galois::MethodFlag::UNPROTECTED because edge_begin() dominates this call
   * and should perform the appropriate locking.
   */
  edge_data_reference
  getEdgeData(edge_iterator ii,
              galois::MethodFlag mflag = MethodFlag::UNPROTECTED) const {
    assert(ii->first()->active);
    // galois::runtime::checkWrite(mflag, false);
    ii->first()->acquire(mflag);
    return *ii->second();
  }

  /**
   * Get edge data for an in-edge
   */
  edge_data_reference
  getEdgeData(in_edge_iterator ii,
              galois::MethodFlag mflag = MethodFlag::UNPROTECTED) const {
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

  //! Returns the destination of an in-edge
  GraphNode getEdgeDst(in_edge_iterator ii) {
    assert(ii->first()->active);
    return GraphNode(ii->first());
  }

  //! Sorts edge of a node by destination.
  void sortEdgesByDst(GraphNode N,
                      galois::MethodFlag mflag = MethodFlag::WRITE) {
    // acquire(N, mflag);
    typedef typename gNode::EdgeInfo EdgeInfo;
    std::sort(N->begin(), N->end(),
              [=](const EdgeInfo& e1, const EdgeInfo& e2) {
                return e1.first() < e2.first();
              });
  }

  //! Sort all edges by destination
  void sortAllEdgesByDst(MethodFlag mflag = MethodFlag::WRITE) {
    galois::do_all(
        galois::iterate(*this),
        [=](GraphNode N) { this->sortEdgesByDst(N, mflag); }, galois::steal());
  }

  // General Things

  //! Returns an iterator to the neighbors of a node
  edge_iterator edge_begin(GraphNode N,
                           galois::MethodFlag mflag = MethodFlag::WRITE) {
    assert(N);
    N->acquire(mflag);

    if (galois::runtime::shouldLock(mflag)) {
      for (typename gNode::iterator ii = N->begin(), ee = N->end(); ii != ee;
           ++ii) {
        if (ii->first()->active && !ii->isInEdge())
          ii->first()->acquire(mflag);
      }
    }
    return boost::make_filter_iterator(is_out_edge(), N->begin(), N->end());
  }

  //! Returns an iterator to the in-neighbors of a node
  template <bool _Undirected = !Directional>
  in_edge_iterator
  in_edge_begin(GraphNode N, galois::MethodFlag mflag = MethodFlag::WRITE,
                typename std::enable_if<!_Undirected>::type* = 0) {
    assert(N);
    N->acquire(mflag);

    if (galois::runtime::shouldLock(mflag)) {
      for (typename gNode::iterator ii = N->begin(), ee = N->end(); ii != ee;
           ++ii) {
        if (ii->first()->active && ii->isInEdge())
          ii->first()->acquire(mflag);
      }
    }
    return boost::make_filter_iterator(is_in_edge(), N->begin(), N->end());
  }

  //! Returns an iterator to the in-neighbors of a node; undirected case
  //! in which it's the same as a regular neighbor
  template <bool _Undirected = !Directional>
  edge_iterator in_edge_begin(GraphNode N,
                              galois::MethodFlag mflag = MethodFlag::WRITE,
                              typename std::enable_if<_Undirected>::type* = 0) {
    return edge_begin(N, mflag);
  }

  //! Returns the end of the neighbor edge iterator
  edge_iterator
  edge_end(GraphNode N,
           galois::MethodFlag GALOIS_UNUSED(mflag) = MethodFlag::WRITE) {
    assert(N);
    // Acquiring lock is not necessary: no valid use for an end pointer should
    // ever require it
    // N->acquire(mflag);
    return boost::make_filter_iterator(is_out_edge(), N->end(), N->end());
  }

  uint64_t getDegree(GraphNode N) {
    uint64_t ret;
    for (auto& edge : out_edges(N)) {
      ret++;
    }
    return ret;
  }

  //! Returns the end of an in-neighbor edge iterator
  template <bool _Undirected = !Directional>
  in_edge_iterator
  in_edge_end(GraphNode N,
              galois::MethodFlag GALOIS_UNUSED(mflag)      = MethodFlag::WRITE,
              typename std::enable_if<!_Undirected>::type* = 0) {
    assert(N);
    // Acquiring lock is not necessary: no valid use for an end pointer should
    // ever require it
    // N->acquire(mflag);
    return boost::make_filter_iterator(is_in_edge(), N->end(), N->end());
  }

  //! Returns the end of an in-neighbor edge iterator, undirected case
  template <bool _Undirected = !Directional>
  edge_iterator in_edge_end(GraphNode N,
                            galois::MethodFlag mflag = MethodFlag::WRITE,
                            typename std::enable_if<_Undirected>::type* = 0) {
    return edge_end(N, mflag);
  }

  //! Return a range of edges that can be iterated over by C++ for-each
  runtime::iterable<NoDerefIterator<edge_iterator>>
  edges(GraphNode N, galois::MethodFlag mflag = MethodFlag::WRITE) {
    return internal::make_no_deref_range(edge_begin(N, mflag),
                                         edge_end(N, mflag));
  }

  //! Return a range of in-edges that can be iterated over by C++ for-each
  template <bool _Undirected = !Directional>
  runtime::iterable<NoDerefIterator<in_edge_iterator>>
  in_edges(GraphNode N, galois::MethodFlag mflag = MethodFlag::WRITE,
           typename std::enable_if<!_Undirected>::type* = 0) {
    return internal::make_no_deref_range(in_edge_begin(N, mflag),
                                         in_edge_end(N, mflag));
  }

  //! Return a range of in-edges that can be iterated over by C++ for-each
  //! Undirected case, equivalent to out-edge iteration
  template <bool _Undirected = !Directional>
  runtime::iterable<NoDerefIterator<edge_iterator>>
  in_edges(GraphNode N, galois::MethodFlag mflag = MethodFlag::WRITE,
           typename std::enable_if<_Undirected>::type* = 0) {
    return edges(N, mflag);
  }

  /**
   * An object with begin() and end() methods to iterate over the outgoing
   * edges of N.
   */
  internal::EdgesIterator<MorphGraph>
  out_edges(GraphNode N, MethodFlag mflag = MethodFlag::WRITE) {
    return internal::EdgesIterator<MorphGraph>(*this, N, mflag);
  }

  /**
   * Returns an iterator to all the nodes in the graph. Not thread-safe.
   */
  iterator begin() {
    return boost::make_transform_iterator(
        boost::make_filter_iterator(is_node(), nodes.begin(), nodes.end()),
        makeGraphNode());
  }

  //! Returns the end of the node iterator. Not thread-safe.
  iterator end() {
    return boost::make_transform_iterator(
        boost::make_filter_iterator(is_node(), nodes.end(), nodes.end()),
        makeGraphNode());
  }

  //! local iterator over nodes
  using local_iterator = iterator;

  //! Return the beginning of local range of nodes
  local_iterator local_begin() {
    return boost::make_transform_iterator(
        boost::make_filter_iterator(is_node(), nodes.local_begin(),
                                    nodes.local_end()),
        makeGraphNode());
  }

  //! Return the end of local range of nodes
  local_iterator local_end() {
    return boost::make_transform_iterator(
        boost::make_filter_iterator(is_node(), nodes.local_end(),
                                    nodes.local_end()),
        makeGraphNode());
  }

  /**
   * Returns the number of nodes in the graph. Not thread-safe.
   */
  unsigned int size() { return std::distance(begin(), end()); }

  //! Returns the size of edge data.
  size_t sizeOfEdgeData() const { return gNode::EdgeInfo::sizeOfSecond(); }

  MorphGraph() = default;

  template <typename EdgeNumFnTy, typename EdgeDstFnTy, typename EdgeDataFnTy>
  MorphGraph(uint32_t numNodes, uint64_t numEdges, EdgeNumFnTy edgeNum,
             EdgeDstFnTy _edgeDst, EdgeDataFnTy _edgeData) {
    std::vector<GraphNode> nodes{numNodes};
    for (size_t n = 0; n < numNodes; ++n) {
      // NodeTy node;
      GraphNode a = this->createNode();
      this->addNode(a);
      nodes[n] = a;
    }
    for (size_t n = 0; n < numNodes; ++n) {
      for (size_t e = 0; e < edgeNum(n); ++e) {
        auto edge = this->addEdge(nodes[n], nodes[_edgeDst(n, e)]);
        if (!std::is_void<EdgeTy>::value)
          this->getEdgeData(edge) = _edgeData(n, e);
      }
    }
  }

#ifdef AUX_MAP
  /**
   * Allocate memory for nodes given a file graph with a particular number of
   * nodes.
   *
   * @param graph FileGraph with a number of nodes to allocate
   * @param aux Data structure in which to allocate space for nodes.
   */
  void allocateFrom(FileGraph& graph, ReadGraphAuxData& aux) {
    size_t numNodes = graph.size();
    aux.nodes.allocateInterleaved(numNodes);
  }

  /**
   * Constructs the MorphGraph nodes given a FileGraph to construct it from.
   * Meant to be called by multiple threads.
   *
   * @param[in] graph FileGraph to construct a morph graph from
   * @param[in] tid Thread id of thread calling this function
   * @param[in] total Total number of threads in current execution
   * @param[in,out] aux Allocated memory to store newly created nodes
   */
  void constructNodesFrom(FileGraph& graph, unsigned tid, unsigned total,
                          ReadGraphAuxData& aux) {
    auto r = graph
                 .divideByNode(sizeof(gNode), sizeof(typename gNode::EdgeInfo),
                               tid, total)
                 .first;
    for (FileGraph::iterator ii = r.first, ei = r.second; ii != ei; ++ii) {
      aux.nodes[*ii] = createNode();
      addNode(aux.nodes[*ii], galois::MethodFlag::UNPROTECTED);
    }
  }

  /**
   * Constructs the MorphGraph edges given a FileGraph to construct it from and
   * already created nodes.
   * Meant to be called by multiple threads.
   *
   * @param[in] graph FileGraph to construct a morph graph from
   * @param[in] tid Thread id of thread calling this function
   * @param[in] total Total number of threads in current execution
   * @param[in] aux Contains created nodes to create edges for
   */
  void constructOutEdgesFrom(FileGraph& graph, unsigned tid, unsigned total,
                             ReadGraphAuxData& aux) {
    auto r = graph
                 .divideByNode(sizeof(gNode), sizeof(typename gNode::EdgeInfo),
                               tid, total)
                 .first;
    auto& map = aux.inNghs.get();

    for (FileGraph::iterator ii = r.first, ei = r.second; ii != ei; ++ii) {
      for (FileGraph::edge_iterator nn = graph.edge_begin(*ii),
                                    en = graph.edge_end(*ii);
           nn != en; ++nn) {
        auto dstID = graph.getEdgeDst(nn);
        auto src = aux.nodes[*ii], dst = aux.nodes[dstID];
        auto e = constructOutEdgeValue(graph, nn, src, dst);
        if (!Directional || InOut) {
          map[dstID].push_back({src, e});
        }
      }
    }
  }

  /**
   * Constructs the MorphGraph in-edges given a FileGraph to construct it from
   * and already created nodes. Meant to be called by multiple threads.
   * DirectedNotInOut = false version
   *
   * @param[in] graph FileGraph to construct a morph graph from
   * @param[in] tid Thread id of thread calling this function
   * @param[in] total Total number of threads in current execution
   * @param[in] aux Contains created nodes to create edges for
   */
  void constructInEdgesFrom(FileGraph& graph, unsigned tid, unsigned total,
                            const ReadGraphAuxData& aux) {
    // only do it if not directioal or an inout graph
    if (!Directional || InOut) {
      auto r = graph
                   .divideByNode(sizeof(gNode),
                                 sizeof(typename gNode::EdgeInfo), tid, total)
                   .first;

      for (size_t i = 0; i < aux.inNghs.numRows(); ++i) {
        const auto& map = aux.inNghs.get(i);
        auto ii         = map.lower_bound(*(r.first));  // inclusive begin
        auto ei         = map.lower_bound(*(r.second)); // exclusive end
        for (; ii != ei; ++ii) {
          auto dst = aux.nodes[ii->first];
          for (const auto& ie : ii->second) {
            constructInEdgeValue(graph, ie.second, ie.first, dst);
          }
        }
      }
    }
  }
#else
  /**
   * Allocate memory for nodes given a file graph with a particular number of
   * nodes.
   *
   * @param graph FileGraph with a number of nodes to allocate
   * @param aux Data structure in which to allocate space for nodes.
   */
  void allocateFrom(FileGraph& graph, ReadGraphAuxData& aux) {
    size_t numNodes = graph.size();
    aux.allocateInterleaved(numNodes);

    if (!DirectedNotInOut) {
      galois::do_all(galois::iterate(size_t{0}, aux.size()),
                     [&](size_t index) { aux.constructAt(index); });
    }
  }

  /**
   * Constructs the MorphGraph nodes given a FileGraph to construct it from.
   * Meant to be called by multiple threads.
   * Version for DirectedNotInOut = false.
   *
   * @param[in] graph FileGraph to construct a morph graph from
   * @param[in] tid Thread id of thread calling this function
   * @param[in] total Total number of threads in current execution
   * @param[in,out] aux Allocated memory to store newly created nodes
   */
  template <bool V = DirectedNotInOut>
  std::enable_if_t<!V> constructNodesFrom(FileGraph& graph, unsigned tid,
                                          unsigned total,
                                          ReadGraphAuxData& aux) {
    auto r = graph
                 .divideByNode(sizeof(gNode), sizeof(typename gNode::EdgeInfo),
                               tid, total)
                 .first;
    for (FileGraph::iterator ii = r.first, ei = r.second; ii != ei; ++ii) {
      auto& auxNode = aux[*ii].get();
      auxNode.n     = createNode();
      addNode(auxNode.n, galois::MethodFlag::UNPROTECTED);
    }
  }

  /**
   * Constructs the MorphGraph nodes given a FileGraph to construct it from.
   * Meant to be called by multiple threads.
   * Version for DirectedNotInOut = true.
   *
   * @param[in] graph FileGraph to construct a morph graph from
   * @param[in] tid Thread id of thread calling this function
   * @param[in] total Total number of threads in current execution
   * @param[in,out] aux Allocated memory to store newly created nodes
   */
  template <bool V = DirectedNotInOut>
  std::enable_if_t<V> constructNodesFrom(FileGraph& graph, unsigned tid,
                                         unsigned total,
                                         ReadGraphAuxData& aux) {
    auto r = graph
                 .divideByNode(sizeof(gNode), sizeof(typename gNode::EdgeInfo),
                               tid, total)
                 .first;
    for (FileGraph::iterator ii = r.first, ei = r.second; ii != ei; ++ii) {
      aux[*ii] = createNode();
      addNode(aux[*ii], galois::MethodFlag::UNPROTECTED);
    }
  }

  /**
   * Constructs the MorphGraph edges given a FileGraph to construct it from and
   * already created nodes.
   * Meant to be called by multiple threads.
   * DirectedNotInOut = false version
   *
   * @param[in] graph FileGraph to construct a morph graph from
   * @param[in] tid Thread id of thread calling this function
   * @param[in] total Total number of threads in current execution
   * @param[in] aux Contains created nodes to create edges for
   */
  template <bool V = DirectedNotInOut>
  std::enable_if_t<!V> constructOutEdgesFrom(FileGraph& graph, unsigned tid,
                                             unsigned total,
                                             ReadGraphAuxData& aux) {
    auto r = graph
                 .divideByNode(sizeof(gNode), sizeof(typename gNode::EdgeInfo),
                               tid, total)
                 .first;

    for (FileGraph::iterator ii = r.first, ei = r.second; ii != ei; ++ii) {
      for (FileGraph::edge_iterator nn = graph.edge_begin(*ii),
                                    en = graph.edge_end(*ii);
           nn != en; ++nn) {
        auto src     = aux[*ii].get().n;
        auto& dstAux = aux[graph.getEdgeDst(nn)].get();
        auto e       = constructOutEdgeValue(graph, nn, src, dstAux.n);
        dstAux.lock.lock();
        dstAux.inNghs.push_back({src, e});
        dstAux.lock.unlock();
      }
    }
  }

  /**
   * Constructs the MorphGraph edges given a FileGraph to construct it from and
   * already created nodes.
   * Meant to be called by multiple threads.
   * DirectedNotInOut = true version
   *
   * @param[in] graph FileGraph to construct a morph graph from
   * @param[in] tid Thread id of thread calling this function
   * @param[in] total Total number of threads in current execution
   * @param[in] aux Contains created nodes to create edges for
   */
  template <bool V = DirectedNotInOut>
  std::enable_if_t<V> constructOutEdgesFrom(FileGraph& graph, unsigned tid,
                                            unsigned total,
                                            const ReadGraphAuxData& aux) {
    auto r = graph
                 .divideByNode(sizeof(gNode), sizeof(typename gNode::EdgeInfo),
                               tid, total)
                 .first;

    for (FileGraph::iterator ii = r.first, ei = r.second; ii != ei; ++ii) {
      for (FileGraph::edge_iterator nn = graph.edge_begin(*ii),
                                    en = graph.edge_end(*ii);
           nn != en; ++nn) {
        constructOutEdgeValue(graph, nn, aux[*ii], aux[graph.getEdgeDst(nn)]);
      }
    }
  }

  /**
   * Constructs the MorphGraph in-edges given a FileGraph to construct it from
   * and already created nodes. Meant to be called by multiple threads.
   * DirectedNotInOut = false version
   *
   * @param[in] graph FileGraph to construct a morph graph from
   * @param[in] tid Thread id of thread calling this function
   * @param[in] total Total number of threads in current execution
   * @param[in] aux Contains created nodes to create edges for
   */
  template <bool V = DirectedNotInOut>
  std::enable_if_t<!V> constructInEdgesFrom(FileGraph& graph, unsigned tid,
                                            unsigned total,
                                            ReadGraphAuxData& aux) {
    auto r = graph
                 .divideByNode(sizeof(gNode), sizeof(typename gNode::EdgeInfo),
                               tid, total)
                 .first;

    for (FileGraph::iterator ii = r.first, ei = r.second; ii != ei; ++ii) {
      auto& auxNode = aux[*ii].get();
      for (auto ie : auxNode.inNghs) {
        constructInEdgeValue(graph, ie.second, ie.first, auxNode.n);
      }
    }
  }

  //! If a directed graph and no in-edges exist (i.e. DirectedNotInOut = true),
  //! then construct in edges should do nothing.
  template <bool V = DirectedNotInOut>
  std::enable_if_t<V> constructInEdgesFrom(FileGraph&, unsigned, unsigned,
                                           ReadGraphAuxData&) {}
#endif
};

} // namespace graphs
} // namespace galois
#endif
