/** Basic graphs -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
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
 * An example of use:
 * 
 * \code
 * struct Node {
 *   ... // Definition of node data
 * };
 *
 * typedef Galois::Graph::FirstGraph<Node,int,true> Graph;
 * 
 * // Create graph
 * Graph g;
 * Node n1, n2;
 * Graph::GraphNode a, b;
 * a = g.createNode(n1);
 * g.addNode(a);
 * b = g.createNode(n2);
 * g.addNode(b);
 * g.addEdge(a, b, 5);
 *
 * // Traverse graph
 * for (Graph::active_iterator i = g.active_begin(), iend = g.active_end();
 *      i != iend;
 *      ++i) {
 *   Graph::GraphNode src = *i;
 *   for (Graph::neighbor_iterator j = g.neighbor_begin(src),
 *                                 jend = g.neighbor_end(src);
 *        j != jend;
 *        ++j) {
 *     Graph::GraphNode dst = *j;
 *     int edgeData = g.getEdgeData(src, dst);
 *     assert(edgeData == 5);
 *   }
 * }
 * \endcode
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#ifndef GALOIS_GRAPHS_GRAPH_H
#define GALOIS_GRAPHS_GRAPH_H

#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/filter_iterator.hpp>
#include <boost/functional.hpp>

#include "Galois/Runtime/Support.h"
#include "Galois/Runtime/Context.h"
#include "Galois/Runtime/InsBag.h"
//#include "Galois/Runtime/MemRegionPool.h"
#include "LLVM/SmallVector.h"

using namespace GaloisRuntime;

namespace Galois {
namespace Graph {

/**
 * What should the runtime do when executing a method.
 *
 * Graph methods take an optional parameter indicating what actions the runtime
 * should do on the user's behalf: (1) checking for conflicts, and/or (2)
 * saving undo information. By default, both are performed (ALL).
 */
enum MethodFlag {
  NONE, ALL, CHECK_CONFLICT, SAVE_UNDO
};

static inline bool shouldLock(MethodFlag g) {
  switch(g) {
  case NONE:
  case SAVE_UNDO:
    return false;
  case ALL:
  case CHECK_CONFLICT:
    return true;
  }
  assert(0 && "Shouldn't get here");
  abort();
}

////////////////////////////////////////////////////////////////////////////////

/**
 * Wrapper class to have a valid type on void nodes
 */
template<typename T>
struct VoidWrapper {
  typedef T type;
  typedef T& ref_type;
};

template<>
struct VoidWrapper<void> {
  struct unit {
  };
  typedef unit type;
  typedef unit ref_type;
};

/**
 * Wrapper class to have a valid type on void edges
 */
template<typename NTy, typename ETy>
struct EdgeItem {
  NTy N;
  ETy E;
  inline NTy getNeighbor() {
    return N;
  }
  inline ETy& getData() {
    return E;
  }
  inline EdgeItem(NTy& n) : N(n) {
  }

  inline EdgeItem(){ }
};

template<typename NTy>
struct EdgeItem<NTy, void> {
  NTy N;
  inline NTy getNeighbor() {
    return N;
  }
  inline typename VoidWrapper<void>::ref_type getData() {
    return VoidWrapper<void>::ref_type();
  }
  inline EdgeItem(NTy& n) : N(n) {
  }
};

////////////////////////////////////////////////////////////////////////////////

/**
 * A Graph.
 *
 * @param NodeTy Type of node data
 * @param EdgeTy Type of edge data
 * @param Directional true if graph is directed
 */
template<typename NodeTy, typename EdgeTy, bool Directional>
class FirstGraph {

  struct gNode: public GaloisRuntime::Lockable {
    //! The storage type for edges
    typedef EdgeItem<gNode*, EdgeTy> EITy;
    //! The return type for edge data
    typedef typename VoidWrapper<EdgeTy>::ref_type REdgeTy;
    typedef llvm::SmallVector<EITy, 3> edgesTy;
    edgesTy edges;
    NodeTy data;
    bool active;

    typedef typename edgesTy::iterator iterator;

    iterator begin() {
      return edges.begin();
    }
    iterator end() {
      return edges.end();
    }

    struct getNeigh : public std::unary_function<EITy, gNode*> {
      gNode* operator()(EITy& e) const { return e.getNeighbor(); }
    };

    typedef typename boost::transform_iterator<getNeigh, iterator> neighbor_iterator;

    neighbor_iterator neighbor_begin() {
      return boost::make_transform_iterator(begin(), getNeigh());
    }
    neighbor_iterator neighbor_end() {
      return boost::make_transform_iterator(end(), getNeigh());
    }

    gNode(const NodeTy& d, bool a) :
      data(d), active(a) {
    }

    void prefetch_neighbors() {
      for (iterator ii = begin(), ee = end(); ii != ee; ++ii)
	if (ii->getNeighbor())
	  __builtin_prefetch(ii->getNeighbor());
    }

    void eraseEdge(gNode* N) {
      for (iterator ii = begin(), ee = end(); ii != ee; ++ii) {
	if (ii->getNeighbor() == N) {
	  edges.erase(ii);
	  return;
	}
      }
    }

    REdgeTy getEdgeData(gNode* N) {
      for (iterator ii = begin(), ee = end(); ii != ee; ++ii)
	if (ii->getNeighbor() == N)
	  return ii->getData();
      assert(0 && "Edge doesn't exist");
      abort();
    }

    REdgeTy getOrCreateEdge(gNode* N) {
      for (iterator ii = begin(), ee = end(); ii != ee; ++ii)
	if (ii->getNeighbor() == N)
	  return ii->getData();
      edges.push_back(EITy(N));
      return edges.back().getData();
    }

    bool isActive() {
      return active;
    }

    bool hasNeighbor(gNode* N) {
      for (iterator ii = begin(), ee = end(); ii != ee; ++ii)
        if (ii->getNeighbor() == N)
          return true;
      return false;
    }
  };

  //The graph manages the lifetimes of the data in the nodes and edges
  typedef GaloisRuntime::galois_insert_bag<gNode> nodeListTy;
  nodeListTy nodes;

  //GaloisRuntime::MemRegionPool<gNode> NodePool;

  //deal with the Node redirction
  NodeTy& getData(gNode* ID, MethodFlag mflag = ALL) {
    assert(ID);
    if (shouldLock(mflag))
      acquire(ID);
    return ID->data;
  }

public:
  /**
   * An opaque handle to a graph node.
   */
  class GraphNode {
    friend class FirstGraph;
    FirstGraph* Parent;
    gNode* ID;

    explicit GraphNode(FirstGraph* p, gNode* id) :
      Parent(p), ID(id) {
    }

  public:
    GraphNode() :
      Parent(0), ID(0) {
    }

    void prefetch_all() {
      if (ID)
	ID->prefetch_neighbors();
    }

    NodeTy& getData(MethodFlag mflag = ALL) const {
      return Parent->getData(ID, mflag);
    }

    FirstGraph* getGraph() const {
      return Parent;
    }

    bool isNull() const {
      return !Parent;
    }

    bool operator!=(const GraphNode& rhs) const {
      return Parent != rhs.Parent || ID != rhs.ID;
    }

    bool operator==(const GraphNode& rhs) const {
      return Parent == rhs.Parent && ID == rhs.ID;
    }

    bool operator<(const GraphNode& rhs) const {
      return Parent < rhs.Parent || (Parent == rhs.Parent && ID < rhs.ID);
    }

    bool operator>(const GraphNode& rhs) const {
      return Parent > rhs.Parent || (Parent == rhs.Parent && ID > rhs.ID);
    }

    bool hasNeighbor(GraphNode& N) const {
      return ID->hasNeighbor(N.ID);
    }
  };

private:
  // Helpers for the iterator classes
  class makeGraphNode: public std::unary_function<gNode, GraphNode> {
    FirstGraph* G;
  public:
    makeGraphNode(FirstGraph* g = 0) :
      G(g) {
    }
    GraphNode operator()(gNode& data) const {
      return GraphNode(G, &data);
    }
  };

  class makeGraphNodePtr: public std::unary_function<gNode*, GraphNode> {
    FirstGraph* G;
  public:
    makeGraphNodePtr(FirstGraph* g = 0) :
      G(g) {
    }
    GraphNode operator()(gNode* data) const {
      return GraphNode(G, data);
    }
  };

public:
  typedef EdgeTy EdgeDataTy;
  typedef NodeTy NodeDataTy;

  //// Node Handling ////
  
  /**
   * Creates a new node holding the indicated data. The node is not added to
   * the graph (see addNode() instead).
   */
  GraphNode createNode(const NodeTy& n) {
    gNode N(n, false);
    return GraphNode(this, &(nodes.push(N)));
  }

  //! Adds a node to the graph.
  bool addNode(const GraphNode& n, MethodFlag mflag = ALL) {
    assert(n.ID);
    if (shouldLock(mflag))
      acquire(n.ID);
    bool oldActive = n.ID->active;
    if (!oldActive) {
      n.ID->active = true;
      //__sync_add_and_fetch(&numActive, 1);
    }
    return !oldActive;
  }

  //! Gets the node data for a node.
  NodeTy& getData(const GraphNode& n, MethodFlag mflag = ALL) const {
    assert(n.ID);
    if (shouldLock(mflag))
      acquire(n.ID);
    return n.ID->data;
  }

  //! Checks if a node is in the graph (already added)
  bool containsNode(const GraphNode& n) const {
    return n.ID && (n.Parent == this) && n.ID->active;
  }

  /**
   * Removes a node from the graph along with all its outgoing/incoming edges
   * for undirected graphs or outgoing edges for directed graphs.
   * 
   */
  // FIXME: incoming edges aren't handled here for directed graphs
  bool removeNode(GraphNode n, MethodFlag mflag = ALL) {
    assert(n.ID);
    if (shouldLock(mflag))
      acquire(n.ID);
    gNode* N = n.ID;
    bool wasActive = N->active;
    if (wasActive) {
      //__sync_sub_and_fetch(&numActive, 1);
      N->active = false;
      //erase the in-edges first
      for (unsigned int i = 0; i < N->edges.size(); ++i) {
	if (N->edges[i].getNeighbor() != N) // don't handle loops yet
	  N->edges[i].getNeighbor()->eraseEdge(N);
      }
      N->edges.clear();
    }
    return wasActive;
  }

  //// Edge Handling ////

  //! Adds an edge to the graph containing the specified data.
  void addEdge(GraphNode src, GraphNode dst,
	       const typename VoidWrapper<EdgeTy>::type& data, 
	       MethodFlag mflag = ALL) {
    assert(src.ID);
    assert(dst.ID);
    if (shouldLock(mflag)) 
      acquire(src.ID);

    if (Directional) {
      src.ID->getOrCreateEdge(dst.ID) = data;
    } else {
      if (shouldLock(mflag))
	acquire(dst.ID);
      EdgeTy& E1 = src.ID->getOrCreateEdge(dst.ID);
      EdgeTy& E2 = dst.ID->getOrCreateEdge(src.ID);
      if (src < dst)
	E1 = data;
      else
	E2 = data;
    }
  }

  //! Adds an edge to the graph
  void addEdge(GraphNode src, GraphNode dst, MethodFlag mflag = ALL) {
    assert(src.ID);
    assert(dst.ID);
    if (shouldLock(mflag))
      acquire(src.ID);
    if (Directional) {
      src.ID->getOrCreateEdge(dst.ID);
    } else {
      if (shouldLock(mflag))
	acquire(dst.ID);
      src.ID->getOrCreateEdge(dst.ID);
      dst.ID->getOrCreateEdge(src.ID);
    }
  }

  //! Removes an edge from the graph
  void removeEdge(GraphNode src, GraphNode dst, MethodFlag mflag = ALL) {
    assert(src.ID);
    assert(dst.ID);
    if (shouldLock(mflag))
      acquire(src.ID);
    if (Directional) {
      src.ID->eraseEdge(dst.ID);
    } else {
      if (shouldLock(mflag))
	acquire(dst.ID);
      src.ID->eraseEdge(dst.ID);
      dst.ID->eraseEdge(src.ID);
    }
  }

  /**
   * Returns the edge data associated with the edge. It is an error to
   * get the edge data for a non-existent edge.
   */
  typename VoidWrapper<EdgeTy>::type& getEdgeData(GraphNode src, GraphNode dst,
						  MethodFlag mflag = ALL) {
    assert(src.ID);
    assert(dst.ID);

    //yes, fault on null (no edge)
    if (shouldLock(mflag))
      acquire(src.ID);

    if (Directional) {
      return src.ID->getEdgeData(dst.ID);
    } else {
      if (shouldLock(mflag))
	acquire(dst.ID);
      if (src < dst)
	return src.ID->getEdgeData(dst.ID);
      else
	return dst.ID->getEdgeData(src.ID);
    }
  }

  //// General Things ////

  //! Returns the number of neighbors
  int neighborsSize(GraphNode N, MethodFlag mflag = ALL) {
    assert(N.ID);
    if (shouldLock(mflag))
      acquire(N.ID);
    return N.ID->edges.size();
  }

  typedef typename boost::transform_iterator<makeGraphNodePtr,
					     typename gNode::neighbor_iterator>
                                               neighbor_iterator;

  //! Returns an iterator to the neighbors of a node 
  neighbor_iterator neighbor_begin(GraphNode N, MethodFlag mflag = ALL) {
    assert(N.ID);
    if (shouldLock(mflag))
      acquire(N.ID);
    for (typename gNode::neighbor_iterator ii = N.ID->neighbor_begin(), ee =
	   N.ID->neighbor_end(); ii != ee; ++ii) {
      __builtin_prefetch(*ii);
      if (shouldLock(mflag))
	acquire(*ii);
    }
    return boost::make_transform_iterator(N.ID->neighbor_begin(),
					  makeGraphNodePtr(this));
  }

  //! Returns the end of the neighbor iterator 
  neighbor_iterator neighbor_end(GraphNode N, MethodFlag mflag = ALL) {
    assert(N.ID);
    if (shouldLock(mflag))
      // Probably not necessary (no valid use for an end pointer should ever
      // require it)
      acquire(N.ID);
    return boost::make_transform_iterator(N.ID->neighbor_end(),
					  makeGraphNodePtr(this));
  }

  typedef boost::transform_iterator<makeGraphNode,
            boost::filter_iterator<std::mem_fun_ref_t<bool, gNode>,
              typename nodeListTy::iterator> > active_iterator;

  /**
   * Returns an iterator to all the nodes in the graph. Not thread-safe.
   */
  active_iterator active_begin() {
    return boost::make_transform_iterator(
        boost::make_filter_iterator(
          std::mem_fun_ref(&gNode::isActive), nodes.begin(), nodes.end()),
        makeGraphNode(this));
  }

  //! Returns the end of the node iterator. Not thread-safe.
  active_iterator active_end() {
    return boost::make_transform_iterator(
        boost::make_filter_iterator(
          std::mem_fun_ref(&gNode::isActive), nodes.end(), nodes.end()), 
        makeGraphNode(this));
  }

  /**
   * Returns the number of nodes in the graph. Not thread-safe.
   */
  unsigned int size() {
    return std::distance(active_begin(), active_end());
  }

  FirstGraph() {
    reportStat("NodeSize", sizeof(gNode));
  }
};

}
}
#endif
