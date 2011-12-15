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

#include "Galois/ConflictFlags.h"
#include "Galois/Runtime/Support.h"
#include "Galois/Runtime/Context.h"
#include "Galois/Runtime/InsBag.h"
//#include "Galois/Runtime/MemRegionPool.h"
#include "llvm/ADT/SmallVector.h"

using namespace GaloisRuntime;

namespace Galois {
namespace Graph {

////////////////////////////////////////////////////////////////////////////////

/**
 * Wrapper class to have a valid type on void 
 */
template<typename T>
struct VoidWrapper {
  typedef T type;
  typedef T& ref_type;
  typedef const T& const_ref_type;
};

template<>
struct VoidWrapper<void> {
  struct unit { 
    bool operator<(const unit& a) {
      return true;
    }
  };
  typedef unit type;
  typedef unit ref_type;
  typedef const unit const_ref_type;
};

//! Short name for unit for use with FirstGraph::createNode for unit node data
typedef VoidWrapper<void>::unit GraphUnit;

/**
 * Wrapper class to have a valid type on void nodes
 */
template<typename NTy>
struct NodeItem {
  NTy N;
  NodeItem(typename VoidWrapper<NTy>::const_ref_type n) : N(n) { }
  NodeItem() :N() { }
  inline typename VoidWrapper<NTy>::ref_type getData() {
    return N;
  }
};

template<>
struct NodeItem<void> {
  NodeItem(VoidWrapper<void>::const_ref_type n) { }
  inline VoidWrapper<void>::ref_type getData() {
    return VoidWrapper<void>::ref_type();
  }
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
  inline EdgeItem(NTy& n) : N(n) { }
  inline EdgeItem(NTy& n, ETy e) : N(n), E(e) { }
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
  inline EdgeItem(NTy& n) : N(n) { }
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
public:
  //! A reference to an edge
  typedef typename VoidWrapper<EdgeTy>::ref_type edge_reference;
  //! A reference to a const edge
  typedef typename VoidWrapper<EdgeTy>::const_ref_type const_edge_reference;
  //! A reference to a node
  typedef typename VoidWrapper<NodeTy>::ref_type node_reference;
  //! A reference to a const node
  typedef typename VoidWrapper<NodeTy>::const_ref_type const_node_reference;
  //! A node 
  typedef typename VoidWrapper<NodeTy>::type node_type;
  //! An edge
  typedef typename VoidWrapper<EdgeTy>::type edge_type;

private:
  struct gNode: public GaloisRuntime::Lockable {
    //! The storage type for an edge
    typedef EdgeItem<gNode*, EdgeTy> EITy;
    //! The storage type for a node
    typedef NodeItem<NodeTy> NITy;

    //! The storage type for edges
    typedef llvm::SmallVector<EITy, 3> EdgesTy;
    typedef typename EdgesTy::iterator iterator;

    NITy data;
    bool active;
    EdgesTy edges;
    //NITy data;
    //bool active;

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

    gNode(const_node_reference d, bool a)
      :data(d), active(a) {
    }
    gNode() :active(false) {
    }

    void eraseEdge(gNode* N) {
      for (iterator ii = begin(), ee = end(); ii != ee; ++ii) {
	if (ii->getNeighbor() == N) {
	  edges.erase(ii);
	  return;
	}
      }
    }

    edge_reference getEdgeData(gNode* N) {
      for (iterator ii = begin(), ee = end(); ii != ee; ++ii)
	if (ii->getNeighbor() == N)
	  return ii->getData();
      assert(0 && "Edge doesn't exist");
      abort();
    }

    edge_reference getEdgeData(iterator ii) {
      return ii->getData();
    }

    edge_reference createEdge(gNode* N) {
      edges.push_back(EITy(N));
      return edges.back().getData();
    }

    edge_reference getOrCreateEdge(gNode* N) {
      for (iterator ii = begin(), ee = end(); ii != ee; ++ii)
	if (ii->getNeighbor() == N)
	  return ii->getData();
      return createEdge(N);
    }

    edge_reference createEdge(gNode* N, const_edge_reference data) {
      edges.push_back(EITy(N, data));
      return edges.back().getData();
    }

    edge_reference getOrCreateEdge(gNode* N, const_edge_reference data) {
      for (iterator ii = begin(), ee = end(); ii != ee; ++ii)
        if (ii->getNeighbor() == N)
          return ii->getData();
      return createEdge(N, data);
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
  typedef GaloisRuntime::galois_insert_bag<gNode> NodeListTy;
  NodeListTy nodes;

  //GaloisRuntime::MemRegionPool<gNode> NodePool;

  //deal with the Node redirction
  node_reference getData(gNode* ID, Galois::MethodFlag mflag = ALL) {
    assert(ID);
    acquire(ID, mflag);
    return ID->data.getData();
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

    //XXX(ddn): erase after experiment is done
    uintptr_t getID() const {
      return reinterpret_cast<uintptr_t>(ID) >> 7;
    }

    void prefetch_all() {
      if (ID)
	ID->prefetch_neighbors();
    }

    node_reference getData(Galois::MethodFlag mflag = ALL) const {
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

    bool hasNeighbor(const GraphNode& N) const {
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
   * the graph (see addNode() instead). For graphs with void node data, 
   * pass ::GraphUnit instead.
   */
  GraphNode createNode(const_node_reference n) {
    gNode N(n, false);
    return GraphNode(this, &(nodes.push(N)));
  }

  //! Adds a node to the graph.
  bool addNode(const GraphNode& n, Galois::MethodFlag mflag = ALL) {
    assert(n.ID);
    acquire(n.ID, mflag);
    bool oldActive = n.ID->active;
    if (!oldActive) {
      n.ID->active = true;
      //__sync_add_and_fetch(&numActive, 1);
    }
    return !oldActive;
  }

  //add a node and reserve the space for edges
  bool addNode(const GraphNode& n, int maxDegree, MethodFlag mflag = ALL) {
    assert(n.ID);
    acquire(n.ID, mflag);
    bool oldActive = n.ID->active;
    if (!oldActive) {
            n.ID->active = true;
            //__sync_add_and_fetch(&numActive, 1);
    }
    n.ID->edges.reserve(maxDegree);
    return !oldActive;
  }

  //! Gets the node data for a node.
  node_reference getData(const GraphNode& n, Galois::MethodFlag mflag = ALL) const {
    assert(n.ID);
    acquire(n.ID, mflag);
    return n.ID->data.getData();
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
  bool removeNode(GraphNode n, Galois::MethodFlag mflag = ALL) {
    assert(n.ID);
    acquire(n.ID, mflag);
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

  //! Adds an edge to graph, replacing existing value if edge already exists
  edge_reference addEdge(GraphNode src, GraphNode dst,
	       const_edge_reference data, 
	       Galois::MethodFlag mflag = ALL) {
    assert(src.ID);
    assert(dst.ID);
    acquire(src.ID, mflag);
    if (Directional) {
      return src.ID->getOrCreateEdge(dst.ID, data);
    } else {
      acquire(dst.ID, mflag);
      if (src < dst) {
        dst.ID->getOrCreateEdge(src.ID, data);
        return src.ID->getOrCreateEdge(dst.ID, data);
      } else {
        src.ID->getOrCreateEdge(dst.ID, data);
        return dst.ID->getOrCreateEdge(src.ID, data);
      }
    }
  }

  //! Adds an edge to graph, always adding new value, thus permiting multiple edges to from
  //! one node to another. Not defined for undirected graphs.
  edge_reference addMultiEdge(GraphNode src, GraphNode dst, const_edge_reference data,
      Galois::MethodFlag mflag = ALL) {
    assert(src.ID);
    assert(dst.ID);
    acquire(src.ID, mflag);
    if (Directional) {
      return src.ID->createEdge(dst.ID, data);
    } else {
      assert(0 && "Not defined for undirected graphs");
      abort();
    }
  }

  //! Adds an edge to graph, replacing existing value if edge already exists
  edge_reference addEdge(GraphNode src, GraphNode dst, Galois::MethodFlag mflag = ALL) {
    assert(src.ID);
    assert(dst.ID);
    acquire(src.ID, mflag);
    if (Directional) {
      return src.ID->getOrCreateEdge(dst.ID);
    } else {
      acquire(dst.ID, mflag);
      if (src < dst) {
        dst.ID->getOrCreateEdge(src.ID);
        return src.ID->getOrCreateEdge(dst.ID);
      } else {
        src.ID->getOrCreateEdge(dst.ID);
        return dst.ID->getOrCreateEdge(src.ID);
      }
    }
  }

  //! Adds an edge to graph, always adding new value, thus permiting multiple edges to from
  //! one node to another. Not defined for undirected graphs.
  edge_reference addMultiEdge(GraphNode src, GraphNode dst, Galois::MethodFlag mflag = ALL) {
    assert(src.ID);
    assert(dst.ID);
    acquire(src.ID, mflag);
    if (Directional) {
      return src.ID->createEdge(dst.ID);
    } else {
      assert(0 && "Not defined for undirected graphs");
      abort();
    }
  }

  //! Removes an edge from the graph
  void removeEdge(GraphNode src, GraphNode dst, Galois::MethodFlag mflag = ALL) {
    assert(src.ID);
    assert(dst.ID);
    acquire(src.ID, mflag);
    if (Directional) {
      src.ID->eraseEdge(dst.ID);
    } else {
      acquire(dst.ID, mflag);
      src.ID->eraseEdge(dst.ID);
      dst.ID->eraseEdge(src.ID);
    }
  }

  /**
   * Returns the edge data associated with the edge. It is an error to
   * get the edge data for a non-existent edge.
   */
  edge_reference getEdgeData(GraphNode src, GraphNode dst,
      Galois::MethodFlag mflag = ALL) const {
    assert(src.ID);
    assert(dst.ID);

    //yes, fault on null (no edge)
    acquire(src.ID, mflag);

    if (Directional) {
      return src.ID->getEdgeData(dst.ID);
    } else {
      acquire(dst.ID, mflag);
      if (src < dst)
	return src.ID->getEdgeData(dst.ID);
      else
	return dst.ID->getEdgeData(src.ID);
    }
  }

  //// General Things ////

  //! Returns the number of neighbors
  size_t neighborsSize(GraphNode N, Galois::MethodFlag mflag = ALL) const {
    assert(N.ID);
    acquire(N.ID, mflag);
    return N.ID->edges.size();
  }

  typedef typename boost::transform_iterator<makeGraphNodePtr,
					     typename gNode::neighbor_iterator>
                                               neighbor_iterator;

  //! Returns an iterator to the neighbors of a node 
  neighbor_iterator neighbor_begin(GraphNode N, Galois::MethodFlag mflag = ALL) {
    assert(N.ID);
    acquire(N.ID, mflag);

    if (shouldLock(mflag)) {
      for (typename gNode::neighbor_iterator ii = N.ID->neighbor_begin(), ee =
             N.ID->neighbor_end(); ii != ee; ++ii) {
        acquire(*ii, mflag);
      }
    }
    return boost::make_transform_iterator(N.ID->neighbor_begin(),
					  makeGraphNodePtr(this));
  }

  //! Returns the end of the neighbor iterator 
  neighbor_iterator neighbor_end(GraphNode N, Galois::MethodFlag mflag = ALL) {
    assert(N.ID);
    // Not necessary; no valid use for an end pointer should ever require it
    //if (shouldLock(mflag))
    //  acquire(N.ID);
    return boost::make_transform_iterator(N.ID->neighbor_end(),
					  makeGraphNodePtr(this));
  }

  //! Returns edge data given a neighbor iterator; neighbor iterator should be
  //! from neighbor_begin with the same src.
  edge_reference getEdgeData(GraphNode src, neighbor_iterator dst,
      Galois::MethodFlag mflag = ALL) {
    assert(src.ID);

    //yes, fault on null (no edge)
    acquire(src.ID, mflag);

    //TODO(ddn): check that neighbor iterator is from the same source node as src
    if (Directional) {
      return src.ID->getEdgeData(dst.base().base());
    } else {
      if (src.ID < dst.base().base()->getNeighbor())
	return src.ID->getEdgeData(dst.base().base()->getNeighbor());
      else
	return dst.base().base()->getNeighbor()->getEdgeData(src.ID);
    }
  }

  //These are not thread safe!!
  typedef boost::transform_iterator<makeGraphNode,
            boost::filter_iterator<std::mem_fun_ref_t<bool, gNode>,
              typename NodeListTy::iterator> > active_iterator;

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
    //std::cerr << "NodeSize " << sizeof(gNode) << "\n";
  }

  template<typename GTy>
  void copyGraph(GTy& graph) {
    //mapping between nodes
    std::map<typename GTy::GraphNode, GraphNode> NodeMap;
    //copy nodes
    for (typename GTy::active_iterator ii = graph.active_begin(), 
	   ee = graph.active_end(); ii != ee; ++ii) {
      GraphNode N = createNode(graph.getData(*ii));
      addNode(N);
      NodeMap[*ii] = N;
    }
    //copy edges
    for (typename GTy::active_iterator ii = graph.active_begin(), 
	   ee = graph.active_end(); ii != ee; ++ii)
      for(typename GTy::neighbor_iterator ni = graph.neighbor_begin(*ii), 
	    ne = graph.neighbor_end(*ii);
	  ni != ne; ++ni)
	addEdge(NodeMap[*ii], NodeMap[*ni], graph.getEdgeData(*ii, *ni));
  }
};

}
}
#endif
