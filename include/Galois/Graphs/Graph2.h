/** Basic v2 graphs -*- C++ -*-
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
#ifndef GALOIS_FASTGRAPHS_GRAPH_H
#define GALOIS_FASTGRAPHS_GRAPH_H

#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/filter_iterator.hpp>
#include <boost/functional.hpp>

#include "Galois/ConflictFlags.h"
#include "Galois/Runtime/Support.h"
#include "Galois/Runtime/Context.h"
#include "Galois/Runtime/InsBag.h"

#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <deque>

using namespace GaloisRuntime;

namespace Galois {
namespace Graph {

/**
 * Wrapper class to have a valid type on void edges
 */
template<typename NTy, typename ETy>
struct EdgeItem {
  NTy* N;
  ETy* E;
  
  inline NTy*&       first()        { return N; }
  inline NTy* const& first()  const { return N; }
  inline ETy*&       second()       { return E; }
  inline ETy* const& second() const { return E; }
  EdgeItem(NTy* n, ETy* e) : N(n), E(e) {}
};

template<typename NTy>
struct EdgeItem<NTy, void> {
  NTy* N;
  inline NTy*&       first()        { return N; }
  inline NTy* const& first()  const { return N; }
  inline void*       second() const { return static_cast<void*>(NULL); }
  EdgeItem(NTy* n, void* e) : N(n) {}
};

/**
 * A Graph.
 *
 * @param NodeTy Type of node data
 * @param EdgeTy Type of edge data
 * @param Directional true if graph is directed
 */
template<typename NodeTy, typename EdgeTy, bool Directional>
class FirstGraphImpl {
  template<typename T>
  struct first_eq {
    T N2;
    first_eq(T& n) :N2(n) {}
    template <typename T2>
    bool operator()(const T2& ii) const { return ii.first() == N2; }
  };

  struct gNode: public GaloisRuntime::Lockable {
    //! The storage type for an edge
    typedef EdgeItem<gNode, EdgeTy> EITy;
    
    //! The storage type for edges
    typedef llvm::SmallVector<EITy, 3> EdgesTy;

    struct is_active_edge : public std::unary_function<EITy&, bool> {
      bool operator()(const EITy& e) const { return e.first()->active; }
    };

    typedef boost::filter_iterator<is_active_edge,typename EdgesTy::iterator> iterator;

    EdgesTy edges;
    NodeTy data;
    bool active;

    gNode(const NodeTy& d) :data(d), active(false) { }
    gNode() :active(false) { }

    iterator begin() {
      return boost::make_filter_iterator(is_active_edge(), edges.begin(), edges.end());
    }
    iterator end()   {
      return boost::make_filter_iterator(is_active_edge(), edges.end(), edges.end());
    }
    void erase(iterator ii) { edges.erase(ii.base()); }
    void erase(gNode* N) { 
      iterator ii = find(N);
      if (ii != end())
        edges.erase(ii.base()); 
    }

    iterator find(gNode* N) {
      return std::find_if(begin(), end(), first_eq<gNode*>(N));
    }

    iterator createEdge(gNode* N, EdgeTy* D) {
      return edges.insert(edges.end(), EITy(N,D));
    }
  };

  struct is_active_node : public std::unary_function<gNode&, bool>{
    bool operator() (const gNode& g) const { return g.active; }
  };
  struct sort_count {
    bool operator() (const gNode* g1, const gNode* g2) const { return g1->count < g2->count; }
  };

  //The graph manages the lifetimes of the data in the nodes and edges
  typedef GaloisRuntime::galois_insert_bag<gNode> NodeListTy;
  NodeListTy nodes;

public:
  /**
   * An opaque handle to a graph node.
   */
  // class GraphNode { 
  //   friend class FirstGraph;
  //   gNode* ID;
  //   explicit GraphNode(gNode* id) :ID(id) {}
  // public:
  //   GraphNode() :ID(0) {}
  //   ~GraphNode() { if (ID) __sync_fetch_and_sub(&ID->count, 1); }

  //   bool operator!=(const GraphNode& rhs) const { return ID != rhs; }
  //   bool operator==(const GraphNode& rhs) const { return ID == rhs; }
  //   bool operator< (const GraphNode& rhs) const { return ID < rhs;  }
  //   bool operator> (const GraphNode& rhs) const { return ID > rhs;  }
  // };

  typedef gNode* GraphNode;

private:
  // Helpers for the iterator classes
  struct makeGraphNode: public std::unary_function<gNode, GraphNode> {
    GraphNode operator()(gNode& data) const { return GraphNode(&data); }
  };

public:
  typedef typename gNode::iterator edge_iterator;

  //// Node Handling ////
  
  /**
   * Creates a new node holding the indicated data.
   */
  GraphNode createNode(const NodeTy& n, Galois::MethodFlag mflag = ALL) {
    gNode* N = &(nodes.push(gNode(n)));
    N->active = true;
    acquire(N, mflag);
    return GraphNode(N);
  }

  //! Gets the node data for a node.
  NodeTy& getData(const GraphNode& n, Galois::MethodFlag mflag = ALL) const {
    assert(n);
    acquire(n, mflag);
    return n->data;
  }

  //! Checks if a node is in the graph
  bool containsNode(const GraphNode& n, Galois::MethodFlag mflag = ALL) const {
    if (!n || !n->active) return false;
    acquire(n, mflag);
    return n->active;
  }

  /**
   * Removes a node from the graph along with all its outgoing/incoming edges
   * for undirected graphs or outgoing edges for directed graphs.
   */
  void removeNode(GraphNode n, Galois::MethodFlag mflag = ALL) {
    assert(n);
    acquire(n, mflag);
    gNode* N = n;
    if (N->active) {
      N->active = false;
      //erase the in-edges first
      if (!Directional) {
	for (unsigned int i = 0; i < N->edges.size(); ++i) {
	  gNode* dst = N->edges[i].first();
	  if (dst != N) { // don't handle loops yet
	    dst->erase(N);
	  }
	}
      }
      N->edges.clear();
    }
  }

  //// Edge Handling ////
protected:
  //! Adds an edge to graph
  edge_iterator addEdgeInternal(GraphNode src, GraphNode dst, EdgeTy* data, bool checkDupes, Galois::MethodFlag mflag = ALL) {
    assert(src);
    assert(dst);
    acquire(src, mflag);
    edge_iterator ii;
    if (checkDupes)
      ii = src->find(dst);
    if (!checkDupes || ii == src->end()) {
      if (Directional) {
	ii = src->createEdge(dst, data);
      } else {
	acquire(dst, mflag);
	dst->createEdge(src, data);
	ii = src->createEdge(dst, data);
      }
    }
    return ii;
  }

public:
  //! Removes an edge from the graph
  void removeEdge(GraphNode src, edge_iterator dst, Galois::MethodFlag mflag = ALL) {
    assert(src);
    acquire(src, mflag);
    if (Directional) {
      src->eraseEdge(dst);
    } else {
      acquire(dst->first(), mflag);
      src->eraseEdge(dst);
      dst->eraseEdge(dst->findEdge(src));
    }
  }

  edge_iterator findEdge(GraphNode src, GraphNode dst, Galois::MethodFlag mflag = ALL) {
    assert(src);
    assert(dst);
    acquire(src, mflag);
    return src->find(dst);
  }

  GraphNode getEdgeDst(edge_iterator ii) const {
    return GraphNode(ii->first());
  }

  //// General Things ////

  //! Returns an iterator to the neighbors of a node 
  edge_iterator edge_begin(GraphNode N, Galois::MethodFlag mflag = ALL) {
    assert(N);
    acquire(N, mflag);

    if (shouldLock(mflag)) {
      for (typename gNode::iterator ii = N->begin(), ee = N->end();
	   ii != ee; ++ii) {
        acquire(ii->first(), mflag);
      }
    }
    return N->begin();
  }

  //! Returns the end of the neighbor iterator 
  edge_iterator edge_end(GraphNode N, Galois::MethodFlag mflag = ALL) {
    assert(N);
    // Not necessary; no valid use for an end pointer should ever require it
    //if (shouldLock(mflag))
    //  acquire(N);
    return N->end();
  }

  //These are not thread safe!!
  typedef boost::transform_iterator<makeGraphNode,
          boost::filter_iterator<is_active_node,
                   typename NodeListTy::iterator> > active_iterator;

  /**
   * Returns an iterator to all the nodes in the graph. Not thread-safe.
   */
  active_iterator active_begin() {
    return boost::make_transform_iterator(
           boost::make_filter_iterator(is_active_node(),
				       nodes.begin(), nodes.end()),
	   makeGraphNode());
  }

  //! Returns the end of the node iterator. Not thread-safe.
  active_iterator active_end() {
    return boost::make_transform_iterator(
           boost::make_filter_iterator(is_active_node(),
				       nodes.end(), nodes.end()), 
	   makeGraphNode());
  }

  /**
   * Returns the number of nodes in the graph. Not thread-safe.
   */
  unsigned int size () {
    return std::distance(active_begin(), active_end());
  }

  FirstGraphImpl() {
    // std::cerr << "NodeSize " << sizeof(gNode) << "\n";
    // std::cerr << "NodeDataSize " << sizeof(NodeTy) << "\n";
    // std::cerr << "NodeEdgesSize " << sizeof(typename gNode::EdgesTy) << "\n";
  }

protected:
  template<typename GTy>
  void copyNodes(GTy& graph, std::map<typename GTy::GraphNode, GraphNode>& nodeMap) {
    //mapping between nodes
    //copy nodes
    for (typename GTy::active_iterator ii = graph.active_begin(), 
	   ee = graph.active_end(); ii != ee; ++ii) {
      GraphNode N = createNode(graph.getData(*ii));
      addNode(N);
      nodeMap[*ii] = N;
    }
  }
};

template<typename NodeTy,typename EdgeTy,bool Directional>
class FirstGraph: public FirstGraphImpl<NodeTy,EdgeTy,Directional> {
  typedef FirstGraphImpl<NodeTy,EdgeTy,Directional> Super;

  typedef GaloisRuntime::galois_insert_bag<EdgeTy> EdgeListTy;
  EdgeListTy edges;

public:
  typedef NodeTy node_type;
  typedef typename Super::edge_iterator edge_iterator;
  typedef typename Super::GraphNode GraphNode;
  typedef typename Super::active_iterator active_iterator;

  /**
   * Returns the edge data associated with the edge. It is an error to
   * get the edge data for a non-existent edge.
   */
  EdgeTy& getEdgeData(edge_iterator dst) const {
    return *dst->second();
  }

  //! Adds an edge to graph
  edge_iterator addEdge(GraphNode src, GraphNode dst, const EdgeTy& data, Galois::MethodFlag mflag = ALL) {
    EdgeTy* d = &(edges.push(data));
    return this->addEdgeInternal(src, dst, d, true, mflag);
  }

  edge_iterator addMultiEdge(GraphNode src, GraphNode dst, const EdgeTy& data, Galois::MethodFlag mflag = ALL) {
    EdgeTy* d = &(edges.push(data));
    return this->addEdgeInternal(src, dst, d, false, mflag);
  }

  template<typename GTy>
  void copyGraph(GTy& graph) {
    //mapping between nodes
    std::map<typename GTy::GraphNode, GraphNode> nodeMap;
    copyNodes(graph, nodeMap);
    //copy edges
    for (typename GTy::active_iterator ii = graph.active_begin(), 
	   ee = graph.active_end(); ii != ee; ++ii)
      for(typename GTy::neighbor_iterator ni = graph.neighbor_begin(*ii), 
	    ne = graph.neighbor_end(*ii);
	  ni != ne; ++ni)
	addMultiEdge(nodeMap[*ii], nodeMap[*ni], graph.getEdgeData(*ii, *ni));
  }
};

template<typename NodeTy,bool Directional>
class FirstGraph<NodeTy,void,Directional>: public FirstGraphImpl<NodeTy,void,Directional> {
  typedef FirstGraphImpl<NodeTy,void,Directional> Super;

public:
  typedef NodeTy node_type;
  typedef typename Super::edge_iterator edge_iterator;
  typedef typename Super::GraphNode GraphNode;
  typedef typename Super::active_iterator active_iterator;

  //! Adds an edge to graph
  edge_iterator addEdge(GraphNode src, GraphNode dst, Galois::MethodFlag mflag = ALL) {
    return addEdgeInternal(src, dst, 0, true, mflag);
  }

  edge_iterator addMultiEdge(GraphNode src, GraphNode dst, Galois::MethodFlag mflag = ALL) {
    return addEdgeInternal(src, dst, 0, false, mflag);
  }

  template<typename GTy>
  void copyGraph(GTy& graph) {
    //mapping between nodes
    std::map<typename GTy::GraphNode, GraphNode> nodeMap;
    copyNodes(graph, nodeMap);
    //copy edges
    for (typename GTy::active_iterator ii = graph.active_begin(), 
	   ee = graph.active_end(); ii != ee; ++ii)
      for(typename GTy::neighbor_iterator ni = graph.neighbor_begin(*ii), 
	    ne = graph.neighbor_end(*ii);
	  ni != ne; ++ni)
	addMultiEdge(nodeMap[*ii], nodeMap[*ni]);
  }
};

}
}
#endif
