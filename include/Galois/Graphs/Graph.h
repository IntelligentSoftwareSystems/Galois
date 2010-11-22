// simple graph -*- C++ -*-

#include <iostream>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/filter_iterator.hpp>
#include <boost/functional.hpp>

#include "Galois/Runtime/Context.h"
#include "Galois/Runtime/InsBag.h"
//#include "Galois/Runtime/MemRegionPool.h"
#include "Support/ThreadSafe/TSIBag.h"
#include "LLVM/SmallVector.h"

using namespace GaloisRuntime;

namespace Galois {
namespace Graph {

enum MethodFlag {
  NONE, ALL, CHECK_CONFLICT, SAVE_UNDO
};

static bool shouldLock(MethodFlag g) {
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
// Wrap void so that we can have a valid type on void nodes

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

template<typename NTy, typename ETy>
struct EdgeItem {
  NTy N;
  ETy E;
  NTy getNeighbor() {
    return N;
  }
  ETy& getData() {
    return E;
  }
  EdgeItem(NTy& n) :
    N(n) {
  }

};

template<typename NTy>
struct EdgeItem<NTy, void> {
  NTy N;
  NTy getNeighbor() {
    return N;
  }
  typename VoidWrapper<void>::ref_type getData() {
    return VoidWrapper<void>::ref_type();
  }
  EdgeItem(NTy& n) :
    N(n) {
  }
};

////////////////////////////////////////////////////////////////////////////////

template<typename NodeTy, typename EdgeTy, bool Directional>
class FirstGraph {

  struct gNode: public GaloisRuntime::Lockable {
    //The storage type for edges
    typedef EdgeItem<gNode*, EdgeTy> EITy;
    //The return type for edge data
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

    typedef typename boost::transform_iterator<boost::mem_fun_ref_t<gNode*,
								    EITy>, iterator> neighbor_iterator;

    neighbor_iterator neighbor_begin() {
      return boost::make_transform_iterator(begin(), boost::mem_fun_ref(
									&EITy::getNeighbor));
    }
    neighbor_iterator neighbor_end() {
      return boost::make_transform_iterator(end(), boost::mem_fun_ref(
								      &EITy::getNeighbor));
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
  };

  //The graph manages the lifetimes of the data in the nodes and edges
  typedef GaloisRuntime::galois_insert_bag<gNode> nodeListTy;
  //typedef threadsafe::ts_insert_bag<gNode> nodeListTy;
  nodeListTy nodes;

  //GaloisRuntime::MemRegionPool<gNode> NodePool;

  //deal with the Node redirction
  template<typename Context>
  NodeTy& getData(gNode* ID, MethodFlag mflag = ALL, Context* C = getThreadContext()) {
    assert(ID);
    if (shouldLock(mflag))
      SimpleRuntimeContext::acquire(C, ID);
    return ID->data;
  }

public:
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

    NodeTy& getData(MethodFlag mflag = ALL, SimpleRuntimeContext* C = getThreadContext()) {
      return Parent->getData(ID, mflag, C);
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

  };

private:
  // Helpers for the iterator classes
  class makeGraphNode: public std::unary_function<gNode, GraphNode> {
    FirstGraph* G;
  public:
    makeGraphNode(FirstGraph* g) :
      G(g) {
    }
    GraphNode operator()(gNode& data) const {
      return GraphNode(G, &data);
    }
  };
  class makeGraphNodePtr: public std::unary_function<gNode*, GraphNode> {
    FirstGraph* G;
  public:
    makeGraphNodePtr(FirstGraph* g) :
      G(g) {
    }
    GraphNode operator()(gNode* data) const {
      return GraphNode(G, data);
    }
  };

public:

  // Node Handling

  // Creates a new node holding the indicated data.
  // Node is not added to the graph
  GraphNode createNode(const NodeTy& n) {
    gNode N(n, false);
    return GraphNode(this, &(nodes.push(N)));
  }

  // Adds a node to the graph.
  bool addNode(const GraphNode& n, MethodFlag mflag = ALL, SimpleRuntimeContext* C = getThreadContext()) {
    assert(n.ID);
    if (shouldLock(mflag))
      SimpleRuntimeContext::acquire(C, n.ID);
    bool oldActive = n.ID->active;
    if (!oldActive) {
      n.ID->active = true;
      //__sync_add_and_fetch(&numActive, 1);
    }
    return !oldActive;
  }

  // Check if a node is in the graph (already added)
  bool containsNode(const GraphNode& n) {
    return n.ID && (n.Parent == this) && n.ID->active;
  }

  // Removes a node from the graph along with all its outgoing/incoming edges.
  // FIXME: incoming edges aren't handled here for directed graphs
  bool removeNode(GraphNode n, MethodFlag mflag = ALL, SimpleRuntimeContext* C = getThreadContext()) {
    assert(n.ID);
    if (shouldLock(mflag))
      SimpleRuntimeContext::acquire(C, n.ID);
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

  // Edge Handling

  // Adds an edge to the graph containing the specified data.
  void addEdge(GraphNode src, GraphNode dst,
	       const typename VoidWrapper<EdgeTy>::type& data, 
	       MethodFlag mflag = ALL, SimpleRuntimeContext* C = getThreadContext()) {
    assert(src.ID);
    assert(dst.ID);
    if (shouldLock(mflag)) 
      SimpleRuntimeContext::acquire(C, src.ID);
    if (Directional) {
      src.ID->getOrCreateEdge(dst.ID) = data;
    } else {
      if (shouldLock(mflag))
	SimpleRuntimeContext::acquire(C, dst.ID);
      EdgeTy& E1 = src.ID->getOrCreateEdge(dst.ID);
      EdgeTy& E2 = dst.ID->getOrCreateEdge(src.ID);
      if (src < dst)
	E1 = data;
      else
	E2 = data;
    }
  }

  // Adds an edge to the graph
  void addEdge(GraphNode src, GraphNode dst, MethodFlag mflag = ALL, SimpleRuntimeContext* C = getThreadContext()) {
    assert(src.ID);
    assert(dst.ID);
    if (shouldLock(mflag))
      SimpleRuntimeContext::acquire(C, src.ID);
    if (Directional) {
      src.ID->getOrCreateEdge(dst.ID);
    } else {
      if (shouldLock(mflag))
	SimpleRuntimeContext::acquire(C, dst.ID);
      src.ID->getOrCreateEdge(dst.ID);
      dst.ID->getOrCreateEdge(src.ID);
    }
  }

  void removeEdge(GraphNode src, GraphNode dst, MethodFlag mflag = ALL, SimpleRuntimeContext* C = getThreadContext()) {
    assert(src.ID);
    assert(dst.ID);
    if (shouldLock(mflag))
      SimpleRuntimeContext::acquire(C, src.ID);
    if (Directional) {
      src.ID->eraseEdge(dst.ID);
    } else {
      if (shouldLock(mflag))
	SimpleRuntimeContext::acquire(C, dst.ID);
      src.ID->eraseEdge(dst.ID);
      dst.ID->eraseEdge(src.ID);
    }
  }

  typename VoidWrapper<EdgeTy>::type& getEdgeData(GraphNode src, GraphNode dst,
						  MethodFlag mflag = ALL,
						  SimpleRuntimeContext* C = getThreadContext()) {
    assert(src.ID);
    assert(dst.ID);

    //yes, fault on null (no edge)
    if (shouldLock(mflag))
      SimpleRuntimeContext::acquire(C, src.ID);

    if (Directional) {
      return src.ID->getEdgeData(dst.ID);
    } else {
      if (shouldLock(mflag))
	SimpleRuntimeContext::acquire(C, dst.ID);
      if (src < dst)
	return src.ID->getEdgeData(dst.ID);
      else
	return dst.ID->getEdgeData(src.ID);
    }
  }

  // General Things

  int neighborsSize(GraphNode N, MethodFlag mflag = ALL, SimpleRuntimeContext* C = getThreadContext()) {
    assert(N.ID);
    if (shouldLock(mflag))
      SimpleRuntimeContext::acquire(C, N.ID);
    return N.ID->edges.size();
  }

  typedef typename boost::transform_iterator<makeGraphNodePtr,
					     typename gNode::neighbor_iterator> neighbor_iterator;

  neighbor_iterator neighbor_begin(GraphNode N, MethodFlag mflag = ALL, SimpleRuntimeContext* C = getThreadContext()) {
    assert(N.ID);
    if (shouldLock(mflag))
      SimpleRuntimeContext::acquire(C, N.ID);
    for (typename gNode::neighbor_iterator ii = N.ID->neighbor_begin(), ee =
	   N.ID->neighbor_end(); ii != ee; ++ii) {
      __builtin_prefetch(*ii);
      if (!Directional && shouldLock(mflag))
	SimpleRuntimeContext::acquire(C, *ii);
    }
    return boost::make_transform_iterator(N.ID->neighbor_begin(),
					  makeGraphNodePtr(this));
  }
  neighbor_iterator neighbor_end(GraphNode N, MethodFlag mflag = ALL, SimpleRuntimeContext* C = getThreadContext()) {
    assert(N.ID);
    if (shouldLock(mflag)) // Probably not necessary (no valid use for an end pointer should ever require it)
      SimpleRuntimeContext::acquire(C, N.ID);
    return boost::make_transform_iterator(N.ID->neighbor_end(),
					  makeGraphNodePtr(this));
  }

  //These are not thread safe!!
  typedef boost::transform_iterator<makeGraphNode, boost::filter_iterator<
						     std::mem_fun_ref_t<bool, gNode>, typename nodeListTy::iterator> >
  active_iterator;

  active_iterator active_begin() {
    return boost::make_transform_iterator(boost::make_filter_iterator(
								      std::mem_fun_ref(&gNode::isActive), nodes.begin(), nodes.end()),
					  makeGraphNode(this));
  }

  active_iterator active_end() {
    return boost::make_transform_iterator(boost::make_filter_iterator(
								      std::mem_fun_ref(&gNode::isActive), nodes.end(), nodes.end()),
					  makeGraphNode(this));
  }
  // The number of nodes in the graph
  unsigned int size() {
    return std::distance(active_begin(), active_end());
  }

  FirstGraph() {
    std::cout << "STAT: NodeSize " << (int) sizeof(gNode) << "\n";
  }

};

}
}
