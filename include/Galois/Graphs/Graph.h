// simple graph -*- C++ -*-

#include <list>
#include <map>
#include <vector>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/filter_iterator.hpp>


#include "Galois/Runtime/Context.h"
#include "Galois/Runtime/InsBag.h"
#include "Support/ThreadSafe/TSIBag.h"
#include "LLVM/SmallVector.h"

////////////////////////////////////////////////////////////////////////////////
// Wrap void so that we can have a valid type on void nodes

template<typename T>
struct VoidWrapper {
  typedef T Type;
};

template<>
struct VoidWrapper<void> {
  struct unit {};
  typedef unit Type;
};

////////////////////////////////////////////////////////////////////////////////

template<typename _NodeTy, typename _EdgeTy>
class FirstGraph {
  typedef typename VoidWrapper<_NodeTy>::Type NodeTy;
  typedef typename VoidWrapper<_EdgeTy>::Type EdgeTy;
  

  struct gNode : public GaloisRuntime::Lockable {
    NodeTy data;
    bool active;
    typedef llvm::SmallVector<std::pair<gNode*, EdgeTy>, 4> edgesTy;
    //typedef std::vector<std::pair<gNode*, EdgeTy> > edgesTy;
    //, __gnu_cxx::malloc_allocator<std::pair<gNode*, EdgeTy> > > edgesTy;
    edgesTy edges;

    gNode(const NodeTy& d, bool a)
      :data(d), active(a)
    {}

    void eraseEdge(gNode* N) {
      for (typename edgesTy::iterator ii = edges.begin(), ee = edges.end(); ii != ee; ++ii) {
	if (ii->first == N) {
	  edges.erase(ii);
	  return;
	}
      }
    }

    EdgeTy& getOrCreateEdge(gNode* N) {
      for (typename edgesTy::iterator ii = edges.begin(), ee = edges.end(); ii != ee; ++ii) 
	if (ii->first == N) 
	  return ii->second;
      edges.push_back(std::make_pair(N, EdgeTy()));
      return edges.back().second;
    }
    EdgeTy& getEdge(gNode* N) {
      for (typename edgesTy::iterator ii = edges.begin(), ee = edges.end(); ii != ee; ++ii) 
	if (ii->first == N)
	  return ii->second;
      assert(0 && "Edge doesn't exist");
      abort();
    }
  };
  
  //The graph manages the lifetimes of the data in the nodes and edges
  typedef GaloisRuntime::galois_insert_bag<gNode> nodeListTy;
  //typedef threadsafe::ts_insert_bag<gNode> nodeListTy;
  nodeListTy nodes;
  
  //deal with the Node redirction
  NodeTy& getData(gNode* ID) {
    assert(ID);
    GaloisRuntime::acquire(ID);
    return ID->data;
  }
  
public:
  
  class GraphNode {
    friend class FirstGraph;
    friend class makeGraphNode;
    friend class makeGraphNode2;
    FirstGraph* Parent;
    gNode* ID;
    GraphNode(FirstGraph* p, gNode* id)
      :Parent(p), ID(id)
    {}
  public:
    //public constructor is the Null pointer
    GraphNode()
      :Parent(0), ID(0)
    {}

    NodeTy& getData() {
      return Parent->getData(ID);
    }

    bool isNull() const {
      return !Parent;
    }

    bool operator!= (const GraphNode& rhs) const {
      return Parent != rhs.Parent || ID != rhs.ID;
    }

    bool operator== (const GraphNode& rhs) const {
      return Parent == rhs.Parent && ID == rhs.ID;
    }

    bool operator< (const GraphNode& rhs) const {
      return Parent < rhs.Parent || (Parent == rhs.Parent && ID < rhs.ID);
    }

    bool operator> (const GraphNode& rhs) const {
      return Parent > rhs.Parent || (Parent == rhs.Parent && ID >rhs.ID);
    }

  };

private:
  // Helpers for the iterator classes
  class makeGraphNode : public std::unary_function<std::pair<gNode*, EdgeTy>, GraphNode >{
    FirstGraph* G;
  public:
    makeGraphNode(FirstGraph* g) : G(g) {}
    GraphNode operator()(std::pair<gNode*, EdgeTy>& data) const {
      return GraphNode(G, data.first);
    }
  };
  class makeGraphNode2 : public std::unary_function<gNode, GraphNode >{
    FirstGraph* G;
  public:
    makeGraphNode2(FirstGraph* g) : G(g) {}
    GraphNode operator()(gNode& data) const {
      return GraphNode(G, &data);
    }
  };
  struct is_active_node : public std::unary_function<gNode, bool>{
    bool operator()(const gNode& g) const {
      return g.active;
    }
  };

public:

  // Node Handling

  // Creates a new node holding the indicated data.
  // Node is not added to the graph
  GraphNode createNode(const NodeTy& n) {
    gNode N(n,false);
    return GraphNode(this, &(nodes.push(N)));
  }

  // Adds a node to the graph.
  bool addNode(const GraphNode& n) {
    assert(n.ID);
    GaloisRuntime::acquire(n.ID);
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
  bool removeNode(GraphNode n) {
    assert(n.ID);
    GaloisRuntime::acquire(n.ID);
    gNode* N = n.ID;
    bool wasActive = N->active;
    if (wasActive) {
      //__sync_sub_and_fetch(&numActive, 1);
      N->active = false;
      //erase the in-edges first
      for (int i = 0; i < N->edges.size(); ++i) {
	if (N->edges[i].first != N) // don't handle loops yet
	  N->edges[i].first->eraseEdge(N);
      }
      N->edges.clear();
    }
    return wasActive;
  }

  // Edge Handling

  // Adds an edge to the graph containing the specified data.
  void addEdge(GraphNode src, GraphNode dst, const EdgeTy& data) {
    assert(src.ID);
    assert(dst.ID);
    GaloisRuntime::acquire(src.ID);
    GaloisRuntime::acquire(dst.ID);
    EdgeTy& E1 = src.ID->getOrCreateEdge(dst.ID);
    EdgeTy& E2 = dst.ID->getOrCreateEdge(src.ID);
    if (src < dst)
      E1 = data;
    else
      E2 = data;
  }

  
  void removeEdge(GraphNode src, GraphNode dst) {
    assert(src.ID);
    assert(dst.ID);
    GaloisRuntime::acquire(src.ID);
    GaloisRuntime::acquire(dst.ID);
    src.ID->eraseEdge(dst.ID);
    dst.ID->eraseEdge(src.ID);
  }

  EdgeTy& getEdgeData(GraphNode src, GraphNode dst) {
    assert(src.ID);
    assert(dst.ID);

    //yes, fault on null (no edge)
    GaloisRuntime::acquire(src.ID);
    GaloisRuntime::acquire(dst.ID);

    if (src < dst)
      return src.ID->getEdge(dst.ID);
    else
      return dst.ID->getEdge(src.ID);
  }

  // General Things

  int neighborsSize(GraphNode N) {
    assert(N.ID);
    GaloisRuntime::acquire(N.ID);
    return N.ID->edges.size();
  }

  typedef boost::transform_iterator<makeGraphNode, typename gNode::edgesTy::iterator > neighbor_iterator;

  neighbor_iterator neighbor_begin(GraphNode N) {
    assert(N.ID);
    GaloisRuntime::acquire(N.ID);
    return boost::make_transform_iterator(N.ID->edges.begin(), makeGraphNode(this));
  }
  neighbor_iterator neighbor_end(GraphNode N) {
    assert(N.ID);
    GaloisRuntime::acquire(N.ID);
    return boost::make_transform_iterator(N.ID->edges.end(), makeGraphNode(this));
  }


  //These are not thread safe!!

  typedef boost::transform_iterator<makeGraphNode2, boost::filter_iterator<is_active_node, typename nodeListTy::iterator> >active_iterator;

  active_iterator active_begin() {
    return boost::make_transform_iterator(boost::make_filter_iterator<is_active_node>(nodes.begin(), nodes.end()), makeGraphNode2(this));
  }

  active_iterator active_end() {
    return boost::make_transform_iterator(boost::make_filter_iterator<is_active_node>(nodes.end(), nodes.end()), makeGraphNode2(this));
  }
  // The number of nodes in the graph
  int size() {
    return std::distance(active_begin(), active_end());
  }

};
