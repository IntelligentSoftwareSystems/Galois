// simple graph -*- C++ -*-

#include <list>
#include <map>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/filter_iterator.hpp>

////////////////////////////////////////////////////////////////////////////////
// Wrap void so that we can have a valid type on void nodes
////////////////////////////////////////////////////////////////////////////////
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
  
  struct gNode {
    NodeTy data;
    bool active;
    gNode(const NodeTy& d, bool a)
      :data(d), active(a)
    {}
  };
  
  struct gEdge {
    EdgeTy data;
    gNode* src;
    gNode* dst;
    gEdge(const EdgeTy& da, gNode* s, gNode* d)
      :data(da), src(s), dst(d)
    {}
  };
  
  
  //The graph manages the lifetimes of the data in the nodes and edges
  std::list<gNode> nodes;
  std::list<gEdge> edges;
  
  int numActive;
  
  std::map<gNode*, std::map<gNode*, gEdge*> >edgeMap;
  
  //deal with the Node redirction
  NodeTy& getData(gNode* ID) {
    assert(ID);
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
  class makeGraphNode : public std::unary_function<std::pair<gNode*, gEdge*>, GraphNode >{
    FirstGraph* G;
  public:
    makeGraphNode(FirstGraph* g) : G(g) {}
    GraphNode operator()(std::pair<gNode* const, gEdge*>& data) const {
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
    typename std::list<gNode>::iterator I = nodes.insert(nodes.begin(), N);
    return GraphNode(this, &*I);
  }

  // Adds a node to the graph.
  bool addNode(const GraphNode& n) {
    assert(n.ID);
    bool oldActive = n.ID->active;
    if (!oldActive) {
      n.ID->active = true;
      ++numActive;
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
    gNode* N = n.ID;
    bool wasActive = N->active;
    if (wasActive) {
      --numActive;
      N->active = false;
      //erase the in-edges first
      for (typename std::map<gNode*, gEdge*>::iterator ii = edgeMap[N].begin(), ee = edgeMap[N].end(); ii != ee; ++ii)
	if (ii->first != N) // don't handle loops yet
	  edgeMap[ii->first].erase(N);
      
      //erase the out-edges second (and loops)
      edgeMap.erase(N);
    }
    return wasActive;
  }

  // Edge Handling

  // Adds an edge to the graph containing the specified data.
  // returns true if there was no previous edge between the two nodes
  bool addEdge(GraphNode src, GraphNode dst, const EdgeTy& data) {
    assert(src.ID);
    assert(dst.ID);
    gEdge N(data, src.ID, dst.ID);
    typename std::list<gEdge>::iterator I = edges.insert(edges.begin(), N);
    gEdge*& rval = edgeMap[src.ID][dst.ID];
    bool retval = (rval == 0);
    rval = &*I;
    edgeMap[dst.ID][src.ID] = &*I;
    return retval;
  }

  
  bool removeEdge(GraphNode src, GraphNode dst) {
    assert(src.ID);
    assert(dst.ID);
    bool retval = edgeMap[src.ID][dst.ID];
    edgeMap[src.ID].erase(dst.ID);
    edgeMap[dst.ID].erase(src.ID);
    return retval;
  }

  EdgeTy& getEdgeData(GraphNode src, GraphNode dst) {
    //yes, fault on null (no edge)
    return edgeMap[src.ID][dst.ID]->data;
  }

  // General Things

  // The number of nodes in the graph
  int size() {
    return numActive;
  }

  int neighborsSize(GraphNode N) {
    assert(N.ID);
    return edgeMap[N.ID].size();
  }

  typedef boost::transform_iterator<makeGraphNode, typename std::map<gNode*, gEdge*>::iterator > neighbor_iterator;

  neighbor_iterator neighbor_begin(GraphNode N) {
    assert(N.ID);
    return boost::make_transform_iterator(edgeMap[N.ID].begin(), makeGraphNode(this));
  }
  neighbor_iterator neighbor_end(GraphNode N) {
    assert(N.ID);
    return boost::make_transform_iterator(edgeMap[N.ID].end(), makeGraphNode(this));
  }

  typedef boost::transform_iterator<makeGraphNode2, boost::filter_iterator<is_active_node, typename std::list<gNode>::iterator> >active_iterator;

  active_iterator active_begin() {
    return boost::make_transform_iterator(boost::make_filter_iterator<is_active_node>(nodes.begin(), nodes.end()), makeGraphNode2(this));
  }

  active_iterator active_end() {
    return boost::make_transform_iterator(boost::make_filter_iterator<is_active_node>(nodes.end(), nodes.end()), makeGraphNode2(this));
  }

};
