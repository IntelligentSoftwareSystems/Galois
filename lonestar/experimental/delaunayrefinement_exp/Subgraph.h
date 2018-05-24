#ifndef SUBGRAPH_H
#define SUBGRAPH_H

#include "Element.h"

#include "galois/Galois.h"
#include "galois/graphs/Graph3.h"
#include "galois/runtime/DistSupport.h"

#include <vector>
#include <algorithm>

// Graph nodes
typedef galois::graphs::ThirdGraph<Element,void,galois::graphs::EdgeDirection::Un> Graph;
typedef Graph::NodeHandle GNode;
typedef typename Graph::pointer Graphp;

// Worklist Graph nodes
typedef galois::graphs::ThirdGraph<GNode,void,galois::graphs::EdgeDirection::Un> WLGraph;
typedef WLGraph::NodeHandle WLGNode;
typedef galois::runtime::gptr<WLGraph> WLGraphp;

struct EdgeTuple {
  GNode src;
  GNode dst;
  Edge data;
  EdgeTuple(GNode s, GNode d, const Edge& _d):src(s), dst(d), data(_d) {}

  bool operator==(const EdgeTuple& rhs) const {
    return src == rhs.src && dst == rhs.dst && data == data;
  }
};

/**
 *  A sub-graph of the mesh. Used to store information about the original 
 *  cavity  
 */
class PreGraph {
  typedef std::vector<GNode,galois::PerIterAllocTy::rebind<GNode>::other> NodesTy;
  NodesTy nodes;

public:
  typedef NodesTy::iterator iterator;

  explicit PreGraph(galois::PerIterAllocTy& cnx): nodes(cnx) {}

  bool containsNode(GNode N) {
    return std::find(nodes.begin(), nodes.end(), N) != nodes.end();
  }

  void addNode(GNode n) { return nodes.push_back(n); }
  void reset() { nodes.clear(); }
  iterator begin() { return nodes.begin(); }
  iterator end() { return nodes.end(); }
};

/**
 *  A sub-graph of the mesh. Used to store information about the original 
 *  and updated cavity  
 */
class PostGraph {
  struct TempEdge {
    size_t src;
    GNode dst;
    Edge edge;
    TempEdge(size_t s, GNode d, const Edge& e): src(s), dst(d), edge(e) { }
  };

  typedef std::vector<GNode,galois::PerIterAllocTy::rebind<GNode>::other> NodesTy;
  typedef std::vector<EdgeTuple,galois::PerIterAllocTy::rebind<EdgeTuple>::other> EdgesTy;

  //! the nodes in the graph before updating
  NodesTy nodes;
  //! the edges that connect the subgraph to the rest of the graph
  EdgesTy edges;

public:
  typedef NodesTy::iterator iterator;
  typedef EdgesTy::iterator edge_iterator;

  explicit PostGraph(galois::PerIterAllocTy& cnx): nodes(cnx), edges(cnx) { }

  void addNode(GNode n) {
    nodes.push_back(n);
  }
 
  void addEdge(GNode src, GNode dst, const Edge& e) {
    edges.push_back(EdgeTuple(src, dst, e));
  }

  void reset() {
    nodes.clear();
    edges.clear();
  }

  iterator begin() { return nodes.begin(); }
  iterator end() { return nodes.end(); }
  edge_iterator edge_begin() { return edges.begin(); }
  edge_iterator edge_end() { return edges.end(); }
};

#endif
