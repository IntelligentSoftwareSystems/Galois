/** Local Computation graphs -*- C++ -*-
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
 * There are two main classes, ::FileGraph and ::LC_FileGraph. The former
 * represents the pure structure of a graph (i.e., whether an edge exists between
 * two nodes) and cannot be modified. The latter allows values to be stored on
 * nodes and edges, but the structure of the graph cannot be modified.
 *
 * An example of use:
 * 
 * \code
 * typedef Galois::Graph::LC_FileGraph<int,int> Graph;
 * 
 * // Create graph
 * Graph g;
 * g.structureFromFile(inputfile);
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
 *     int nodeData = g.getData(dst);
 *   }
 * }
 * \endcode
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#include "Galois/Graphs/FileGraph.h"
#include "Galois/Runtime/mem.h"

#include <iterator>
#include <new>

using namespace GaloisRuntime;

namespace Galois {
namespace Graph {

//! Local computation graph (i.e., graph structure does not change)
template<typename NodeTy, typename EdgeTy>
class LC_CRS_Graph {
protected:
  struct NodeInfo : public GaloisRuntime::Lockable {
    NodeTy data;
  };

  NodeInfo* NodeData;
  uint64_t* EdgeIndData;
  EdgeTy* EdgeData;
  uint32_t* EdgeDst;

  uint64_t numNodes;

  uint64_t raw_neighbor_begin(uint32_t N) const {
    return (N == 0) ? 0 : EdgeIndData[N-1];
  }

  uint64_t raw_neighbor_end(uint32_t N) const {
    return EdgeIndData[N];
  }

  uint64_t getEdgeIdx(uint32_t src, uint32_t dst) {
    for (uint64_t ii = raw_neighbor_begin(src),
	   ee = raw_neighbor_end(src); ii != ee; ++ii)
      if (EdgeDst[ii] == dst)
	return ii;
    return ~static_cast<uint64_t>(0);
  }

public:
  typedef uint32_t GraphNode;
  typedef boost::counting_iterator<uint64_t> edge_iterator;
  typedef boost::counting_iterator<uint32_t> active_iterator;

  LC_CRS_Graph() {}
  ~LC_CRS_Graph() {}

  NodeTy& getData(GraphNode N, MethodFlag mflag = ALL) {
    NodeInfo& NI = NodeData[N];
    GaloisRuntime::acquire(&NI, mflag);
    return NI.data;
  }

  bool hasNeighbor(GraphNode src, GraphNode dst, MethodFlag mflag = ALL) {
    return getEdgeIdx(src,dst) != ~static_cast<uint64_t>(0);
  }

  EdgeTy& getEdgeData(GraphNode src, GraphNode dst, MethodFlag mflag = ALL) {
    GaloisRuntime::acquire(&NodeData[src], mflag);
    return EdgeData[getEdgeIdx(src,dst)];
  }

  EdgeTy& getEdgeData(edge_iterator ni) {
    return EdgeData[*ni];
  }

  GraphNode getEdgeDst(edge_iterator ni) {
    return EdgeDst[*ni];
  }

  uint64_t size() const {
    return numNodes;
  }

  active_iterator active_begin() const {
    return active_iterator(0);
  }

  active_iterator active_end() const {
    return active_iterator(numNodes);
  }

  edge_iterator edge_begin(GraphNode N, MethodFlag mflag = ALL) {
    NodeInfo& NI = NodeData[N];
    GaloisRuntime::acquire(&NI, mflag);
    return edge_iterator(raw_neighbor_begin(N));
  }

  edge_iterator edge_end(GraphNode N, MethodFlag mflag = ALL) {
    NodeInfo& NI = NodeData[N];
    GaloisRuntime::acquire(&NI, mflag);
    return edge_iterator(raw_neighbor_end(N));
  }

  void structureFromFile(const std::string& fname) {
    FileGraph graph;
    graph.structureFromFile(fname);
    numNodes = graph.size();
    NodeData = reinterpret_cast<NodeInfo*>(GaloisRuntime::MM::largeAlloc(sizeof(NodeInfo) * numNodes));
    EdgeIndData = reinterpret_cast<uint64_t*>(GaloisRuntime::MM::largeAlloc(sizeof(uint64_t) * numNodes));
    EdgeData = reinterpret_cast<EdgeTy*>(GaloisRuntime::MM::largeAlloc(sizeof(EdgeTy) * graph.sizeEdges()));
    EdgeDst = reinterpret_cast<uint32_t*>(GaloisRuntime::MM::largeAlloc(sizeof(uint32_t) * graph.sizeEdges()));
    std::copy(graph.edgeid_begin(), graph.edgeid_end(), &EdgeIndData[0]);
    std::copy(graph.nodeid_begin(), graph.nodeid_end(), &EdgeDst[0]);
    std::copy(graph.edgedata_begin<EdgeTy>(), graph.edgedata_end<EdgeTy>(),
	      &EdgeData[0]);
    for (unsigned x = 0; x < numNodes; ++x)
      new (&NodeData[x]) NodeTy; // inplace new
  }
};


//! Local computation graph (i.e., graph structure does not change)
template<typename NodeTy, typename EdgeTy>
class LC_CRSInline_Graph {
protected:
  struct EdgeInfo;
  struct NodeInfo : public GaloisRuntime::Lockable {
    NodeTy data;
    EdgeInfo* edgebegin;
    EdgeInfo* edgeend;
  };
  struct EdgeInfo {
    EdgeTy data;
    NodeInfo* dst;
  };

  NodeInfo* NodeData;
  EdgeInfo* EdgeData;
  uint64_t numNodes;
  NodeInfo* endNode;

  uint64_t getEdgeIdx(uint64_t src, uint64_t dst) {
    NodeInfo& NI = NodeData[src];
    for (uint64_t x = NI.edgebegin; x < NI.edgeend; ++x)
      if (EdgeData[x].dst == dst)
	return x;
    return ~static_cast<uint64_t>(0);
  }

public:
  typedef NodeInfo* GraphNode;
  typedef EdgeInfo* edge_iterator;
  class active_iterator : std::iterator<std::random_access_iterator_tag, GraphNode> {
    NodeInfo* at;

  public:
    active_iterator(NodeInfo* a) :at(a) {}
    active_iterator(const active_iterator& m) :at(m.at) {}
    active_iterator& operator++() { ++at; return *this; }
    active_iterator operator++(int) { active_iterator tmp(*this); ++at; return tmp; }
    active_iterator& operator--() { --at; return *this; }
    active_iterator operator--(int) { active_iterator tmp(*this); --at; return tmp; }
    bool operator==(const active_iterator& rhs) { return at == rhs.at; }
    bool operator!=(const active_iterator& rhs) { return at != rhs.at; }
    GraphNode operator*() { return at; }
  };

  LC_CRSInline_Graph() {}
  ~LC_CRSInline_Graph() {}

  NodeTy& getData(GraphNode N, MethodFlag mflag = ALL) {
    GaloisRuntime::acquire(N, mflag);
    return N->data;
  }
  
  EdgeTy& getEdgeData(GraphNode src, GraphNode dst, MethodFlag mflag = ALL) {
    GaloisRuntime::acquire(src, mflag);
    return EdgeData[getEdgeIdx(src,dst)].data;
  }

  EdgeTy& getEdgeData(edge_iterator ni) const {
    return ni->data;
   }

  GraphNode getEdgeDst(edge_iterator ni) const {
    return ni->dst;
  }

  uint64_t size() const {
    return numNodes;
  }

  active_iterator active_begin() const {
    return active_iterator(&NodeData[0]);
  }

  active_iterator active_end() const {
    return active_iterator(endNode);
  }

  edge_iterator edge_begin(GraphNode N, MethodFlag mflag = ALL) {
    GaloisRuntime::acquire(N, mflag);
    return N->edgebegin;
  }

  edge_iterator edge_end(GraphNode N, MethodFlag mflag = ALL) {
    GaloisRuntime::acquire(N, mflag);
    return N->edgeend;
  }

  void structureFromFile(const std::string& fname) {
    FileGraph graph;
    graph.structureFromFile(fname);
    numNodes = graph.size();
    NodeData.allocate(numNodes);
    EdgeData.allocate(graph.sizeEdges());
    std::vector<NodeInfo*> node_ids;
    node_ids.resize(numNodes);
    for (FileGraph::active_iterator ii = graph.active_begin(),
	   ee = graph.active_end(); ii != ee; ++ii) {
      NodeInfo* curNode = &NodeData[*ii];
      new (&curNode->data) NodeTy; //inplace new
      node_ids[*ii] = curNode;
    }
    endNode = &NodeData[numNodes];

    //layout the edges
    EdgeInfo* curEdge = &EdgeData[0];
    for (FileGraph::active_iterator ii = graph.active_begin(),
	   ee = graph.active_end(); ii != ee; ++ii) {
      node_ids[*ii]->edgebegin = curEdge;
      for (FileGraph::neighbor_iterator ni = graph.neighbor_begin(*ii),
	     ne = graph.neighbor_end(*ii); ni != ne; ++ni) {
	new (&curEdge->data) EdgeTy(graph.getEdgeData<EdgeTy>(ni)); //inplace new
	curEdge->dst = node_ids[*ni];
	++curEdge;
      }
      node_ids[*ii]->edgeend = curEdge;
    }
  }
};


//! Local computation graph (i.e., graph structure does not change)
template<typename NodeTy, typename EdgeTy>
class LC_Linear_Graph {
protected:
  struct NodeInfo;
  struct EdgeInfo {
    EdgeTy data;
    NodeInfo* dst;
  };
  struct NodeInfo : public GaloisRuntime::Lockable {
    NodeTy data;
    int numEdges;
    EdgeInfo* edgeBegin() {
      NodeInfo* n = this;
      ++n; //start of edges
      return reinterpret_cast<EdgeInfo*>(n);
    }
    EdgeInfo* edgeEnd() {
      EdgeInfo* ei = edgeBegin();
      ei += numEdges;
      return ei;
    }
    NodeInfo* next() {
      NodeInfo* ni = this;
      EdgeInfo* ei = edgeEnd();
      while (reinterpret_cast<char*>(ni) < reinterpret_cast<char*>(ei))
	++ni;
      return ni;
    }
  };

  void* Data;
  NodeInfo* endNode;
  uint64_t numNodes;

  EdgeInfo* getEdgeIdx(NodeInfo* src, NodeInfo* dst) {
    EdgeInfo* eb = src->edgeBegin();
    EdgeInfo* ee = src->edgeEnd();
    for (; eb != ee; ++eb)
      if (eb->dst == dst)
	return eb;
    return 0;
  }

public:
  typedef NodeInfo* GraphNode;
  typedef EdgeInfo* edge_iterator;
  class active_iterator : std::iterator<std::forward_iterator_tag, GraphNode> {
    NodeInfo* at;
    void incA() {
      at = at->next();
    }

  public:
    active_iterator(NodeInfo* a) :at(a) {}
    active_iterator(const active_iterator& m) :at(m.at) {}
    active_iterator& operator++() { incA(); return *this; }
    active_iterator operator++(int) { active_iterator tmp(*this); incA(); return tmp; }
    bool operator==(const active_iterator& rhs) { return at == rhs.at; }
    bool operator!=(const active_iterator& rhs) { return at != rhs.at; }
    GraphNode operator*() { return at; }
  };

  LC_Linear_Graph() {}
  ~LC_Linear_Graph() {}

  NodeTy& getData(GraphNode N, MethodFlag mflag = ALL) {
    GaloisRuntime::acquire(N, mflag);
    return N->data;
  }
  
  EdgeTy& getEdgeData(GraphNode src, GraphNode dst, MethodFlag mflag = ALL) {
    GaloisRuntime::acquire(src, mflag);
    return getEdgeIdx(src,dst)->data;
  }

  EdgeTy& getEdgeData(edge_iterator ni) const {
    return ni->data;
  }

  GraphNode getEdgeDst(edge_iterator ni) const {
    return ni->dst;
  }

  uint64_t size() const {
    return numNodes;
  }

  active_iterator active_begin() const {
    return active_iterator(reinterpret_cast<NodeInfo*>(Data));
  }

  active_iterator active_end() const {
    return active_iterator(endNode);
  }

  edge_iterator edge_begin(GraphNode N, MethodFlag mflag = ALL) {
    GaloisRuntime::acquire(N, mflag);
    // EdgeInfo* eb = N->edgeBegin();
    // EdgeInfo* ee = N->edgeEnd();
    // for (; eb != ee; ++eb)
    //   __builtin_prefetch(eb->dst);
    return N->edgeBegin();
  }

  edge_iterator edge_end(GraphNode N, MethodFlag mflag = ALL) {
    GaloisRuntime::acquire(N, mflag);
    return N->edgeEnd();
  }

  void structureFromFile(const std::string& fname) {
    FileGraph graph;
    graph.structureFromFile(fname);
    numNodes = graph.size();
    Data = GaloisRuntime::MM::largeAlloc(numNodes * 2 * sizeof(NodeInfo) +
					 graph.sizeEdges() * sizeof(EdgeInfo));
    std::vector<NodeInfo*> node_ids;
    node_ids.resize(numNodes);
    NodeInfo* curNode = reinterpret_cast<NodeInfo*>(Data);
    for (FileGraph::active_iterator ii = graph.active_begin(),
	   ee = graph.active_end(); ii != ee; ++ii) {
      new (&curNode->data) NodeTy; //inplace new
      curNode->numEdges = graph.neighborsSize(*ii);
      node_ids[*ii] = curNode;
      curNode = curNode->next();
    }
    endNode = curNode;

    //layout the edges
    for (FileGraph::active_iterator ii = graph.active_begin(),
	   ee = graph.active_end(); ii != ee; ++ii) {
      EdgeInfo* edge = node_ids[*ii]->edgeBegin();
      for (FileGraph::neighbor_iterator ni = graph.neighbor_begin(*ii),
	     ne = graph.neighbor_end(*ii); ni != ne; ++ni) {
	new (&edge->data) EdgeTy(graph.getEdgeData<EdgeTy>(ni)); //inplace new
	edge->dst = node_ids[*ni];
	++edge;
      }
    }
  }
};


}
}
