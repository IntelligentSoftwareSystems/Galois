/** Local Computation graphs -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
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
 * @author Andrew Lenharth <andrewl@lenharth.org>
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#ifndef GALOIS_GRAPH_LC_LINEAR_GRAPH_H
#define GALOIS_GRAPH_LC_LINEAR_GRAPH_H

#include "Galois/LargeArray.h"
#include "Galois/Graph/FileGraph.h"
#include "Galois/Graph/Util.h"
#include "Galois/Runtime/MethodFlags.h"

#include <boost/utility/enable_if.hpp>

namespace Galois {
namespace Graph {

namespace Linear_InOutGraphImpl {

//! Dispatch type to handle implementation of in edges
template<typename Graph, bool HasPointerData>
class InEdges;

template<typename Graph>
class InEdges<Graph,false> {
  typedef typename Graph::NodeInfo NodeInfo;
  typedef typename Graph::Nodes Nodes;
  typedef typename Graph::EdgeInfo EdgeInfo;

  Graph graph;
  Nodes* nodes;

public:
  EdgeInfo* raw_begin(NodeInfo* n) const {
    return (*nodes)[n->getId()]->edgeBegin();
  }

  EdgeInfo* raw_end(NodeInfo* n) const {
    return (*nodes)[n->getId()]->edgeEnd();
  }

  EdgeInfo* edge_sort_begin(NodeInfo* n) {
    return raw_begin(n);
  }

  EdgeInfo* edge_sort_end(NodeInfo* n) {
    return raw_end(n);
  }

  typename EdgeInfo::reference getEdgeData(EdgeInfo* e) {
    return e->get();
  }

  NodeInfo* getEdgeDst(EdgeInfo* e) {
    return (*nodes)[e->dst->getId()];
  }

  void initialize(Graph* g, bool symmetric) {
    if (symmetric) {
      this->nodes = &g->nodes;
      return;
    }

    // TODO: Transpose graph
    abort();
  }

  void initialize(FileGraph& transpose) {
    graph.structureFromGraph(transpose);
    nodes = &graph.nodes;
  }
};

template<typename Graph>
class InEdges<Graph,true> {
  // TODO: implement for when we don't store copies of edges
};

} // end namespace

//! Local computation graph (i.e., graph structure does not change)
template<typename NodeTy, typename EdgeTy, typename _IntrusiveId=void>
class LC_Linear_Graph: boost::noncopyable {
  template<typename Graph,bool HasPointerData>
    friend class Linear_InOutGraphImpl::InEdges;
protected:
  struct NodeInfo;
  typedef EdgeInfoBase<NodeInfo*,EdgeTy> EdgeInfo;
  typedef LargeArray<NodeInfo*,true> Nodes;

  struct NodeInfo : public NodeInfoBase<NodeTy>, public IntrusiveId<_IntrusiveId> {
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
 
  LargeArray<char,true> data;
  uint64_t numNodes;
  uint64_t numEdges;
  Nodes nodes;

public:
  typedef NodeInfo* GraphNode;
  typedef EdgeTy edge_data_type;
  typedef NodeTy node_data_type;
  typedef typename NodeInfo::reference node_data_reference;
  typedef typename EdgeInfo::reference edge_data_reference;
  typedef EdgeInfo* edge_iterator;
  typedef NodeInfo** iterator;
  typedef NodeInfo*const * const_iterator;
  typedef iterator local_iterator;
  typedef const_iterator const_local_iterator;

  LC_Linear_Graph() { }

  ~LC_Linear_Graph() { 
    for (typename Nodes::iterator ii = nodes.begin(), ei = nodes.end(); ii != ei; ++ii) {
      NodeInfo* n = *ii;
      EdgeInfo* edgeBegin = n->edgeBegin();
      EdgeInfo* edgeEnd = n->edgeEnd();

      n->destruct();
      if (EdgeInfo::has_value) {
        while (edgeBegin != edgeEnd) {
          edgeBegin->destroy();
          ++edgeBegin;
        }
      }
    }
  }

  node_data_reference getData(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    Galois::Runtime::checkWrite(mflag, false);
    Galois::Runtime::acquire(N, mflag);
    return N->getData();
  }
  
  edge_data_reference getEdgeData(edge_iterator ni, MethodFlag mflag = MethodFlag::NONE) const {
    Galois::Runtime::checkWrite(mflag, false);
    return ni->get();
  }

  GraphNode getEdgeDst(edge_iterator ni) const {
    return ni->dst;
  }

  uint64_t size() const { return numNodes; }
  uint64_t sizeEdges() const { return numEdges; }
  iterator begin() { return &nodes[0]; }
  iterator end() { return &nodes[numNodes]; }
  const_iterator begin() const { return &nodes[0]; }
  const_iterator end() const { return &nodes[numNodes]; }
  local_iterator local_begin() { return &nodes[localStart(numNodes)]; }
  local_iterator local_end() { return &nodes[localEnd(numNodes)]; }
  const_local_iterator local_begin() const { return &nodes[localStart(numNodes)]; }
  const_local_iterator local_end() const { return &nodes[localEnd(numNodes)]; }

  edge_iterator edge_begin(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    Galois::Runtime::acquire(N, mflag);
    if (Galois::Runtime::shouldLock(mflag)) {
      for (edge_iterator ii = N->edgeBegin(), ee = N->edgeEnd(); ii != ee; ++ii) {
        Galois::Runtime::acquire(ii->dst, mflag);
      }
    }
    return N->edgeBegin();
  }

  edge_iterator edge_end(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    Galois::Runtime::acquire(N, mflag);
    return N->edgeEnd();
  }

  EdgesIterator<LC_Linear_Graph> out_edges(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    return EdgesIterator<LC_Linear_Graph>(*this, N, mflag);
  }

  /**
   * Sorts outgoing edges of a node. Comparison function is over EdgeTy.
   */
  template<typename CompTy>
  void sortEdgesByEdgeData(GraphNode N, const CompTy& comp = std::less<EdgeTy>(), MethodFlag mflag = MethodFlag::ALL) {
    Galois::Runtime::acquire(N, mflag);
    std::sort(N->edgeBegin(), N->edgeEnd(), EdgeSortCompWrapper<EdgeInfo,CompTy>(comp));
  }

  /**
   * Sorts outgoing edges of a node. Comparison function is over <code>EdgeSortValue<EdgeTy></code>.
   */
  template<typename CompTy>
  void sortEdges(GraphNode N, const CompTy& comp, MethodFlag mflag = MethodFlag::ALL) {
    Galois::Runtime::acquire(N, mflag);
    std::sort(N->edgeBegin(), N->edgeEnd(), comp);
  }

  void structureFromFile(const std::string& fname) { Graph::structureFromFile(*this, fname); }

  void structureFromGraph(FileGraph& graph) {
    typedef typename EdgeInfo::value_type EDV;

    numNodes = graph.size();
    numEdges = graph.sizeEdges();
    data.allocate(sizeof(NodeInfo) * numNodes * 2 + sizeof(EdgeInfo) * numEdges);
    nodes.allocate(numNodes);
    NodeInfo* curNode = reinterpret_cast<NodeInfo*>(data.data());
    size_t id = 0;
    for (FileGraph::iterator ii = graph.begin(), ee = graph.end(); ii != ee; ++ii, ++id) {
      curNode->construct();
      curNode->setId(id);
      curNode->numEdges = std::distance(graph.neighbor_begin(*ii), graph.neighbor_end(*ii));
      nodes[*ii] = curNode;
      curNode = curNode->next();
    }

    //layout the edges
    for (FileGraph::iterator ii = graph.begin(), ee = graph.end(); ii != ee; ++ii) {
      EdgeInfo* edge = nodes[*ii]->edgeBegin();
      for (FileGraph::neighbor_iterator ni = graph.neighbor_begin(*ii),
          ne = graph.neighbor_end(*ii); ni != ne; ++ni) {
        if (EdgeInfo::has_value)
          edge->construct(graph.getEdgeData<EDV>(ni));
        edge->dst = nodes[*ni];
        ++edge;
      }
    }
  }
};

/**
 * Local computation graph (i.e., graph structure does not change) that
 * supports in and out edges.
 *
 * An additional template parameter specifies whether the in edge stores a
 * reference to the corresponding out edge data (the default) or a copy of the
 * corresponding out edge data. If you want to populate this graph from a
 * FileGraph that is already symmetric (i.e., (u,v) \in E ==> (v,u) \in E),
 * this parameter should be true.
 *
 * @tparam NodeTy data on nodes
 * @tparam EdgeTy data on out edges
 * @tparam CopyInEdgeData in edges hold a copy of out edge data
 */
template<typename NodeTy, typename EdgeTy, bool CopyInEdgeData = false>
class LC_Linear_InOutGraph: public LC_Linear_Graph<NodeTy,EdgeTy,uint32_t> {
  typedef LC_Linear_Graph<NodeTy,EdgeTy,uint32_t> Super;

protected:
  Linear_InOutGraphImpl::InEdges<Super,!CopyInEdgeData> inEdges;

public:
  typedef typename Super::GraphNode GraphNode;
  typedef typename Super::edge_data_type edge_data_type;
  typedef typename Super::node_data_type node_data_type;
  typedef typename Super::edge_data_reference edge_data_reference;
  typedef typename Super::node_data_reference node_data_reference;
  typedef typename Super::edge_iterator edge_iterator;
  typedef edge_iterator in_edge_iterator;
  typedef typename Super::iterator iterator;
  typedef typename Super::const_iterator const_iterator;
  typedef typename Super::local_iterator local_iterator;
  typedef typename Super::const_local_iterator const_local_iterator;

  edge_data_reference getInEdgeData(in_edge_iterator ni, MethodFlag mflag = MethodFlag::NONE) { 
    Galois::Runtime::checkWrite(mflag, false);
    return inEdges.getEdgeData(ni);
  }

  GraphNode getInEdgeDst(in_edge_iterator ni) {
    return inEdges.getEdgeDst(ni);
  }

  in_edge_iterator in_edge_begin(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    Galois::Runtime::acquire(N, mflag);
    if (Galois::Runtime::shouldLock(mflag)) {
      for (edge_iterator ii = inEdges.raw_begin(N), ee = inEdges.raw_end(N); ii != ee; ++ii) {
        Galois::Runtime::acquire(ii->dst, mflag);
      }
    }
    return in_edge_iterator(inEdges.raw_begin(N));
  }

  in_edge_iterator in_edge_end(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    Galois::Runtime::acquire(N, mflag);
    return in_edge_iterator(inEdges.raw_end(N));
  }

  InEdgesIterator<LC_Linear_InOutGraph> in_edges(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    return InEdgesIterator<LC_Linear_InOutGraph>(*this, N, mflag);
  }

  /**
   * Sorts incoming edges of a node. Comparison function is over EdgeTy.
   */
  template<typename CompTy>
  void sortInEdgesByEdgeData(GraphNode N, const CompTy& comp = std::less<EdgeTy>(), MethodFlag mflag = MethodFlag::ALL) {
    Galois::Runtime::acquire(N, mflag);
    std::sort(inEdges.edge_sort_begin(N), inEdges.edge_sort_end(N), EdgeSortCompWrapper<EdgeSortValue<GraphNode,EdgeTy>,CompTy>(comp));
  }

  /**
   * Sorts incoming edges of a node. Comparison function is over <code>EdgeSortValue<EdgeTy></code>.
   */
  template<typename CompTy>
  void sortInEdges(GraphNode N, const CompTy& comp, MethodFlag mflag = MethodFlag::ALL) {
    Galois::Runtime::acquire(N, mflag);
    std::sort(inEdges.edge_sort_begin(N), inEdges.edge_sort_end(N), comp);
  }

  size_t idFromNode(GraphNode N) {
    return N->getId();
  }

  GraphNode nodeFromId(size_t N) {
    return this->nodes[N];
  }

  void structureFromFile(const std::string& fname, bool symmetric) { Graph::structureFromFile(*this, fname, symmetric); }

  void structureFromFile(const std::string& fname, const std::string& tname, typename boost::enable_if_c<CopyInEdgeData>::type* dummy = 0) {
    FileGraph inputGraph, transposeGraph;
    inputGraph.structureFromFile(fname);
    transposeGraph.structureFromFile(tname);
    structureFromGraph(inputGraph, transposeGraph);
  }

  void structureFromGraph(FileGraph& graph, FileGraph& transpose, typename boost::enable_if_c<CopyInEdgeData>::type* dummy = 0) {
    if (graph.size() != transpose.size()) {
      GALOIS_ERROR(true, "number of nodes in graph and its transpose do not match");
    } else if (graph.sizeEdges() != transpose.sizeEdges()) {
      GALOIS_ERROR(true, "number of edges in graph and its transpose do not match");
    }

    Super::structureFromGraph(graph);
    inEdges.initialize(transpose);
  }

  void structureFromGraph(FileGraph& graph, bool symmetric) {
    Super::structureFromGraph(graph);
    inEdges.initialize(this, symmetric);
  }
};

} // end namespace
} // end namespace

#endif
