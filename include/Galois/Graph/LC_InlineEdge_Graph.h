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
#ifndef GALOIS_GRAPH_LC_INLINEEDGE_GRAPH_H
#define GALOIS_GRAPH_LC_INLINEEDGE_GRAPH_H

#include "Galois/LargeArray.h"
#include "Galois/Graph/FileGraph.h"
#include "Galois/Graph/Util.h"
#include "Galois/Runtime/MethodFlags.h"

#include <boost/utility/enable_if.hpp>

namespace Galois {
namespace Graph {

namespace InlineEdge_InOutGraphImpl {

//! Dispatch type to handle implementation of in edges
template<typename Graph, bool HasPointerData>
class InEdges;

template<typename Graph>
class InEdges<Graph,false> {
  typedef typename Graph::EdgeInfo EdgeInfo;
  typedef typename Graph::NodeInfo NodeInfo;
  Graph graphStore;
  Graph* graph;

public:
  EdgeInfo* raw_begin(size_t n) const {
    return graph->nodeData[n].edgeBegin();
  }

  EdgeInfo* raw_end(size_t n) const {
    return graph->nodeData[n].edgeEnd();
  }

#if 0
  EdgeInfo* edge_sort_begin(NodeInfo* n) {
    return raw_begin(n);
  }

  EdgeInfo* edge_sort_end(NodeInfo* n) {
    return raw_end(n);
  }
#endif

  typename EdgeInfo::reference getEdgeData(EdgeInfo* e) {
    return e->get();
  }

  size_t getEdgeDst(EdgeInfo* n) {
    return std::distance(graph->nodeData.data(), graph->getEdgeDst(n));
  }

  void initialize(Graph* g, bool symmetric) {
    if (symmetric) {
      this->graph = g;
      return;
    }

    // TODO: Transpose graph
    abort();
  }

  void initialize(FileGraph& transpose) {
    graphStore.structureFromGraph(transpose);
    graph = &graphStore;
  }
};

template<typename Graph>
class InEdges<Graph,true> {
  // TODO: implement for when we don't store copies of edges
};

} // end namespace

/**
 * Local computation graph (i.e., graph structure does not change). The data representation
 * is a modification of {@link LC_CSR_Graph} where the edge data is stored inline with the
 * adjacency information.
 *
 * @tparam CompressNodePtr
 *  Compress representation of graph at the expense of one level of indirection on accessing
 *  neighbors of a node
 */
template<typename NodeTy, typename EdgeTy, bool CompressNodePtr=false>
class LC_InlineEdge_Graph: boost::noncopyable {
  template<typename Graph,bool HasPointerData>
    friend class InlineEdge_InOutGraphImpl::InEdges;
protected:
  struct NodeInfo;
  typedef EdgeInfoBase<typename boost::mpl::if_c<CompressNodePtr,uint32_t,NodeInfo*>::type,EdgeTy> EdgeInfo;
  typedef LargeArray<EdgeInfo> EdgeData;

  class NodeInfo: public NodeInfoBase<NodeTy,true> {
    EdgeInfo* m_edgeBegin;
    EdgeInfo* m_edgeEnd;
  public:
    EdgeInfo*& edgeBegin() { return m_edgeBegin; }
    EdgeInfo*& edgeEnd() { return m_edgeEnd; }
  };

  LargeArray<NodeInfo> nodeData;
  EdgeData edgeData;
  uint64_t numNodes;
  uint64_t numEdges;
  NodeInfo* endNode;

  template<bool Compressed>
  NodeInfo* getDst(EdgeInfo* ii, typename boost::enable_if_c<Compressed>::type* x = 0) const {
    return const_cast<NodeInfo*>(&nodeData[ii->dst]);
  }

  template<bool Compressed>
  NodeInfo* getDst(EdgeInfo* ii, typename boost::enable_if_c<!Compressed>::type* x = 0) const {
    return ii->dst;
  }

  template<bool Compressed,typename Container,typename Index>
  void setEdgeDst(Container& c, EdgeInfo* edge, Index idx, typename boost::enable_if_c<Compressed>::type* = 0) {
    edge->dst = idx;
  }

  template<bool Compressed,typename Container,typename Index>
  void setEdgeDst(Container& c, EdgeInfo* edge, Index idx, typename boost::enable_if_c<!Compressed>::type* = 0) {
    edge->dst = c[idx];
  }

public:
  typedef NodeInfo* GraphNode;
  typedef EdgeTy edge_data_type;
  typedef NodeTy node_data_type;
  typedef typename EdgeInfo::reference edge_data_reference;
  typedef typename NodeInfo::reference node_data_reference;
  typedef EdgeInfo* edge_iterator;
  typedef Galois::NoDerefIterator<NodeInfo*> iterator;
  typedef Galois::NoDerefIterator<const NodeInfo*> const_iterator;
  typedef iterator local_iterator;
  typedef const_iterator const_local_iterator;

  ~LC_InlineEdge_Graph() {
    if (!EdgeInfo::has_value) return;
    if (numNodes == 0) return;

    for (edge_iterator ii = nodeData[0].edgeBegin(), ei = endNode->edgeEnd(); ii != ei; ++ii) {
      ii->destroy();
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
    return getDst<CompressNodePtr>(ni);
  }

  uint64_t size() const { return numNodes; }
  uint64_t sizeEdges() const { return numEdges; }

  const_iterator begin() const { return const_iterator(nodeData.data()); }
  const_iterator end() const { return const_iterator(endNode); }
  iterator begin() { return iterator(nodeData.data()); }
  iterator end() { return iterator(endNode); }

  local_iterator local_begin() { return local_iterator(&nodeData[localStart(numNodes)]); }
  local_iterator local_end() { return local_iterator(&nodeData[localEnd(numNodes)]); }
  const_local_iterator local_begin() const { return const_local_iterator(&nodeData[localStart(numNodes)]); }
  const_local_iterator local_end() const { return const_local_iterator(&nodeData[localEnd(numNodes)]); }

  edge_iterator edge_begin(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    Galois::Runtime::acquire(N, mflag);
    if (Galois::Runtime::shouldLock(mflag)) {
      for (edge_iterator ii = N->edgeBegin(), ee = N->edgeEnd(); ii != ee; ++ii) {
        Galois::Runtime::acquire(getDst<CompressNodePtr>(ii), mflag);
      }
    }
    return N->edgeBegin();
  }

  edge_iterator edge_end(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    Galois::Runtime::acquire(N, mflag);
    return N->edgeEnd();
  }

  EdgesIterator<LC_InlineEdge_Graph> out_edges(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    return EdgesIterator<LC_InlineEdge_Graph>(*this, N, mflag);
  }

#if 0
  /**
   * Sorts outgoing edges of a node. Comparison function is over EdgeTy.
   */
  template<typename CompTy>
  void sortEdgesByEdgeData(GraphNode N, const CompTy& comp = std::less<EdgeTy>(), MethodFlag mflag = MethodFlag::ALL) {
    Galois::Runtime::acquire(N, mflag);
    std::sort(edge_sort_begin(N), edge_sort_end(N), EdgeSortCompWrapper<EdgeSortValue<GraphNode,EdgeTy>,CompTy>(comp));
  }

  /**
   * Sorts outgoing edges of a node. Comparison function is over <code>EdgeSortValue<EdgeTy></code>.
   */
  template<typename CompTy>
  void sortEdges(GraphNode N, const CompTy& comp, MethodFlag mflag = MethodFlag::ALL) {
    Galois::Runtime::acquire(N, mflag);
    std::sort(edge_sort_begin(N), edge_sort_end(N), comp);
  }
#endif

  void structureFromFile(const std::string& fname) { Graph::structureFromFile(*this, fname); }

  void structureFromGraph(FileGraph& graph) {
    typedef typename EdgeInfo::value_type EDV;

    numNodes = graph.size();
    numEdges = graph.sizeEdges();
    nodeData.create(numNodes);
    edgeData.allocateInterleaved(numEdges);

    std::vector<NodeInfo*> node_ids;
    node_ids.resize(numNodes);
    for (FileGraph::iterator ii = graph.begin(), ee = graph.end(); ii != ee; ++ii) {
      NodeInfo* curNode = &nodeData[*ii];
      node_ids[*ii] = curNode;
    }
    endNode = numNodes ? &nodeData[numNodes-1] : 0;

    //layout the edges
    EdgeInfo* curEdge = edgeData.data();
    for (FileGraph::iterator ii = graph.begin(), ee = graph.end(); ii != ee; ++ii) {
      node_ids[*ii]->edgeBegin() = curEdge;
      for (FileGraph::neighbor_iterator ni = graph.neighbor_begin(*ii), ne = graph.neighbor_end(*ii); ni != ne; ++ni) {
        if (EdgeInfo::has_value)
          curEdge->construct(graph.getEdgeData<EDV>(ni));
        setEdgeDst<CompressNodePtr>(node_ids, curEdge, *ni);
        ++curEdge;
      }
      node_ids[*ii]->edgeEnd() = curEdge;
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
template<typename NodeTy, typename EdgeTy, bool CompressNodePtr = false, bool CopyInEdgeData = false>
class LC_InlineEdge_InOutGraph: public LC_InlineEdge_Graph<NodeTy,EdgeTy,CompressNodePtr> {
  typedef LC_InlineEdge_Graph<NodeTy,EdgeTy,CompressNodePtr> Super;

protected:
  InlineEdge_InOutGraphImpl::InEdges<Super,!CopyInEdgeData> inEdges;

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
    return &this->nodeData[inEdges.getEdgeDst(ni)];
  }

  in_edge_iterator in_edge_begin(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    Galois::Runtime::acquire(N, mflag);
    if (Galois::Runtime::shouldLock(mflag)) {
      for (edge_iterator ii = inEdges.raw_begin(idFromNode(N)), ee = inEdges.raw_end(idFromNode(N)); ii != ee; ++ii) {
        Galois::Runtime::acquire(&this->nodeData[inEdges.getEdgeDst(ii)], mflag);
      }
    }
    return in_edge_iterator(inEdges.raw_begin(idFromNode(N)));
  }

  in_edge_iterator in_edge_end(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    Galois::Runtime::acquire(N, mflag);
    return in_edge_iterator(inEdges.raw_end(idFromNode(N)));
  }

  InEdgesIterator<LC_InlineEdge_InOutGraph> in_edges(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    return InEdgesIterator<LC_InlineEdge_InOutGraph>(*this, N, mflag);
  }

#if 0
  /**
   * Sorts incoming edges of a node. Comparison function is over EdgeTy.
   */
  template<typename CompTy>
  void sortInEdgesByEdgeData(GraphNode N, const CompTy& comp = std::less<EdgeTy>(), MethodFlag mflag = MethodFlag::ALL) {
    Galois::Runtime::acquire(N, mflag);
    std::sort(inEdges.edge_sort_begin(N), inEdges.edge_sort_end(N), EdgeSortCompWrapper<EdgeSortValue<EdgeTy>,CompTy>(comp));
  }

  /**
   * Sorts incoming edges of a node. Comparison function is over <code>EdgeSortValue<EdgeTy></code>.
   */
  template<typename CompTy>
  void sortInEdges(GraphNode N, const CompTy& comp, MethodFlag mflag = MethodFlag::ALL) {
    Galois::Runtime::acquire(N, mflag);
    std::sort(inEdges.edge_sort_begin(N), inEdges.edge_sort_end(N), comp);
  }
#endif

  size_t idFromNode(GraphNode N) {
    return std::distance(this->nodeData.data(), N);
  }

  GraphNode nodeFromId(size_t N) {
    return &this->nodeData[N];
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
