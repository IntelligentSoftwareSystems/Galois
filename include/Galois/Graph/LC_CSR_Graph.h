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
#ifndef GALOIS_GRAPH_LC_CSR_GRAPH_H
#define GALOIS_GRAPH_LC_CSR_GRAPH_H

#include "Galois/LargeArray.h"
#include "Galois/Graph/FileGraph.h"
#include "Galois/Graph/Util.h"
#include "Galois/Runtime/MethodFlags.h"

#include <boost/utility/enable_if.hpp>

namespace Galois {
namespace Graph {

namespace CSR_InOutGraphImpl {

//! Dispatch type to handle implementation of in edges
template<typename Graph, bool HasPointerData>
class InEdges;

template<typename Graph>
class InEdges<Graph,false> {
  Graph graphStore; 
  Graph* graph;

  typedef typename Graph::GraphNode GraphNode;
  typedef typename Graph::edge_iterator edge_iterator;
  typedef typename Graph::edge_data_reference edge_data_reference;

public:
  edge_iterator raw_begin(GraphNode n) const {
    return graph->raw_neighbor_begin(n);
  }

  edge_iterator raw_end(GraphNode n) const {
    return graph->raw_neighbor_end(n);
  }

  GraphNode getEdgeDst(edge_iterator e) {
    return graph->getEdgeDst(e);
  }

  typename Graph::edge_sort_iterator edge_sort_begin(GraphNode n) {
    return graph->edge_sort_begin(n);
  }

  typename Graph::edge_sort_iterator edge_sort_end(GraphNode n) {
    return graph->edge_sort_end(n);
  }

  edge_data_reference getEdgeData(edge_iterator e) {
    return graph->getEdgeData(e);
  }

  void initialize(Graph* g, bool symmetric) {
    if (symmetric) {
      graph = g;
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
 * is the traditional compressed-sparse-row (CSR) format.
 *
 * An example of use:
 * 
 * \code
 * typedef Galois::Graph::LC_CSR_Graph<int,int> Graph;
 * 
 * // Create graph
 * Graph g;
 * g.structureFromFile(inputfile);
 *
 * // Traverse graph
 * for (Graph::iterator ii = g.begin(), ei = g.end(); ii != ei; ++ii) {
 *   Graph::GraphNode src = *ii;
 *   for (Graph::edge_iterator jj = g.edge_begin(src), ej = g.edge_end(src); jj != ej; ++jj) {
 *     Graph::GraphNode dst = g.getEdgeDst(jj);
 *     int edgeData = g.getEdgeData(jj);
 *     int nodeData = g.getData(dst);
 *   }
 * }
 * \endcode
 *
 * And in C++11:
 *
 * \code
 * typedef Galois::Graph::LC_CSR_Graph<int,int> Graph;
 * 
 * // Create graph
 * Graph g;
 * g.structureFromFile(inputfile);
 *
 * // Traverse graph
 * for (Graph::GraphNode src : g) {
 *   for (Graph::edge_iterator edge : g.out_edges(src)) {
 *     Graph::GraphNode dst = g.getEdgeDst(edge);
 *     int edgeData = g.getEdgeData(edge);
 *     int nodeData = g.getData(dst);
 *   }
 * }
 * \endcode
 *
 * @tparam NodeTy data on nodes
 * @tparam EdgeTy data on out edges
 */
template<typename NodeTy, typename EdgeTy>
class LC_CSR_Graph: boost::noncopyable {
  template<typename Graph,bool HasPointerData>
    friend class CSR_InOutGraphImpl::InEdges;

protected:
  typedef LargeArray<EdgeTy> EdgeData;
  typedef LargeArray<uint32_t> EdgeDst;
  typedef NodeInfoBase<NodeTy,true> NodeInfo;
  typedef LargeArray<uint64_t> EdgeIndData;
  typedef LargeArray<NodeInfo> NodeData;

  NodeData nodeData;
  EdgeIndData edgeIndData;
  EdgeDst edgeDst;
  EdgeData edgeData;

  uint64_t numNodes;
  uint64_t numEdges;

public:
  typedef uint32_t GraphNode;
  typedef EdgeTy edge_data_type;
  typedef NodeTy node_data_type;
  typedef typename EdgeData::reference edge_data_reference;
  typedef typename NodeInfo::reference node_data_reference;
  typedef boost::counting_iterator<typename EdgeIndData::value_type> edge_iterator;
  typedef boost::counting_iterator<typename EdgeDst::value_type> iterator;
  typedef iterator const_iterator;
  typedef iterator local_iterator;
  typedef iterator const_local_iterator;

protected:
  typedef EdgeSortIterator<GraphNode,typename EdgeIndData::value_type,EdgeDst,EdgeData> edge_sort_iterator;

  edge_iterator raw_neighbor_begin(uint32_t N) const {
    return edge_iterator((N == 0) ? 0 : edgeIndData[N-1]);
  }

  edge_iterator raw_neighbor_end(uint32_t N) const {
    return edge_iterator(edgeIndData[N]);
  }

  edge_sort_iterator edge_sort_begin(uint32_t src) {
    return edge_sort_iterator(*raw_neighbor_begin(src), &edgeDst, &edgeData);
  }

  edge_sort_iterator edge_sort_end(uint32_t src) {
    return edge_sort_iterator(*raw_neighbor_end(src), &edgeDst, &edgeData);
  }

public:
  node_data_reference getData(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    Galois::Runtime::checkWrite(mflag, false);
    NodeInfo& NI = nodeData[N];
    Galois::Runtime::acquire(&NI, mflag);
    return NI.getData();
  }

  edge_data_reference getEdgeData(edge_iterator ni, MethodFlag mflag = MethodFlag::NONE) {
    Galois::Runtime::checkWrite(mflag, false);
    return edgeData[*ni];
  }

  GraphNode getEdgeDst(edge_iterator ni) {
    return edgeDst[*ni];
  }

  uint64_t size() const { return numNodes; }
  uint64_t sizeEdges() const { return numEdges; }

  iterator begin() const { return iterator(0); }
  iterator end() const { return iterator(numNodes); }

  local_iterator local_begin() const { return iterator(localStart(numNodes)); }
  local_iterator local_end() const { return iterator(localEnd(numNodes)); }

  edge_iterator edge_begin(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    Galois::Runtime::acquire(&nodeData[N], mflag);
    if (Galois::Runtime::shouldLock(mflag)) {
      for (edge_iterator ii = raw_neighbor_begin(N), ee = raw_neighbor_end(N); ii != ee; ++ii) {
        Galois::Runtime::acquire(&nodeData[edgeDst[*ii]], mflag);
      }
    }
    return raw_neighbor_begin(N);
  }

  edge_iterator edge_end(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    NodeInfo& NI = nodeData[N];
    Galois::Runtime::acquire(&NI, mflag);
    return raw_neighbor_end(N);
  }

  EdgesIterator<LC_CSR_Graph> out_edges(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    return EdgesIterator<LC_CSR_Graph>(*this, N, mflag);
  }

  /**
   * Sorts outgoing edges of a node. Comparison function is over EdgeTy.
   */
  template<typename CompTy>
  void sortEdgesByEdgeData(GraphNode N, const CompTy& comp = std::less<EdgeTy>(), MethodFlag mflag = MethodFlag::ALL) {
    Galois::Runtime::acquire(&nodeData[N], mflag);
    std::sort(edge_sort_begin(N), edge_sort_end(N), EdgeSortCompWrapper<EdgeSortValue<GraphNode,EdgeTy>,CompTy>(comp));
  }

  /**
   * Sorts outgoing edges of a node. Comparison function is over <code>EdgeSortValue<EdgeTy></code>.
   */
  template<typename CompTy>
  void sortEdges(GraphNode N, const CompTy& comp, MethodFlag mflag = MethodFlag::ALL) {
    Galois::Runtime::acquire(&nodeData[N], mflag);
    std::sort(edge_sort_begin(N), edge_sort_end(N), comp);
  }

  void structureFromFile(const std::string& fname) { Graph::structureFromFile(*this, fname); }

  void structureFromGraph(FileGraph& graph) {
    numNodes = graph.size();
    numEdges = graph.sizeEdges();
    nodeData.create(numNodes);
    edgeIndData.create(numNodes);
    edgeDst.create(numEdges);
    edgeData.create(numEdges);

    typedef typename EdgeData::value_type EDV;

    if (EdgeData::has_value)
      edgeData.copyIn(graph.edge_data_begin<EDV>(), graph.edge_data_end<EDV>());
    std::copy(graph.edge_id_begin(), graph.edge_id_end(), edgeIndData.data());
    std::copy(graph.node_id_begin(), graph.node_id_end(), edgeDst.data());
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
class LC_CSR_InOutGraph: public LC_CSR_Graph<NodeTy,EdgeTy> {
  typedef LC_CSR_Graph<NodeTy,EdgeTy> Super;

protected:
  CSR_InOutGraphImpl::InEdges<Super,!CopyInEdgeData> inEdges;

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
    return inEdges.getEdgeData(*ni);
  }

  GraphNode getInEdgeDst(in_edge_iterator ni) {
    return inEdges.getEdgeDst(*ni);
  }

  in_edge_iterator in_edge_begin(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    Galois::Runtime::acquire(&this->nodeData[N], mflag);
    if (Galois::Runtime::shouldLock(mflag)) {
      for (edge_iterator ii = inEdges.raw_begin(N), ee = inEdges.raw_end(N); ii != ee; ++ii) {
        Galois::Runtime::acquire(&this->nodeData[inEdges.getEdgeDst(ii)], mflag);
      }
    }
    return inEdges.raw_begin(N);
  }

  in_edge_iterator in_edge_end(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    Galois::Runtime::acquire(&this->nodeData[N], mflag);
    return inEdges.raw_end(N);
  }

  InEdgesIterator<LC_CSR_InOutGraph> in_edges(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    return InEdgesIterator<LC_CSR_InOutGraph>(*this, N, mflag);
  }

  /**
   * Sorts incoming edges of a node. Comparison function is over EdgeTy.
   */
  template<typename CompTy>
  void sortInEdgesByEdgeData(GraphNode N, const CompTy& comp = std::less<EdgeTy>(), MethodFlag mflag = MethodFlag::ALL) {
    Galois::Runtime::acquire(&this->nodeData[N], mflag);
    std::sort(inEdges.edge_sort_begin(N), inEdges.edge_sort_end(N), EdgeSortCompWrapper<EdgeSortValue<GraphNode,EdgeTy>,CompTy>(comp));
  }

  /**
   * Sorts incoming edges of a node. Comparison function is over <code>EdgeSortValue<EdgeTy></code>.
   */
  template<typename CompTy>
  void sortInEdges(GraphNode N, const CompTy& comp, MethodFlag mflag = MethodFlag::ALL) {
    Galois::Runtime::acquire(&this->nodeData[N], mflag);
    std::sort(inEdges.edge_sort_begin(N), inEdges.edge_sort_end(N), comp);
  }

  size_t idFromNode(GraphNode N) {
    return N;
  }

  GraphNode nodeFromId(size_t N) {
    return N;
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
      GALOIS_DIE("number of nodes in graph and its transpose do not match");
    } else if (graph.sizeEdges() != transpose.sizeEdges()) {
      GALOIS_DIE("number of edges in graph and its transpose do not match");
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
