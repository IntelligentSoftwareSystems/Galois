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

#include "Galois/config.h"
#include "Galois/LargeArray.h"
#include "Galois/Graph/FileGraph.h"
#include "Galois/Graph/Details.h"
#include "Galois/Runtime/MethodFlags.h"

#include GALOIS_CXX11_STD_HEADER(type_traits)

namespace Galois {
namespace Graph {

/**
 * Local computation graph (i.e., graph structure does not change). The data representation
 * is the traditional compressed-sparse-row (CSR) format.
 *
 * The position of template parameters may change between Galois releases; the
 * most robust way to specify them is through the with_XXX nested templates.
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
 * // or typedef Galois::Graph::LC_CSR_Graph<int,int>::with_no_lockable<true>::with_numa_alloc<true>
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
template<typename NodeTy, typename EdgeTy,
  bool HasNoLockable=false,
  bool UseNumaAlloc=false,
  bool HasOutOfLineLockable=false>
class LC_CSR_Graph:
    private boost::noncopyable,
    private detail::LocalIteratorFeature<UseNumaAlloc>,
    private detail::OutOfLineLockableFeature<HasOutOfLineLockable && !HasNoLockable> {
  template<typename Graph> friend class LC_InOut_Graph;

public:
  template<bool _has_id>
  struct with_id { typedef LC_CSR_Graph type; };

  template<typename _node_data>
  struct with_node_data { typedef LC_CSR_Graph<_node_data,EdgeTy,HasNoLockable,UseNumaAlloc,HasOutOfLineLockable> type; };

  //! If true, do not use abstract locks in graph
  template<bool _has_no_lockable>
  struct with_no_lockable { typedef LC_CSR_Graph<NodeTy,EdgeTy,_has_no_lockable,UseNumaAlloc,HasOutOfLineLockable> type; };

  //! If true, use NUMA-aware graph allocation
  template<bool _use_numa_alloc>
  struct with_numa_alloc { typedef LC_CSR_Graph<NodeTy,EdgeTy,HasNoLockable,_use_numa_alloc,HasOutOfLineLockable> type; };

  //! If true, store abstract locks separate from nodes
  template<bool _has_out_of_line_lockable>
  struct with_out_of_line_lockable { typedef LC_CSR_Graph<NodeTy,EdgeTy,HasNoLockable,UseNumaAlloc,_has_out_of_line_lockable> type; };

  typedef read_default_graph_tag read_tag;

protected:
  typedef LargeArray<EdgeTy> EdgeData;
  typedef LargeArray<uint32_t> EdgeDst;
  typedef detail::NodeInfoBaseTypes<NodeTy,!HasNoLockable && !HasOutOfLineLockable> NodeInfoTypes;
  typedef detail::NodeInfoBase<NodeTy,!HasNoLockable && !HasOutOfLineLockable> NodeInfo;
  typedef LargeArray<uint64_t> EdgeIndData;
  typedef LargeArray<NodeInfo> NodeData;

public:
  typedef uint32_t GraphNode;
  typedef EdgeTy edge_data_type;
  typedef NodeTy node_data_type;
  typedef typename EdgeData::reference edge_data_reference;
  typedef typename NodeInfoTypes::reference node_data_reference;
  typedef boost::counting_iterator<typename EdgeIndData::value_type> edge_iterator;
  typedef boost::counting_iterator<typename EdgeDst::value_type> iterator;
  typedef iterator const_iterator;
  typedef iterator local_iterator;
  typedef iterator const_local_iterator;

protected:
  NodeData nodeData;
  EdgeIndData edgeIndData;
  EdgeDst edgeDst;
  EdgeData edgeData;

  uint64_t numNodes;
  uint64_t numEdges;

  typedef detail::EdgeSortIterator<GraphNode,typename EdgeIndData::value_type,EdgeDst,EdgeData> edge_sort_iterator;

  edge_iterator raw_begin(GraphNode N) const {
    return edge_iterator((N == 0) ? 0 : edgeIndData[N-1]);
  }

  edge_iterator raw_end(GraphNode N) const {
    return edge_iterator(edgeIndData[N]);
  }

  edge_sort_iterator edge_sort_begin(GraphNode N) {
    return edge_sort_iterator(*raw_begin(N), &edgeDst, &edgeData);
  }

  edge_sort_iterator edge_sort_end(GraphNode N) {
    return edge_sort_iterator(*raw_end(N), &edgeDst, &edgeData);
  }

  template<bool _A1 = HasNoLockable, bool _A2 = HasOutOfLineLockable>
  void acquireNode(GraphNode N, MethodFlag mflag, typename std::enable_if<!_A1 && !_A2>::type* = 0) {
    Galois::Runtime::acquire(&nodeData[N], mflag);
  }

  template<bool _A1 = HasOutOfLineLockable, bool _A2 = HasNoLockable>
  void acquireNode(GraphNode N, MethodFlag mflag, typename std::enable_if<_A1 && !_A2>::type* = 0) {
    this->outOfLineAcquire(getId(N), mflag);
  }

  template<bool _A1 = HasOutOfLineLockable, bool _A2 = HasNoLockable>
  void acquireNode(GraphNode N, MethodFlag mflag, typename std::enable_if<_A2>::type* = 0) { }

  size_t getId(GraphNode N) {
    return N;
  }

  GraphNode getNode(size_t n) {
    return n;
  }

public:
  node_data_reference getData(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    Galois::Runtime::checkWrite(mflag, false);
    NodeInfo& NI = nodeData[N];
    acquireNode(N, mflag);
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

  const_local_iterator local_begin() const { return const_local_iterator(this->localBegin(numNodes)); }
  const_local_iterator local_end() const { return const_local_iterator(this->localEnd(numNodes)); }
  local_iterator local_begin() { return local_iterator(this->localBegin(numNodes)); }
  local_iterator local_end() { return local_iterator(this->localEnd(numNodes)); }

  edge_iterator edge_begin(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    acquireNode(N, mflag);
    if (Galois::Runtime::shouldLock(mflag)) {
      for (edge_iterator ii = raw_begin(N), ee = raw_end(N); ii != ee; ++ii) {
        acquireNode(edgeDst[*ii], mflag);
      }
    }
    return raw_begin(N);
  }

  edge_iterator edge_end(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    acquireNode(N, mflag);
    return raw_end(N);
  }

  detail::EdgesIterator<LC_CSR_Graph> out_edges(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    return detail::EdgesIterator<LC_CSR_Graph>(*this, N, mflag);
  }

  /**
   * Sorts outgoing edges of a node. Comparison function is over EdgeTy.
   */
  template<typename CompTy>
  void sortEdgesByEdgeData(GraphNode N, const CompTy& comp = std::less<EdgeTy>(), MethodFlag mflag = MethodFlag::ALL) {
    acquireNode(N, mflag);
    std::sort(edge_sort_begin(N), edge_sort_end(N), detail::EdgeSortCompWrapper<EdgeSortValue<GraphNode,EdgeTy>,CompTy>(comp));
  }

  /**
   * Sorts outgoing edges of a node. Comparison function is over <code>EdgeSortValue<EdgeTy></code>.
   */
  template<typename CompTy>
  void sortEdges(GraphNode N, const CompTy& comp, MethodFlag mflag = MethodFlag::ALL) {
    acquireNode(N, mflag);
    std::sort(edge_sort_begin(N), edge_sort_end(N), comp);
  }

  void allocateFrom(FileGraph& graph) {
    numNodes = graph.size();
    numEdges = graph.sizeEdges();
    if (UseNumaAlloc) {
      nodeData.allocateLocal(numNodes, false);
      edgeIndData.allocateLocal(numNodes, false);
      edgeDst.allocateLocal(numEdges, false);
      edgeData.allocateLocal(numEdges, false);
      this->outOfLineAllocateLocal(numNodes, false);
    } else {
      nodeData.allocateInterleaved(numNodes);
      edgeIndData.allocateInterleaved(numNodes);
      edgeDst.allocateInterleaved(numEdges);
      edgeData.allocateInterleaved(numEdges);
      this->outOfLineAllocateInterleaved(numNodes);
    }
  }

  void constructFrom(FileGraph& graph, unsigned tid, unsigned total) {
    auto r = graph.divideBy(
        NodeData::size_of::value + EdgeIndData::size_of::value + LC_CSR_Graph::size_of_out_of_line::value,
        EdgeDst::size_of::value + EdgeData::size_of::value,
        tid, total);
    this->setLocalRange(*r.first, *r.second);
    for (FileGraph::iterator ii = r.first, ei = r.second; ii != ei; ++ii) {
      nodeData.constructAt(*ii);
      edgeIndData[*ii] = *graph.edge_end(*ii);
      this->outOfLineConstructAt(*ii);
      for (FileGraph::edge_iterator nn = graph.edge_begin(*ii), en = graph.edge_end(*ii); nn != en; ++nn) {
        if (EdgeData::has_value)
          edgeData.set(*nn, graph.getEdgeData<typename EdgeData::value_type>(nn));
        edgeDst[*nn] = graph.getEdgeDst(nn);
      }
    }
  }
};

} // end namespace
} // end namespace

#endif
