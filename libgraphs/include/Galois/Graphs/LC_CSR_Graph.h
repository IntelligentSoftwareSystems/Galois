/** Local Computation graphs -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, version 2.1 of the
 * License.
 *
 * Galois is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Galois.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @section Copyright
 *
 * Copyright (C) 2015, The University of Texas at Austin. All rights
 * reserved.
 *
 * @section Description
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */

#ifndef GALOIS_GRAPH__LC_CSR_GRAPH_H
#define GALOIS_GRAPH__LC_CSR_GRAPH_H

#include "Galois/LargeArray.h"
#include "Galois/Graphs/FileGraph.h"
#include "Galois/Graphs/Details.h"
#include "Galois/Runtime/CompilerHelperFunctions.h"

#include <type_traits>

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
 * \snippet test/graph.cpp Using a graph
 *
 * And in C++11:
 *
 * \snippet test/graph.cpp Using a graph cxx11
 *
 * @tparam NodeTy data on nodes
 * @tparam EdgeTy data on out edges
 */
template<typename NodeTy, typename EdgeTy,
  bool HasNoLockable=false,
  bool UseNumaAlloc=false,
  bool HasOutOfLineLockable=false,
  typename FileEdgeTy=EdgeTy>
class LC_CSR_Graph:
    private boost::noncopyable,
    private detail::LocalIteratorFeature<UseNumaAlloc>,
    private detail::OutOfLineLockableFeature<HasOutOfLineLockable && !HasNoLockable> {
  template<typename Graph> friend class LC_InOut_Graph;

public:
  template<bool _has_id>
  struct with_id { typedef LC_CSR_Graph type; };

  template<typename _node_data>
  struct with_node_data { typedef LC_CSR_Graph<_node_data,EdgeTy,HasNoLockable,UseNumaAlloc,HasOutOfLineLockable,FileEdgeTy> type; };

  template<typename _edge_data>
  struct with_edge_data { typedef LC_CSR_Graph<NodeTy,_edge_data,HasNoLockable,UseNumaAlloc,HasOutOfLineLockable,FileEdgeTy> type; };

  template<typename _file_edge_data>
  struct with_file_edge_data { typedef LC_CSR_Graph<NodeTy,EdgeTy,HasNoLockable,UseNumaAlloc,HasOutOfLineLockable,_file_edge_data> type; };

  //! If true, do not use abstract locks in graph
  template<bool _has_no_lockable>
  struct with_no_lockable { typedef LC_CSR_Graph<NodeTy,EdgeTy,_has_no_lockable,UseNumaAlloc,HasOutOfLineLockable,FileEdgeTy> type; };
  template<bool _has_no_lockable>
  using _with_no_lockable = LC_CSR_Graph<NodeTy,EdgeTy,_has_no_lockable,UseNumaAlloc,HasOutOfLineLockable,FileEdgeTy>;

  //! If true, use NUMA-aware graph allocation
  template<bool _use_numa_alloc>
  struct with_numa_alloc { typedef LC_CSR_Graph<NodeTy,EdgeTy,HasNoLockable,_use_numa_alloc,HasOutOfLineLockable,FileEdgeTy> type; };
  template<bool _use_numa_alloc>
  using _with_numa_alloc = LC_CSR_Graph<NodeTy,EdgeTy,HasNoLockable,_use_numa_alloc,HasOutOfLineLockable,FileEdgeTy>;

  //! If true, store abstract locks separate from nodes
  template<bool _has_out_of_line_lockable>
  struct with_out_of_line_lockable { typedef LC_CSR_Graph<NodeTy,EdgeTy,HasNoLockable,UseNumaAlloc,_has_out_of_line_lockable,FileEdgeTy> type; };

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
  typedef FileEdgeTy file_edge_data_type;
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

  template<bool _A1 = EdgeData::has_value, bool _A2 = LargeArray<FileEdgeTy>::has_value>
  void constructEdgeValue(FileGraph& graph, typename FileGraph::edge_iterator nn, 
      typename std::enable_if<!_A1 || _A2>::type* = 0) {
    typedef LargeArray<FileEdgeTy> FED;
    if (EdgeData::has_value)
      edgeData.set(*nn, graph.getEdgeData<typename FED::value_type>(nn));
  }

  template<bool _A1 = EdgeData::has_value, bool _A2 = LargeArray<FileEdgeTy>::has_value>
  void constructEdgeValue(FileGraph& graph, typename FileGraph::edge_iterator nn,
      typename std::enable_if<_A1 && !_A2>::type* = 0) {
    edgeData.set(*nn, {});
  }

  size_t getId(GraphNode N) {
    return N;
  }

  GraphNode getNode(size_t n) {
    return n;
  }

public:

  LC_CSR_Graph(LC_CSR_Graph&& rhs) = default;
  LC_CSR_Graph() = default;

  LC_CSR_Graph& operator=(LC_CSR_Graph&&) = default;

  template<typename EdgeNumFnTy, typename EdgeDstFnTy, typename EdgeDataFnTy>
    LC_CSR_Graph(uint32_t _numNodes, uint64_t _numEdges,
                 EdgeNumFnTy edgeNum, EdgeDstFnTy _edgeDst, EdgeDataFnTy _edgeData)
    :numNodes(_numNodes), numEdges(_numEdges)
  {
//    std::cerr << "\n**" << numNodes << " " << numEdges << "\n\n";
    if (UseNumaAlloc) {
      nodeData.allocateLocal(numNodes);
      edgeIndData.allocateLocal(numNodes);
      edgeDst.allocateLocal(numEdges);
      edgeData.allocateLocal(numEdges);
      this->outOfLineAllocateLocal(numNodes, false);
    } else {
      nodeData.allocateInterleaved(numNodes);
      edgeIndData.allocateInterleaved(numNodes);
      edgeDst.allocateInterleaved(numEdges);
      edgeData.allocateInterleaved(numEdges);
      this->outOfLineAllocateInterleaved(numNodes);
    }
//    std::cerr << "Done Alloc\n";
    for (size_t n = 0; n < numNodes; ++n) {
      nodeData.constructAt(n);
    }
//    std::cerr << "Done Node Construct\n";
    uint64_t cur = 0;
    for (size_t n = 0; n < numNodes; ++n) {
      cur += edgeNum(n);
      edgeIndData[n] = cur;
    }
//    std::cerr << "Done Edge Reserve\n";
    cur = 0;
    for (size_t n = 0; n < numNodes; ++n) {
//      if (n % (1024*128) == 0)
//        std::cout << n << " " << cur << "\n";
      for (uint64_t e = 0, ee = edgeNum(n); e < ee; ++e) {
        if (EdgeData::has_value)
          edgeData.set(cur, _edgeData(n, e));
        edgeDst[cur] = _edgeDst(n, e);
        ++cur;
      }
    }
//    std::cerr << "Done Construct\n";
  }

  friend void swap(LC_CSR_Graph& lhs, LC_CSR_Graph& rhs) {
    swap(lhs.nodeData, rhs.nodeData);
    swap(lhs.edgeIndData, rhs.edgeIndData);
    swap(lhs.edgeDst, rhs.edgeDst);
    swap(lhs.edgeData, rhs.edgeData);
    std::swap(lhs.numNodes, rhs.numNodes);
    std::swap(lhs.numEdges, rhs.numEdges);
  }
  
  node_data_reference getData(GraphNode N, MethodFlag mflag = MethodFlag::WRITE) {
    // Galois::Runtime::checkWrite(mflag, false);
    NodeInfo& NI = nodeData[N];
    acquireNode(N, mflag);
    return NI.getData();
  }

  edge_data_reference getEdgeData(edge_iterator ni, MethodFlag mflag = MethodFlag::UNPROTECTED) {
    // Galois::Runtime::checkWrite(mflag, false);
    return edgeData[*ni];
  }

  GraphNode getEdgeDst(edge_iterator ni) {
    return edgeDst[*ni];
  }

  size_t size() const { return numNodes; }
  size_t sizeEdges() const { return numEdges; }

  iterator begin() const { return iterator(0); }
  iterator end() const { return iterator(numNodes); }

  const_local_iterator local_begin() const { return const_local_iterator(this->localBegin(numNodes)); }
  const_local_iterator local_end() const { return const_local_iterator(this->localEnd(numNodes)); }
  local_iterator local_begin() { return local_iterator(this->localBegin(numNodes)); }
  local_iterator local_end() { return local_iterator(this->localEnd(numNodes)); }

  edge_iterator edge_begin(GraphNode N, MethodFlag mflag = MethodFlag::WRITE) {
    acquireNode(N, mflag);
    if (Galois::Runtime::shouldLock(mflag)) {
      for (edge_iterator ii = raw_begin(N), ee = raw_end(N); ii != ee; ++ii) {
        acquireNode(edgeDst[*ii], mflag);
      }
    }
    return raw_begin(N);
  }

  edge_iterator edge_end(GraphNode N, MethodFlag mflag = MethodFlag::WRITE) {
    acquireNode(N, mflag);
    return raw_end(N);
  }

  Runtime::iterable<NoDerefIterator<edge_iterator>> edges(GraphNode N, MethodFlag mflag = MethodFlag::WRITE) {
    return detail::make_no_deref_range(edge_begin(N, mflag), edge_end(N, mflag));
  }

  Runtime::iterable<NoDerefIterator<edge_iterator>> out_edges(GraphNode N, MethodFlag mflag = MethodFlag::WRITE) {
    return edges(N, mflag);
  }

  /**
   * Sorts outgoing edges of a node. Comparison function is over EdgeTy.
   */
  template<typename CompTy>
  void sortEdgesByEdgeData(GraphNode N, const CompTy& comp = std::less<EdgeTy>(), MethodFlag mflag = MethodFlag::WRITE) {
    acquireNode(N, mflag);
    std::sort(edge_sort_begin(N), edge_sort_end(N), detail::EdgeSortCompWrapper<EdgeSortValue<GraphNode,EdgeTy>,CompTy>(comp));
  }

  /**
   * Sorts outgoing edges of a node. Comparison function is over <code>EdgeSortValue<EdgeTy></code>.
   */
  template<typename CompTy>
  void sortEdges(GraphNode N, const CompTy& comp, MethodFlag mflag = MethodFlag::WRITE) {
    acquireNode(N, mflag);
    std::sort(edge_sort_begin(N), edge_sort_end(N), comp);
  }


  template <typename F>
  ptrdiff_t partition_neighbors (GraphNode N, const F& func) {

    auto beg = &edgeDst[*raw_begin (N)];
    auto end = &edgeDst[*raw_end (N)];
    auto mid = std::partition (beg, end, func);
    return (mid - beg);
  }

  void allocateFrom(FileGraph& graph) {
    numNodes = graph.size();
    numEdges = graph.sizeEdges();
    if (UseNumaAlloc) {
      nodeData.allocateBlocked(numNodes);
      edgeIndData.allocateBlocked(numNodes);
      edgeDst.allocateBlocked(numEdges);
      edgeData.allocateBlocked(numEdges);
      this->outOfLineAllocateBlocked(numNodes);
    } else {
      nodeData.allocateInterleaved(numNodes);
      edgeIndData.allocateInterleaved(numNodes);
      edgeDst.allocateInterleaved(numEdges);
      edgeData.allocateInterleaved(numEdges);
      this->outOfLineAllocateInterleaved(numNodes);
    }
  }

  void allocateFrom(uint32_t nNodes, uint64_t nEdges) {
    numNodes = nNodes;
    numEdges = nEdges;
    if (UseNumaAlloc) {
      nodeData.allocateLocal(numNodes);
      edgeIndData.allocateLocal(numNodes);
      edgeDst.allocateLocal(numEdges);
      edgeData.allocateLocal(numEdges);
      this->outOfLineAllocateLocal(numNodes);
    } else {
      nodeData.allocateInterleaved(numNodes);
      edgeIndData.allocateInterleaved(numNodes);
      edgeDst.allocateInterleaved(numEdges);
      edgeData.allocateInterleaved(numEdges);
      this->outOfLineAllocateInterleaved(numNodes);
    }
  }

  void constructNodes() {
    for (uint32_t x = 0; x < numNodes; ++x) {
      nodeData.constructAt(x);
      this->outOfLineConstructAt(x);
    }
  }

  void constructEdge(uint64_t e, uint32_t dst, const typename EdgeData::value_type& val) {
    edgeData.set(e, val);
    edgeDst[e] = dst;
  }

  void constructEdge(uint64_t e, uint32_t dst) {
    edgeDst[e] = dst;
  }

  void fixEndEdge(uint32_t n, uint64_t e) {
    edgeIndData[n] = e;
  }

  // perform an in-memory tranpose of the graph, replacing the original
  // CSR to CSC
  void transpose() {
    EdgeDst edgeDst_old;
    EdgeData edgeData_new;
    EdgeIndData edgeIndData_old;
    LargeArray< std::atomic<uint64_t> > edgeIndData_temp;
    if (UseNumaAlloc) {
      edgeIndData_old.allocateLocal(numNodes);
      edgeIndData_temp.allocateLocal(numNodes);
      edgeDst_old.allocateLocal(numEdges);
      edgeData_new.allocateLocal(numEdges);
    } else {
      edgeIndData_old.allocateInterleaved(numNodes);
      edgeIndData_temp.allocateLocal(numNodes);
      edgeDst_old.allocateInterleaved(numEdges);
      edgeData_new.allocateInterleaved(numEdges);
    }
    Galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(numNodes),
      [&](uint32_t n){
        edgeIndData_old[n] = edgeIndData[n];
        edgeIndData_temp[n] = 0;
      }, Galois::loopname("TRANSPOSE_EDGEINTDATA_COPY"), Galois::numrun("0"));
    Galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(numEdges),
      [&](uint32_t e){
        auto dst = edgeDst[e];
        edgeDst_old[e] = dst;
        Galois::atomicAdd(edgeIndData_temp[dst], (uint64_t)1);
      }, Galois::loopname("TRANSPOSE_COUNT_INCOMING_EDGES"), Galois::numrun("0"));
    for (uint32_t n = 1; n < numNodes; ++n) {
      edgeIndData_temp[n] += edgeIndData_temp[n-1];
    }
    Galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(numNodes),
      [&](uint32_t n){
        edgeIndData[n] = edgeIndData_temp[n];
      }, Galois::loopname("TRANSPOSE_EDGEINTDATA_SET"), Galois::numrun("0"));
    edgeIndData_temp[0] = 0;
    Galois::do_all(boost::counting_iterator<uint32_t>(1), boost::counting_iterator<uint32_t>(numNodes),
      [&](uint32_t n){
        edgeIndData_temp[n] = edgeIndData[n-1];
      }, Galois::loopname("TRANSPOSE_EDGEINTDATA_TEMP"), Galois::numrun("0"));
#ifndef GALOIS_TRANSPOSE_SERIAL
    Galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(numNodes),
      [&](uint32_t src){
        uint64_t e;
        if (src == 0) e = 0;
        else e = edgeIndData_old[src-1];
        while (e < edgeIndData_old[src]) {
          auto dst = edgeDst_old[e];
          auto e_new = Galois::atomicAdd(edgeIndData_temp[dst], (uint64_t)1);
          edgeDst[e_new] = src;
          edgeDataCopy(edgeData_new, edgeData, e_new, e);
          e++;
        }
      }, Galois::loopname("TRANSPOSE_REORDER_EDGES"), Galois::numrun("0"));
#else
    for (uint32_t src = 0; src < numNodes; ++src) {
      uint64_t e;
      if (src == 0) e = 0;
      else e = edgeIndData_old[src-1];
      while (e < edgeIndData_old[src]) {
        auto dst = edgeDst_old[e];
        auto e_new = Galois::atomicAdd(edgeIndData_temp[dst], (uint64_t)1);
        edgeDst[e_new] = src;
        edgeDataCopy(edgeData_new, edgeData, e_new, e);
        e++;
      }
    }
#endif
    if (EdgeData::has_value) {
      Galois::do_all(boost::counting_iterator<uint32_t>(0), boost::counting_iterator<uint32_t>(numEdges),
        [&](uint32_t e){
          edgeDataCopy(edgeData, edgeData_new, e, e);
        }, Galois::loopname("TRANSPOSE_EDGEDATA_SET"), Galois::numrun("0"));
    }
  }

  template<bool is_non_void = EdgeData::has_value>
  void edgeDataCopy(EdgeData &edgeData_new, EdgeData &edgeData, uint64_t e_new, uint64_t e, typename std::enable_if<is_non_void>::type* = 0) {
    edgeData_new[e_new] = edgeData[e];
  }

  template<bool is_non_void = EdgeData::has_value>
  void edgeDataCopy(EdgeData &edgeData_new, EdgeData &edgeData, uint64_t e_new, uint64_t e, typename std::enable_if<!is_non_void>::type* = 0) {
  }

  void constructFrom(FileGraph& graph, unsigned tid, unsigned total) {
    auto r = graph.divideByNode(
        NodeData::size_of::value + EdgeIndData::size_of::value + LC_CSR_Graph::size_of_out_of_line::value,
        EdgeDst::size_of::value + EdgeData::size_of::value,
        tid, total).first;
    this->setLocalRange(*r.first, *r.second);
    for (FileGraph::iterator ii = r.first, ei = r.second; ii != ei; ++ii) {
      nodeData.constructAt(*ii);
      edgeIndData[*ii] = *graph.edge_end(*ii);
      this->outOfLineConstructAt(*ii);
      for (FileGraph::edge_iterator nn = graph.edge_begin(*ii), en = graph.edge_end(*ii); nn != en; ++nn) {
        constructEdgeValue(graph, nn);
        edgeDst[*nn] = graph.getEdgeDst(nn);
      }
    }
  }
};

} // end namespace
} // end namespace

#endif
