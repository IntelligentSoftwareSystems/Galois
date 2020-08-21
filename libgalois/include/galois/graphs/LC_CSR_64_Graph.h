/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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
 */

#ifndef GALOIS_GRAPHS_LC_CSR_64_GRAPH_H
#define GALOIS_GRAPHS_LC_CSR_64_GRAPH_H

#include <fstream>
#include <type_traits>

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/binary_object.hpp>
#include <boost/serialization/serialization.hpp>

#include "galois/config.h"
#include "galois/Galois.h"
#include "galois/graphs/Details.h"
#include "galois/graphs/FileGraph.h"
#include "galois/graphs/GraphHelpers.h"
#include "galois/PODResizeableArray.h"

namespace galois::graphs {
/**
 * Local computation graph (i.e., graph structure does not change). The data
 * representation is the traditional compressed-sparse-row (CSR) format.
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
//! [doxygennuma]
template <typename NodeTy, typename EdgeTy, bool HasNoLockable = false,
          bool UseNumaAlloc = false, bool HasOutOfLineLockable = false,
          typename FileEdgeTy = EdgeTy>
class LC_CSR_64_Graph :
    //! [doxygennuma]
    private boost::noncopyable,
    private internal::LocalIteratorFeature<UseNumaAlloc>,
    private internal::OutOfLineLockableFeature<HasOutOfLineLockable &&
                                               !HasNoLockable> {
  template <typename Graph>
  friend class LC_InOut_Graph;

public:
  template <bool _has_id>
  struct with_id {
    typedef LC_CSR_64_Graph type;
  };

  template <typename _node_data>
  struct with_node_data {
    typedef LC_CSR_64_Graph<_node_data, EdgeTy, HasNoLockable, UseNumaAlloc,
                            HasOutOfLineLockable, FileEdgeTy>
        type;
  };

  template <typename _edge_data>
  struct with_edge_data {
    typedef LC_CSR_64_Graph<NodeTy, _edge_data, HasNoLockable, UseNumaAlloc,
                            HasOutOfLineLockable, FileEdgeTy>
        type;
  };

  template <typename _file_edge_data>
  struct with_file_edge_data {
    typedef LC_CSR_64_Graph<NodeTy, EdgeTy, HasNoLockable, UseNumaAlloc,
                            HasOutOfLineLockable, _file_edge_data>
        type;
  };

  //! If true, do not use abstract locks in graph
  template <bool _has_no_lockable>
  struct with_no_lockable {
    typedef LC_CSR_64_Graph<NodeTy, EdgeTy, _has_no_lockable, UseNumaAlloc,
                            HasOutOfLineLockable, FileEdgeTy>
        type;
  };
  template <bool _has_no_lockable>
  using _with_no_lockable =
      LC_CSR_64_Graph<NodeTy, EdgeTy, _has_no_lockable, UseNumaAlloc,
                      HasOutOfLineLockable, FileEdgeTy>;

  //! If true, use NUMA-aware graph allocation; otherwise, use NUMA interleaved
  //! allocation.
  template <bool _use_numa_alloc>
  struct with_numa_alloc {
    typedef LC_CSR_64_Graph<NodeTy, EdgeTy, HasNoLockable, _use_numa_alloc,
                            HasOutOfLineLockable, FileEdgeTy>
        type;
  };
  template <bool _use_numa_alloc>
  using _with_numa_alloc =
      LC_CSR_64_Graph<NodeTy, EdgeTy, HasNoLockable, _use_numa_alloc,
                      HasOutOfLineLockable, FileEdgeTy>;

  //! If true, store abstract locks separate from nodes
  template <bool _has_out_of_line_lockable>
  struct with_out_of_line_lockable {
    typedef LC_CSR_64_Graph<NodeTy, EdgeTy, HasNoLockable, UseNumaAlloc,
                            _has_out_of_line_lockable, FileEdgeTy>
        type;
  };

  typedef read_default_graph_tag read_tag;

protected:
  typedef LargeArray<EdgeTy> EdgeData;
  typedef LargeArray<uint64_t> EdgeDst;
  typedef internal::NodeInfoBaseTypes<NodeTy,
                                      !HasNoLockable && !HasOutOfLineLockable>
      NodeInfoTypes;
  typedef internal::NodeInfoBase<NodeTy,
                                 !HasNoLockable && !HasOutOfLineLockable>
      NodeInfo;
  typedef LargeArray<uint64_t> EdgeIndData;
  typedef LargeArray<NodeInfo> NodeData;

public:
  typedef uint64_t GraphNode;
  typedef EdgeTy edge_data_type;
  typedef FileEdgeTy file_edge_data_type;
  typedef NodeTy node_data_type;
  typedef typename EdgeData::reference edge_data_reference;
  typedef typename NodeInfoTypes::reference node_data_reference;
  using edge_iterator =
      boost::counting_iterator<typename EdgeIndData::value_type>;
  using iterator = boost::counting_iterator<typename EdgeDst::value_type>;
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

  typedef internal::EdgeSortIterator<
      GraphNode, typename EdgeIndData::value_type, EdgeDst, EdgeData>
      edge_sort_iterator;

  edge_iterator raw_begin(GraphNode N) const {
    return edge_iterator((N == 0) ? 0 : edgeIndData[N - 1]);
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

  template <bool _A1 = HasNoLockable, bool _A2 = HasOutOfLineLockable>
  void acquireNode(GraphNode N, MethodFlag mflag,
                   typename std::enable_if<!_A1&& !_A2>::type* = 0) {
    galois::runtime::acquire(&nodeData[N], mflag);
  }

  template <bool _A1 = HasOutOfLineLockable, bool _A2 = HasNoLockable>
  void acquireNode(GraphNode N, MethodFlag mflag,
                   typename std::enable_if<_A1&& !_A2>::type* = 0) {
    this->outOfLineAcquire(getId(N), mflag);
  }

  template <bool _A1 = HasOutOfLineLockable, bool _A2 = HasNoLockable>
  void acquireNode(GraphNode, MethodFlag,
                   typename std::enable_if<_A2>::type* = 0) {}

  template <bool _A1 = EdgeData::has_value,
            bool _A2 = LargeArray<FileEdgeTy>::has_value>
  void constructEdgeValue(FileGraph& graph,
                          typename FileGraph::edge_iterator nn,
                          typename std::enable_if<!_A1 || _A2>::type* = 0) {
    typedef LargeArray<FileEdgeTy> FED;
    if (EdgeData::has_value)
      edgeData.set(*nn, graph.getEdgeData<typename FED::value_type>(nn));
  }

  template <bool _A1 = EdgeData::has_value,
            bool _A2 = LargeArray<FileEdgeTy>::has_value>
  void constructEdgeValue(FileGraph&, typename FileGraph::edge_iterator nn,
                          typename std::enable_if<_A1&& !_A2>::type* = 0) {
    edgeData.set(*nn, {});
  }

  uint64_t getId(GraphNode N) { return N; }

  GraphNode getNode(uint64_t n) { return n; }

private:
  friend class boost::serialization::access;

  template <typename Archive>
  void save(Archive& ar, const unsigned int) const {
    ar << numNodes;
    ar << numEdges;

    // Large Arrays
    ar << edgeIndData;
    ar << edgeDst;
    ar << edgeData;
  }

  template <typename Archive>
  void load(Archive& ar, const unsigned int) {
    ar >> numNodes;
    ar >> numEdges;

    // Large Arrays
    ar >> edgeIndData;
    ar >> edgeDst;
    ar >> edgeData;

    if (!nodeData.data()) {
      if (UseNumaAlloc) {
        nodeData.allocateBlocked(numNodes);
        this->outOfLineAllocateBlocked(numNodes);
      } else {
        nodeData.allocateInterleaved(numNodes);
        this->outOfLineAllocateInterleaved(numNodes);
      }

      // Construct nodeData largeArray
      for (size_t n = 0; n < numNodes; ++n) {
        nodeData.constructAt(n);
      }
    }
  }

  // The macro BOOST_SERIALIZATION_SPLIT_MEMBER() generates code which invokes
  // the save or load depending on whether the archive is used for saving or
  // loading
  BOOST_SERIALIZATION_SPLIT_MEMBER()

public:
  LC_CSR_64_Graph(LC_CSR_64_Graph&& rhs) = default;

  LC_CSR_64_Graph() = default;

  LC_CSR_64_Graph& operator=(LC_CSR_64_Graph&&) = default;

  /**
   * Serializes node data using Boost.
   *
   * @param ar Boost archive to serialize to.
   */
  void serializeNodeData(boost::archive::binary_oarchive& ar) const {
    ar << nodeData;
  }

  /**
   * Deserializes a Boost archive containing node data to the local node data
   * variable.
   *
   * @param ar Boost archive to deserialize from.
   */
  void deSerializeNodeData(boost::archive::binary_iarchive& ar) {
    ar >> nodeData;
  }

  /**
   * Serializes graph using Boost.
   *
   * @param ar Boost archive to serialize to.
   */
  void serializeGraph(boost::archive::binary_oarchive& ar) const {
    ar << numNodes;
    ar << numEdges;

    // Large Arrays
    ar << nodeData;
    ar << edgeIndData;
    ar << edgeDst;
    ar << edgeData;
  }

  /**
   * Deserializes a Boost archive to the local graph.
   *
   * @param ar Boost archive to deserialize from.
   */
  void deSerializeGraph(boost::archive::binary_iarchive& ar) {
    ar >> numNodes;
    ar >> numEdges;

    // Large Arrays
    ar >> nodeData;
    ar >> edgeIndData;
    ar >> edgeDst;
    ar >> edgeData;
  }

  /**
   * Accesses the "prefix sum" of this graph; takes advantage of the fact
   * that edge_end(n) is basically prefix_sum[n] (if a prefix sum existed +
   * if prefix_sum[0] = number of edges in node 0).
   *
   * ONLY USE IF GRAPH HAS BEEN LOADED
   *
   * @param n Index into edge prefix sum
   * @returns The value that would be located at index n in an edge prefix sum
   * array
   */
  uint64_t operator[](uint64_t n) { return *(edge_end(n)); }

  template <typename EdgeNumFnTy, typename EdgeDstFnTy, typename EdgeDataFnTy>
  LC_CSR_64_Graph(uint64_t _numNodes, uint64_t _numEdges, EdgeNumFnTy edgeNum,
                  EdgeDstFnTy _edgeDst, EdgeDataFnTy _edgeData)
      : numNodes(_numNodes), numEdges(_numEdges) {
    if (UseNumaAlloc) {
      //! [numaallocex]
      nodeData.allocateBlocked(numNodes);
      edgeIndData.allocateBlocked(numNodes);
      edgeDst.allocateBlocked(numEdges);
      edgeData.allocateBlocked(numEdges);
      //! [numaallocex]
      this->outOfLineAllocateBlocked(numNodes);
    } else {
      nodeData.allocateInterleaved(numNodes);
      edgeIndData.allocateInterleaved(numNodes);
      edgeDst.allocateInterleaved(numEdges);
      edgeData.allocateInterleaved(numEdges);
      this->outOfLineAllocateInterleaved(numNodes);
    }
    for (size_t n = 0; n < numNodes; ++n) {
      nodeData.constructAt(n);
    }
    uint64_t cur = 0;
    for (size_t n = 0; n < numNodes; ++n) {
      cur += edgeNum(n);
      edgeIndData[n] = cur;
    }
    cur = 0;
    for (size_t n = 0; n < numNodes; ++n) {
      for (uint64_t e = 0, ee = edgeNum(n); e < ee; ++e) {
        if (EdgeData::has_value)
          edgeData.set(cur, _edgeData(n, e));
        edgeDst[cur] = _edgeDst(n, e);
        ++cur;
      }
    }
  }

  friend void swap(LC_CSR_64_Graph& lhs, LC_CSR_64_Graph& rhs) {
    swap(lhs.nodeData, rhs.nodeData);
    swap(lhs.edgeIndData, rhs.edgeIndData);
    swap(lhs.edgeDst, rhs.edgeDst);
    swap(lhs.edgeData, rhs.edgeData);
    std::swap(lhs.numNodes, rhs.numNodes);
    std::swap(lhs.numEdges, rhs.numEdges);
  }

  node_data_reference getData(GraphNode N,
                              MethodFlag mflag = MethodFlag::WRITE) {
    // galois::runtime::checkWrite(mflag, false);
    NodeInfo& NI = nodeData[N];
    acquireNode(N, mflag);
    return NI.getData();
  }

  edge_data_reference
  getEdgeData(edge_iterator ni,
              MethodFlag GALOIS_UNUSED(mflag) = MethodFlag::UNPROTECTED) {
    // galois::runtime::checkWrite(mflag, false);
    return edgeData[*ni];
  }

  GraphNode getEdgeDst(edge_iterator ni) { return edgeDst[*ni]; }

  size_t size() const { return numNodes; }
  size_t sizeEdges() const { return numEdges; }

  iterator begin() const { return iterator(0); }
  iterator end() const { return iterator(numNodes); }

  const_local_iterator local_begin() const {
    return const_local_iterator(this->localBegin(numNodes));
  }

  const_local_iterator local_end() const {
    return const_local_iterator(this->localEnd(numNodes));
  }

  local_iterator local_begin() {
    return local_iterator(this->localBegin(numNodes));
  }

  local_iterator local_end() {
    return local_iterator(this->localEnd(numNodes));
  }

  edge_iterator edge_begin(GraphNode N, MethodFlag mflag = MethodFlag::WRITE) {
    acquireNode(N, mflag);
    if (!HasNoLockable && galois::runtime::shouldLock(mflag)) {
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

  uint64_t getDegree(GraphNode N) const { return (raw_end(N) - raw_begin(N)); }

  edge_iterator findEdge(GraphNode N1, GraphNode N2) {
    return std::find_if(edge_begin(N1), edge_end(N1),
                        [=](edge_iterator e) { return getEdgeDst(e) == N2; });
  }

  edge_iterator findEdgeSortedByDst(GraphNode N1, GraphNode N2) {
    auto e = std::lower_bound(
        edge_begin(N1), edge_end(N1), N2,
        [=](edge_iterator e, GraphNode N) { return getEdgeDst(e) < N; });
    return (getEdgeDst(e) == N2) ? e : edge_end(N1);
  }

  runtime::iterable<NoDerefIterator<edge_iterator>>
  edges(GraphNode N, MethodFlag mflag = MethodFlag::WRITE) {
    return internal::make_no_deref_range(edge_begin(N, mflag),
                                         edge_end(N, mflag));
  }

  runtime::iterable<NoDerefIterator<edge_iterator>>
  out_edges(GraphNode N, MethodFlag mflag = MethodFlag::WRITE) {
    return edges(N, mflag);
  }

  /**
   * Sorts outgoing edges of a node. Comparison function is over EdgeTy.
   */
  template <typename CompTy>
  void sortEdgesByEdgeData(GraphNode N,
                           const CompTy& comp = std::less<EdgeTy>(),
                           MethodFlag mflag   = MethodFlag::WRITE) {
    acquireNode(N, mflag);
    std::sort(
        edge_sort_begin(N), edge_sort_end(N),
        internal::EdgeSortCompWrapper<EdgeSortValue<GraphNode, EdgeTy>, CompTy>(
            comp));
  }

  /**
   * Sorts outgoing edges of a node.
   * Comparison function is over <code>EdgeSortValue<EdgeTy></code>.
   */
  template <typename CompTy>
  void sortEdges(GraphNode N, const CompTy& comp,
                 MethodFlag mflag = MethodFlag::WRITE) {
    acquireNode(N, mflag);
    std::sort(edge_sort_begin(N), edge_sort_end(N), comp);
  }

  /**
   * Sorts outgoing edges of a node. Comparison is over getEdgeDst(e).
   */
  void sortEdgesByDst(GraphNode N, MethodFlag mflag = MethodFlag::WRITE) {
    acquireNode(N, mflag);
    typedef EdgeSortValue<GraphNode, EdgeTy> EdgeSortVal;
    std::sort(edge_sort_begin(N), edge_sort_end(N),
              [=](const EdgeSortVal& e1, const EdgeSortVal& e2) {
                return e1.dst < e2.dst;
              });
  }

  /**
   * Sorts all outgoing edges of all nodes in parallel. Comparison is over
   * getEdgeDst(e).
   */
  void sortAllEdgesByDst(MethodFlag mflag = MethodFlag::WRITE) {
    galois::do_all(
        galois::iterate(size_t{0}, this->size()),
        [=](GraphNode N) { this->sortEdgesByDst(N, mflag); },
        galois::no_stats(), galois::steal());
  }

  void allocateFrom(const FileGraph& graph) {
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

  void allocateFrom(uint64_t nNodes, uint64_t nEdges) {
    numNodes = nNodes;
    numEdges = nEdges;

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

  void destroyAndAllocateFrom(uint64_t nNodes, uint64_t nEdges) {
    numNodes = nNodes;
    numEdges = nEdges;

    deallocate();
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

  void constructNodes() {
#ifndef GALOIS_GRAPH_CONSTRUCT_SERIAL
    for (uint64_t x = 0; x < numNodes; ++x) {
      nodeData.constructAt(x);
      this->outOfLineConstructAt(x);
    }
#else
    galois::do_all(
        galois::iterate(UINT64_C(0), numNodes),
        [&](uint64_t x) {
          nodeData.constructAt(x);
          this->outOfLineConstructAt(x);
        },
        galois::no_stats(), galois::loopname("CONSTRUCT_NODES"));
#endif
  }

  void deallocate() {
    nodeData.destroy();
    nodeData.deallocate();

    edgeIndData.deallocate();
    edgeIndData.destroy();

    edgeDst.deallocate();
    edgeDst.destroy();

    edgeData.deallocate();
    edgeData.destroy();
  }

  void constructEdge(uint64_t e, uint64_t dst,
                     const typename EdgeData::value_type& val) {
    edgeData.set(e, val);
    edgeDst[e] = dst;
  }

  void constructEdge(uint64_t e, uint64_t dst) { edgeDst[e] = dst; }

  void fixEndEdge(uint64_t n, uint64_t e) { edgeIndData[n] = e; }

  /**
   * Perform an in-memory transpose of the graph, replacing the original
   * CSR to CSC
   */
  void transpose(const char* regionName = NULL) {
    galois::StatTimer timer("TIMER_GRAPH_TRANSPOSE", regionName);
    timer.start();

    EdgeDst edgeDst_old;
    EdgeData edgeData_new;
    EdgeIndData edgeIndData_old;
    EdgeIndData edgeIndData_temp;

    if (UseNumaAlloc) {
      edgeIndData_old.allocateBlocked(numNodes);
      edgeIndData_temp.allocateBlocked(numNodes);
      edgeDst_old.allocateBlocked(numEdges);
      edgeData_new.allocateBlocked(numEdges);
    } else {
      edgeIndData_old.allocateInterleaved(numNodes);
      edgeIndData_temp.allocateInterleaved(numNodes);
      edgeDst_old.allocateInterleaved(numEdges);
      edgeData_new.allocateInterleaved(numEdges);
    }

    // Copy old node->index location + initialize the temp array
    galois::do_all(
        galois::iterate(UINT64_C(0), numNodes),
        [&](uint64_t n) {
          edgeIndData_old[n]  = edgeIndData[n];
          edgeIndData_temp[n] = 0;
        },
        galois::no_stats(), galois::loopname("TRANSPOSE_EDGEINTDATA_COPY"));

    // get destination of edge, copy to array, and
    galois::do_all(
        galois::iterate(UINT64_C(0), numEdges),
        [&](uint64_t e) {
          auto dst       = edgeDst[e];
          edgeDst_old[e] = dst;
          // counting outgoing edges in the tranpose graph by
          // counting incoming edges in the original graph
          __sync_add_and_fetch(&edgeIndData_temp[dst], 1);
        },
        galois::no_stats(), galois::loopname("TRANSPOSE_EDGEINTDATA_INC"));

    // TODO is it worth doing parallel prefix sum?
    // prefix sum calculation of the edge index array
    for (uint64_t n = 1; n < numNodes; ++n) {
      edgeIndData_temp[n] += edgeIndData_temp[n - 1];
    }

    // copy over the new tranposed edge index data
    galois::do_all(
        galois::iterate(UINT64_C(0), numNodes),
        [&](uint64_t n) { edgeIndData[n] = edgeIndData_temp[n]; },
        galois::no_stats(), galois::loopname("TRANSPOSE_EDGEINTDATA_SET"));

    // edgeIndData_temp[i] will now hold number of edges that all nodes
    // before the ith node have
    if (numNodes >= 1) {
      edgeIndData_temp[0] = 0;
      galois::do_all(
          galois::iterate(UINT64_C(1), numNodes),
          [&](uint64_t n) { edgeIndData_temp[n] = edgeIndData[n - 1]; },
          galois::no_stats(), galois::loopname("TRANSPOSE_EDGEINTDATA_TEMP"));
    }

    galois::do_all(
        galois::iterate(UINT64_C(0), numNodes),
        [&](uint64_t src) {
          // e = start index into edge array for a particular node
          uint64_t e = (src == 0) ? 0 : edgeIndData_old[src - 1];

          // get all outgoing edges of a particular node in the
          // non-transpose and convert to incoming
          while (e < edgeIndData_old[src]) {
            // destination nodde
            auto dst = edgeDst_old[e];
            // location to save edge
            auto e_new = __sync_fetch_and_add(&(edgeIndData_temp[dst]), 1);
            // save src as destination
            edgeDst[e_new] = src;
            // copy edge data to "new" array
            edgeDataCopy(edgeData_new, edgeData, e_new, e);
            e++;
          }
        },
        galois::no_stats(), galois::loopname("TRANSPOSE_EDGEDST"));

    // if edge weights, then overwrite edgeData with new edge data
    if (EdgeData::has_value) {
      galois::do_all(
          galois::iterate(UINT64_C(0), numEdges),
          [&](uint64_t e) { edgeDataCopy(edgeData, edgeData_new, e, e); },
          galois::no_stats(), galois::loopname("TRANSPOSE_EDGEDATA_SET"));
    }

    timer.stop();
  }

  template <bool is_non_void = EdgeData::has_value>
  void edgeDataCopy(EdgeData& edgeData_new, EdgeData& edgeData, uint64_t e_new,
                    uint64_t e,
                    typename std::enable_if<is_non_void>::type* = 0) {
    edgeData_new[e_new] = edgeData[e];
  }

  template <bool is_non_void = EdgeData::has_value>
  void edgeDataCopy(EdgeData&, EdgeData&, uint64_t, uint64_t,
                    typename std::enable_if<!is_non_void>::type* = 0) {
    // does nothing
  }

  template <typename E                                            = EdgeTy,
            std::enable_if_t<!std::is_same<E, void>::value, int>* = nullptr>
  void constructFrom(FileGraph& graph, unsigned tid, unsigned total,
                     const bool readUnweighted = false) {
    // at this point memory should already be allocated
    auto r =
        graph
            .divideByNode(
                NodeData::size_of::value + EdgeIndData::size_of::value +
                    LC_CSR_64_Graph::size_of_out_of_line::value,
                EdgeDst::size_of::value + EdgeData::size_of::value, tid, total)
            .first;

    this->setLocalRange(*r.first, *r.second);

    for (FileGraph::iterator ii = r.first, ei = r.second; ii != ei; ++ii) {
      nodeData.constructAt(*ii);
      edgeIndData[*ii] = *graph.edge_end(*ii);

      this->outOfLineConstructAt(*ii);

      for (FileGraph::edge_iterator nn = graph.edge_begin(*ii),
                                    en = graph.edge_end(*ii);
           nn != en; ++nn) {
        if (readUnweighted) {
          edgeData.set(*nn, {});
        } else {
          constructEdgeValue(graph, nn);
        }
        edgeDst[*nn] = graph.getEdgeDst(nn);
      }
    }
  }

  template <typename E                                           = EdgeTy,
            std::enable_if_t<std::is_same<E, void>::value, int>* = nullptr>
  void constructFrom(FileGraph& graph, unsigned tid, unsigned total,
                     const bool GALOIS_UNUSED(readUnweighted) = false) {
    // at this point memory should already be allocated
    auto r =
        graph
            .divideByNode(
                NodeData::size_of::value + EdgeIndData::size_of::value +
                    LC_CSR_64_Graph::size_of_out_of_line::value,
                EdgeDst::size_of::value + EdgeData::size_of::value, tid, total)
            .first;

    this->setLocalRange(*r.first, *r.second);

    for (FileGraph::iterator ii = r.first, ei = r.second; ii != ei; ++ii) {
      nodeData.constructAt(*ii);
      edgeIndData[*ii] = *graph.edge_end(*ii);

      this->outOfLineConstructAt(*ii);

      for (FileGraph::edge_iterator nn = graph.edge_begin(*ii),
                                    en = graph.edge_end(*ii);
           nn != en; ++nn) {
        constructEdgeValue(graph, nn);
        edgeDst[*nn] = graph.getEdgeDst(nn);
      }
    }
  }

  /**
   * Returns the reference to the edgeIndData LargeArray
   * (a prefix sum of edges)
   *
   * @returns reference to LargeArray edgeIndData
   */
  const EdgeIndData& getEdgePrefixSum() const { return edgeIndData; }

  auto divideByNode(size_t nodeSize, size_t edgeSize, size_t id, size_t total) {
    return galois::graphs::divideNodesBinarySearch(
        numNodes, numEdges, nodeSize, edgeSize, id, total, edgeIndData);
  }
  /**
   *
   * custom allocator for vector<vector<>>
   * Adding for Louvain clustering
   * TODO: Find better way to do this
   */
  void constructFrom(uint64_t numNodes, uint64_t numEdges,
                     std::vector<uint64_t>& prefix_sum,
                     std::vector<std::vector<uint64_t>>& edges_id,
                     std::vector<std::vector<EdgeTy>>& edges_data) {
    // allocateFrom(numNodes, numEdges);
    /*
     * Deallocate if reusing the graph
     */
    destroyAndAllocateFrom(numNodes, numEdges);
    constructNodes();

    galois::do_all(galois::iterate((uint64_t)0, numNodes),
                   [&](uint64_t n) { edgeIndData[n] = prefix_sum[n]; });

    galois::do_all(galois::iterate((uint64_t)0, numNodes), [&](uint64_t n) {
      if (n == 0) {
        if (edgeIndData[n] > 0) {
          std::copy(edges_id[n].begin(), edges_id[n].end(), edgeDst.begin());
          std::copy(edges_data[n].begin(), edges_data[n].end(),
                    edgeData.begin());
        }
      } else {
        if (edgeIndData[n] - edgeIndData[n - 1] > 0) {
          std::copy(edges_id[n].begin(), edges_id[n].end(),
                    edgeDst.begin() + edgeIndData[n - 1]);
          std::copy(edges_data[n].begin(), edges_data[n].end(),
                    edgeData.begin() + edgeIndData[n - 1]);
        }
      }
    });

    initializeLocalRanges();
  }
  void constructFrom(
      uint64_t numNodes, uint64_t numEdges, std::vector<uint64_t>& prefix_sum,
      galois::gstl::Vector<galois::PODResizeableArray<uint64_t>>& edges_id,
      std::vector<std::vector<EdgeTy>>& edges_data) {
    allocateFrom(numNodes, numEdges);
    constructNodes();

    galois::do_all(galois::iterate((uint64_t)0, numNodes),
                   [&](uint64_t n) { edgeIndData[n] = prefix_sum[n]; });

    galois::do_all(galois::iterate((uint64_t)0, numNodes), [&](uint64_t n) {
      if (n == 0) {
        if (edgeIndData[n] > 0) {
          std::copy(edges_id[n].begin(), edges_id[n].end(), edgeDst.begin());
          std::copy(edges_data[n].begin(), edges_data[n].end(),
                    edgeData.begin());
        }
      } else {
        if (edgeIndData[n] - edgeIndData[n - 1] > 0) {
          std::copy(edges_id[n].begin(), edges_id[n].end(),
                    edgeDst.begin() + edgeIndData[n - 1]);
          std::copy(edges_data[n].begin(), edges_data[n].end(),
                    edgeData.begin() + edgeIndData[n - 1]);
        }
      }
    });

    initializeLocalRanges();
  }

  /**
   * Reads the GR files directly into in-memory
   * data-structures of LC_CSR graphs using freads.
   *
   * Edge is not void.
   *
   */
  template <
      typename U                                                      = void,
      typename std::enable_if<!std::is_void<EdgeTy>::value, U>::type* = nullptr>
  void readGraphFromGRFile(const std::string& filename) {
    std::ifstream graphFile(filename.c_str());
    if (!graphFile.is_open()) {
      GALOIS_DIE("failed to open file");
    }
    uint64_t header[4];
    graphFile.read(reinterpret_cast<char*>(header), sizeof(uint64_t) * 4);
    uint64_t version = header[0];
    numNodes         = header[2];
    numEdges         = header[3];
    galois::gPrint("Number of Nodes: ", numNodes,
                   ", Number of Edges: ", numEdges, "\n");
    allocateFrom(numNodes, numEdges);
    constructNodes();
    /**
     * Load outIndex array
     **/
    assert(edgeIndData.data());
    if (!edgeIndData.data()) {
      GALOIS_DIE("out of memory");
    }

    // start position to read index data
    uint64_t readPosition = (4 * sizeof(uint64_t));
    graphFile.seekg(readPosition);
    graphFile.read(reinterpret_cast<char*>(edgeIndData.data()),
                   sizeof(uint64_t) * numNodes);
    /**
     * Load edgeDst array
     **/
    assert(edgeDst.data());
    if (!edgeDst.data()) {
      GALOIS_DIE("out of memory");
    }

    readPosition = ((4 + numNodes) * sizeof(uint64_t));
    graphFile.seekg(readPosition);
    if (version == 1) {
      graphFile.read(reinterpret_cast<char*>(edgeDst.data()),
                     sizeof(uint32_t) * numEdges);
      readPosition =
          ((4 + numNodes) * sizeof(uint64_t) + numEdges * sizeof(uint32_t));
      // version 1 padding TODO make version agnostic
      if (numEdges % 2) {
        readPosition += sizeof(uint32_t);
      }
    } else if (version == 2) {
      graphFile.read(reinterpret_cast<char*>(edgeDst.data()),
                     sizeof(uint64_t) * numEdges);
      readPosition =
          ((4 + numNodes) * sizeof(uint64_t) + numEdges * sizeof(uint64_t));
      if (numEdges % 2) {
        readPosition += sizeof(uint64_t);
      }
    } else {
      GALOIS_DIE("unknown file version: ", version);
    }
    /**
     * Load edge data array
     **/
    assert(edgeData.data());
    if (!edgeData.data()) {
      GALOIS_DIE("out of memory");
    }
    graphFile.seekg(readPosition);
    graphFile.read(reinterpret_cast<char*>(edgeData.data()),
                   sizeof(EdgeTy) * numEdges);

    initializeLocalRanges();
    graphFile.close();
  }

  /**
   * Reads the GR files directly into in-memory
   * data-structures of LC_CSR graphs using freads.
   *
   * Edge is void.
   *
   */
  template <
      typename U                                                     = void,
      typename std::enable_if<std::is_void<EdgeTy>::value, U>::type* = nullptr>
  void readGraphFromGRFile(const std::string& filename) {
    std::ifstream graphFile(filename.c_str());
    if (!graphFile.is_open()) {
      GALOIS_DIE("failed to open file");
    }
    uint64_t header[4];
    graphFile.read(reinterpret_cast<char*>(header), sizeof(uint64_t) * 4);
    uint64_t version = header[0];
    numNodes         = header[2];
    numEdges         = header[3];
    galois::gPrint("Number of Nodes: ", numNodes,
                   ", Number of Edges: ", numEdges, "\n");
    allocateFrom(numNodes, numEdges);
    constructNodes();
    /**
     * Load outIndex array
     **/
    assert(edgeIndData.data());
    if (!edgeIndData.data()) {
      GALOIS_DIE("out of memory");
    }
    // start position to read index data
    uint64_t readPosition = (4 * sizeof(uint64_t));
    graphFile.seekg(readPosition);
    graphFile.read(reinterpret_cast<char*>(edgeIndData.data()),
                   sizeof(uint64_t) * numNodes);
    /**
     * Load edgeDst array
     **/
    assert(edgeDst.data());
    if (!edgeDst.data()) {
      GALOIS_DIE("out of memory");
    }
    readPosition = ((4 + numNodes) * sizeof(uint64_t));
    graphFile.seekg(readPosition);
    if (version == 1) {
      graphFile.read(reinterpret_cast<char*>(edgeDst.data()),
                     sizeof(uint32_t) * numEdges);
    } else if (version == 2) {
      graphFile.read(reinterpret_cast<char*>(edgeDst.data()),
                     sizeof(uint64_t) * numEdges);
    } else {
      GALOIS_DIE("unknown file version: ", version);
    }

    initializeLocalRanges();
    graphFile.close();
  }

  /**
   * Given a manually created graph, initialize the local ranges on this graph
   * so that threads can iterate over a balanced number of vertices.
   */
  void initializeLocalRanges() {
    galois::on_each([&](unsigned tid, unsigned total) {
      auto r = divideByNode(0, 1, tid, total).first;
      this->setLocalRange(*r.first, *r.second);
    });
  }
};

} // namespace galois::graphs

#endif
