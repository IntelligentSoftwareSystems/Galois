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

#ifndef GALOIS_GRAPHS_LS_LC_CSR_64_GRAPH_H
#define GALOIS_GRAPHS_LS_LC_CSR_64_GRAPH_H

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
#include "galois/PrefixSum.h"

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
class LS_LC_CSR_64_Graph :
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
    typedef LS_LC_CSR_64_Graph type;
  };

  template <typename _node_data>
  struct with_node_data {
    typedef LS_LC_CSR_64_Graph<_node_data, EdgeTy, HasNoLockable, UseNumaAlloc,
                               HasOutOfLineLockable, FileEdgeTy>
        type;
  };

  template <typename _edge_data>
  struct with_edge_data {
    typedef LS_LC_CSR_64_Graph<NodeTy, _edge_data, HasNoLockable, UseNumaAlloc,
                               HasOutOfLineLockable, FileEdgeTy>
        type;
  };

  template <typename _file_edge_data>
  struct with_file_edge_data {
    typedef LS_LC_CSR_64_Graph<NodeTy, EdgeTy, HasNoLockable, UseNumaAlloc,
                               HasOutOfLineLockable, _file_edge_data>
        type;
  };

  //! If true, do not use abstract locks in graph
  template <bool _has_no_lockable>
  struct with_no_lockable {
    typedef LS_LC_CSR_64_Graph<NodeTy, EdgeTy, _has_no_lockable, UseNumaAlloc,
                               HasOutOfLineLockable, FileEdgeTy>
        type;
  };
  template <bool _has_no_lockable>
  using _with_no_lockable =
      LS_LC_CSR_64_Graph<NodeTy, EdgeTy, _has_no_lockable, UseNumaAlloc,
                         HasOutOfLineLockable, FileEdgeTy>;

  //! If true, use NUMA-aware graph allocation; otherwise, use NUMA interleaved
  //! allocation.
  template <bool _use_numa_alloc>
  struct with_numa_alloc {
    typedef LS_LC_CSR_64_Graph<NodeTy, EdgeTy, HasNoLockable, _use_numa_alloc,
                               HasOutOfLineLockable, FileEdgeTy>
        type;
  };
  template <bool _use_numa_alloc>
  using _with_numa_alloc =
      LS_LC_CSR_64_Graph<NodeTy, EdgeTy, HasNoLockable, _use_numa_alloc,
                         HasOutOfLineLockable, FileEdgeTy>;

  //! If true, store abstract locks separate from nodes
  template <bool _has_out_of_line_lockable>
  struct with_out_of_line_lockable {
    typedef LS_LC_CSR_64_Graph<NodeTy, EdgeTy, HasNoLockable, UseNumaAlloc,
                               _has_out_of_line_lockable, FileEdgeTy>
        type;
  };

  typedef read_default_graph_tag read_tag;

protected:
  enum VertexState : uint16_t {
    UNLK = 0x0 << 0,
    LOCK = 0x1 << 0,
    TOMB = 0x1 << 1,
    UMAX = 0x1 << 2
  };

  constexpr uint64_t mask(uint8_t mask, uint8_t shift) { return mask << shift; }
  constexpr uint64_t lower(uint8_t num) { return (1 << num) - 1; }

  // Pack things in the same order of VertexState
  template <typename T>
  struct __attribute__((packed)) PackedVal {
    VertexState get_vertex_state(uint64_t v) const {
      return (VertexState)(v >> 48);
    }
    uint64_t get_raw_value(uint64_t v) const { return v & lower(48); }
    uint16_t get_flags_unlock(uint16_t f) const { return f & (lower(15) << 1); }
    uint16_t get_flags_untomb(uint16_t f) const {
      return f & (lower(14) << 2 | 0x1);
    }

    volatile uint16_t flags : 16;
    uint64_t value : 48;

    PackedVal(T t)
        : flags(get_vertex_state((uint64_t)t)),
          value(get_raw_value((uint64_t)t)) {}

    inline VertexState try_lock() {
      uint16_t f = __atomic_load_2(this, __ATOMIC_RELAXED);
      bool b     = false;
      if (!(f & LOCK))
        b = __atomic_compare_exchange_2(this, &f, f | LOCK, true,
                                        __ATOMIC_ACQUIRE, __ATOMIC_RELAXED);
      return (VertexState)((b ? UNLK : LOCK) | get_flags_unlock(f));
    }

    // Make an explicit function that returns tombstone and locks
    inline bool lock() {
      uint64_t ret;
      VertexState s;
      do {
        s = this->try_lock();
      } while (s & LOCK);
      return !(s & TOMB);
    }

    inline void unlock() {
      uint64_t f = flags;
      __atomic_store_2(this, f & (~LOCK), __ATOMIC_RELEASE);
    }

    inline void set_value(T p) {
      if ((uint64_t)p == UINT64_MAX) {
        flags |= UMAX;
      } else {
        value = get_raw_value(p);
      }
    }

    inline T get_value() { return (flags & UMAX) ? (T)UINT64_MAX : (T)value; }

    inline void unset_tomb() { flags = flags & (~TOMB); }

    inline void set_tomb() { flags = flags | TOMB; }

    inline bool is_tomb() { return flags & TOMB; }

    inline bool atomic_is_tomb() {
      return __atomic_load_2(this, __ATOMIC_RELAXED) & TOMB;
    }

    inline PackedVal<T>& operator=(const T& val) {
      this.set_value(val);
      return *this;
    }
  };

  struct EdgeInd {
    uint64_t first;
    uint64_t second;
    operator uint64_t() const { return second; }
    uint64_t operator++() { return ++second; }
    uint64_t operator--() { return --second; }
    uint64_t operator+=(uint64_t t) { return (second += t); }
  };

  typedef LargeArray<EdgeTy> EdgeData;
  typedef LargeArray<uint64_t> EdgeDst;
  typedef internal::NodeInfoBaseTypes<NodeTy,
                                      !HasNoLockable && !HasOutOfLineLockable>
      NodeInfoTypes;
  typedef internal::NodeInfoBase<NodeTy,
                                 !HasNoLockable && !HasOutOfLineLockable>
      NodeInfo;
  typedef LargeArray<EdgeInd> EdgeIndData;
  typedef LargeArray<NodeInfo> NodeData;

public:
  typedef uint64_t GraphNode;
  typedef EdgeTy edge_data_type;
  typedef FileEdgeTy file_edge_data_type;
  typedef NodeTy node_data_type;
  typedef typename EdgeData::reference edge_data_reference;
  typedef typename NodeInfoTypes::reference node_data_reference;
  using edge_iterator = boost::counting_iterator<uint64_t>;
  using iterator      = boost::counting_iterator<typename EdgeDst::value_type>;
  typedef iterator const_iterator;
  typedef iterator local_iterator;
  typedef iterator const_local_iterator;

protected:
  NodeData nodeData;
  EdgeIndData edgeIndData;
  EdgeDst edgeDst;
  EdgeData edgeData;
  EdgeDst prefixSumCache;

  static uint64_t transmute(const EdgeInd& p) { return p.second - p.first; };
  static uint64_t scan_op(const EdgeInd& p, const EdgeDst::value_type& l) {
    return p.second - p.first + l;
  };
  static uint64_t combiner(const EdgeDst::value_type& f,
                           const EdgeDst::value_type& s) {
    return f + s;
  };

  PrefixSum<EdgeInd, EdgeDst::value_type, transmute, scan_op, combiner,
            CacheLinePaddedArr>
      pfxsum{&edgeIndData[0], &prefixSumCache[0]};

  std::atomic<bool> prefixValid = false;
  std::atomic<uint64_t> numNodes;
  std::atomic<uint64_t> numEdges = 0;
  std::atomic<uint64_t> edgeEnd  = 0;

  uint64_t maxNodes = ((uint64_t)1) << 30;
  uint64_t maxEdges = ((uint64_t)1) << 32;

  typedef internal::EdgeSortIterator<GraphNode, uint64_t, EdgeDst, EdgeData>
      edge_sort_iterator;

  edge_iterator raw_begin(GraphNode N) const {
    return edge_iterator(edgeIndData[N].first);
  }

  edge_iterator raw_end(GraphNode N) const {
    return edge_iterator(edgeIndData[N].second);
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

  /**
  template <bool _A1 = HasNoLockable, bool _A2 = HasOutOfLineLockable>
  void releaseNode(GraphNode N,
                   typename std::enable_if<!_A1 && !_A2>::type* = 0) {
    galois::runtime::release(&nodeData[N]);
  }

  template <bool _A1 = HasOutOfLineLockable, bool _A2 = HasNoLockable>
  void releaseNode(GraphNode N,
                   typename std::enable_if<_A1 && !_A2>::type* = 0) {
    this->outOfLineRelease(getId(N));
  }

  template <bool _A1 = HasOutOfLineLockable, bool _A2 = HasNoLockable>
  void releaseNode(GraphNode,
                   typename std::enable_if<_A2>::type* = 0) {}
  */

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

  size_t getId(GraphNode N) { return N; }

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
  LS_LC_CSR_64_Graph(LS_LC_CSR_64_Graph&& rhs) = default;

  LS_LC_CSR_64_Graph() = default;

  LS_LC_CSR_64_Graph& operator=(LS_LC_CSR_64_Graph&&) = default;

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

  void resetPrefixSum() {
    pfxsum.src = &edgeIndData[0];
    pfxsum.dst = &prefixSumCache[0];
  }
  // Compute the prefix sum using the two level method
  void computePrefixSum() {
    pfxsum.computePrefixSum(numNodes);
    prefixValid = true;
  }

  /**
   * DO NOT USE WHILE MODIFYING THE GRAPH!
   * ONLY USE IF GRAPH HAS BEEN LOADED
   *
   * @param n Index into edge prefix sum
   * @returns The value that would be located at index n in an edge prefix sum
   * array
   */
  uint64_t operator[](uint64_t n) {
    if (!prefixValid)
      computePrefixSum();
    return prefixSumCache[n];
  }

  template <typename EdgeNumFnTy, typename EdgeDstFnTy, typename EdgeDataFnTy>
  LS_LC_CSR_64_Graph(uint64_t _numNodes, uint64_t _numEdges,
                     EdgeNumFnTy edgeNum, EdgeDstFnTy _edgeDst,
                     EdgeDataFnTy _edgeData)
      : numNodes(_numNodes), numEdges(_numEdges), edgeEnd(_numEdges) {
    assert(numNodes <= maxNodes);
    assert(numEdges <= maxEdges);
    if (UseNumaAlloc) {
      //! [numaallocex]
      nodeData.allocateBlocked(maxNodes);
      edgeIndData.allocateBlocked(maxNodes);
      edgeDst.allocateBlocked(maxEdges);
      edgeData.allocateBlocked(maxEdges);
      prefixSumCache.allocateBlocked(maxNodes);
      //! [numaallocex]
      this->outOfLineAllocateBlocked(maxNodes);
    } else {
      nodeData.allocateInterleaved(maxNodes);
      edgeIndData.allocateInterleaved(maxNodes);
      edgeDst.allocateInterleaved(maxEdges);
      edgeData.allocateInterleaved(maxEdges);
      prefixSumCache.allocateInterleaved(maxNodes);
      this->outOfLineAllocateInterleaved(maxNodes);
    }
    resetPrefixSum();
    for (size_t n = 0; n < numNodes; ++n) {
      nodeData.constructAt(n);
    }
    uint64_t cur = 0;
    for (size_t n = 0; n < numNodes; ++n) {
      edgeIndData[n].first = cur;
      cur += edgeNum(n);
      edgeIndData[n].second = cur;
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

  /* dangerous parallel constructor, call this outside a galois kernel */
  template <typename EdgeNumFnTy, typename EdgeDstFnTy, typename EdgeDataFnTy>
  LS_LC_CSR_64_Graph(bool setEdgeVals, uint64_t _numNodes, uint64_t _numEdges,
                     EdgeNumFnTy edgeNum, EdgeDstFnTy _edgeDst,
                     EdgeDataFnTy _edgeData)
      : numNodes(_numNodes), numEdges(_numEdges), edgeEnd(_numEdges) {
    assert(numNodes <= maxNodes);
    assert(numEdges <= maxEdges);
    if (UseNumaAlloc) {
      //! [numaallocex]
      nodeData.allocateBlocked(maxNodes);
      edgeIndData.allocateBlocked(maxNodes);
      edgeDst.allocateBlocked(maxEdges);
      edgeData.allocateBlocked(maxEdges);
      prefixSumCache.allocateBlocked(maxNodes);
      //! [numaallocex]
      this->outOfLineAllocateBlocked(maxNodes);
    } else {
      nodeData.allocateInterleaved(maxNodes);
      edgeIndData.allocateInterleaved(maxNodes);
      edgeDst.allocateInterleaved(maxEdges);
      edgeData.allocateInterleaved(maxEdges);
      prefixSumCache.allocateInterleaved(maxNodes);
      this->outOfLineAllocateInterleaved(maxNodes);
    }
    resetPrefixSum();
    galois::do_all(
        galois::iterate((uint64_t)0, _numNodes),
        [&](uint64_t n) { nodeData.constructAt(n); }, galois::steal());

    galois::do_all(
        galois::iterate((uint64_t)0, _numNodes),
        [&](uint64_t n) {
          addEdgesUnSort(setEdgeVals, n, _edgeDst(n), _edgeData(n), edgeNum(n));
        },
        galois::steal());
  }

  /**
   * Add edges into the graph
   *
   * @param setEdgeVals if true, will set edges data
   * @param src source node of edges to add
   * @param dst array of edges dst
   * @param dst_data array of dst nodes data
   * @param num_dst number of dst these edges has
   * @param keep_size if true, the number of edges in the graph are not
   * increment, by default is false
   */
  template <typename T>
  void addEdgesUnSort(bool setEdgeVals, GraphNode src, EdgeDst::value_type* dst,
                      T* dst_data, uint64_t num_dst, bool keep_size = false) {
    acquireNode(src, galois::MethodFlag::WRITE);
    auto orig_deg = getDegree(src);
    auto ee = edgeEnd.fetch_add(num_dst + orig_deg, std::memory_order_relaxed);

    auto edgeStart = ee;
    auto orig_itr  = edge_begin(src);

    std::memcpy(&edgeDst[edgeStart], &edgeDst[*orig_itr],
                sizeof(EdgeDst::value_type) * orig_deg);
    std::memcpy(&edgeDst[edgeStart + orig_deg], dst,
                sizeof(EdgeDst::value_type) * num_dst);

    if (EdgeData::has_value && setEdgeVals) {
      for (uint64_t i = 0; i < orig_deg; i++) {
        edgeData.set(edgeStart + i, edgeData[*orig_itr]);
      }
      for (uint64_t i = 0; i < num_dst; i++) {
        edgeData.set(edgeStart + orig_deg + i, dst_data[i]);
      }
    }

    edgeIndData[src].first  = edgeStart;
    edgeIndData[src].second = edgeStart + num_dst + orig_deg;

    if (!keep_size) {
      numEdges.fetch_add(num_dst, std::memory_order_relaxed);
    }
    prefixValid = false;
  }

  void addEdgeSort(const uint64_t src, const uint64_t dst) {
    acquireNode(src, galois::MethodFlag::WRITE);
    auto orig_deg  = getDegree(src);
    auto ee        = edgeEnd.fetch_add(1 + orig_deg, std::memory_order_relaxed);
    auto edgeStart = ee;
    auto edgePlace = ee;
    auto orig_itr  = edge_begin(src);
    auto orig_end  = edge_end(src);
    bool dst_insert = false;

    uint64_t orig_dst;
    while (orig_itr != orig_end || !dst_insert) {
      if (dst_insert || (orig_dst = getEdgeDst(orig_itr)) < dst) {
        edgeDst[edgePlace] = orig_dst;
        orig_itr++;
      } else if (orig_itr == orig_end ||
                 dst < (orig_dst = getEdgeDst(orig_itr))) {
        edgeDst[edgePlace] = dst;
        dst_insert         = true;
      } else {
        edgeDst[edgePlace] = dst;
        dst_insert         = true;
        orig_itr++;
      }
      edgePlace++;
    }

    edgeIndData[src].first  = edgeStart;
    edgeIndData[src].second = edgePlace;
    numEdges.fetch_add(edgePlace - edgeStart - orig_deg,
                       std::memory_order_relaxed);
    prefixValid = false;
  }

  template <typename PQ>
  void addEdges(uint64_t src, PQ& dst) {
    acquireNode(src, galois::MethodFlag::WRITE);
    auto orig_deg = getDegree(src);
    auto num_dst  = dst.size();
    auto ee = edgeEnd.fetch_add(num_dst + orig_deg, std::memory_order_relaxed);
    auto edgeStart = ee;
    auto edgePlace = ee;
    auto orig_itr  = edge_begin(src);
    auto orig_end  = edge_end(src);

    bool empty = dst.empty();
    while (orig_itr != orig_end || !empty) {

      auto orig_dst = getEdgeDst(orig_itr);
      if (orig_itr != orig_end && (empty || orig_dst < dst.top())) {
        edgeDst[edgePlace] = orig_dst;
        /*
        if (EdgeData::has_value)
        {
          edgeData.set(edgePlace, getEdgeData(orig_itr));
        }
        */
        orig_itr++;
      } else if (orig_itr == orig_end || dst.top() < orig_dst) {
        edgeDst[edgePlace] = dst.top();
        /*
        if(EdgeData::has_value)
          edgeData.set(edgePlace, *dst_data);

        dst_data++;
        */
        dst.pop();
      } else {
        edgeDst[edgePlace] = dst.top();
        /*
        if(EdgeData::has_value)
          edgeData.set(edgePlace, *dst_data);

        dst_data++;
        */
        dst.pop();
        orig_itr++;
      }
      edgePlace++;
      empty = dst.empty();
    }

    edgeIndData[src].first  = edgeStart;
    edgeIndData[src].second = edgePlace;
    numEdges.fetch_add(edgePlace - edgeStart - orig_deg,
                       std::memory_order_relaxed);
    prefixValid = false;
    // releaseNode(src);
  }

  template <typename PTM>
  void insertEdgesSerially(uint64_t src, const PTM& dst) {
    acquireNode(src, galois::MethodFlag::WRITE);
    auto orig_deg    = getDegree(src);
    uint64_t num_dst = 0;
    for (uint64_t t = 0; t < dst.numRows(); t++) {
      const auto& map = dst.get(t);
      if (auto search = map.find(src); search != map.end()) {
        num_dst += search->second.size();
      }
    }

    auto ee = edgeEnd.fetch_add(num_dst + orig_deg, std::memory_order_relaxed);
    auto edgeStart = ee;
    auto edgePlace = ee;
    auto orig_itr  = edgeIndData[src].first;
    auto orig_end  = edgeIndData[src].second;

    std::memcpy(&edgeDst[edgePlace], &edgeDst[orig_itr],
                sizeof(EdgeDst::value_type) * orig_deg);
    edgePlace += orig_deg;

    uint64_t i = 0;

    for (uint64_t t = 0; t < dst.numRows(); t++) {
      auto& map = dst.get(t);
      if (auto search = map.find(src); search != map.end()) {
        const auto& stack = search->second;
        for (auto it = stack.begin(); it != stack.end(); it++, i++) {
          edgeDst[edgePlace + i] = *it;
        }
      }
    }

    assert(i == num_dst);

    edgeIndData[src].first  = edgeStart;
    edgeIndData[src].second = edgePlace + num_dst;
    numEdges.fetch_add(num_dst, std::memory_order_relaxed);
    prefixValid = false;
  }

  template <typename Cont>
  void insertEdgesSerially(uint64_t src, uint64_t dst_sz, uint64_t start_index,
                           const Cont& cont) {
    acquireNode(src, galois::MethodFlag::WRITE);
    auto orig_deg = getDegree(src);

    auto ee = edgeEnd.fetch_add(orig_deg + dst_sz, std::memory_order_relaxed);
    auto edgeStart = ee;
    auto edgePlace = ee;
    auto orig_itr  = edgeIndData[src].first;
    auto orig_end  = edgeIndData[src].second;

    auto dst_end   = cont.end();
    auto dst_start = cont.begin() + start_index;
    auto dst_left  = dst_sz;

    std::memcpy(&edgeDst[edgePlace], &edgeDst[orig_itr],
                sizeof(EdgeDst::value_type) * orig_deg);
    edgePlace += orig_deg;

    while (dst_left != 0) {
      while (dst_start->first != src)
        dst_start++;
      edgeDst[edgePlace] = dst_start->second;
      edgePlace++;
      dst_start++;
      dst_left--;
    }

    edgeIndData[src].first  = edgeStart;
    edgeIndData[src].second = edgePlace;
    numEdges.fetch_add(dst_sz, std::memory_order_relaxed);
    prefixValid = false;
  }

  void sortVertexSerially(uint64_t src) {
    acquireNode(src, galois::MethodFlag::WRITE);
    auto orig_itr = edgeIndData[src].first;
    auto orig_end = edgeIndData[src].second;
    std::sort(&edgeDst[orig_itr], &edgeDst[orig_end],
              [](const EdgeDst::value_type& e0, const EdgeDst::value_type& e1) {
                return e0 < e1;
              });
  }

  template <typename PQ>
  void addEdges(PQ* edges) {
    for (uint64_t i = 0; i < numNodes; i++) {
      if (!edges[i].empty())
        addEdges<PQ>(i, edges[i]);
    }
  }

  friend void swap(LS_LC_CSR_64_Graph& lhs, LS_LC_CSR_64_Graph& rhs) {
    swap(lhs.nodeData, rhs.nodeData);
    swap(lhs.edgeIndData, rhs.edgeIndData);
    swap(lhs.edgeDst, rhs.edgeDst);
    swap(lhs.edgeData, rhs.edgeData);
    swap(lhs.pfxsum, rhs.pfxsum);
    swap(lhs.prefixSumCache, rhs.prefixSumCache);

    bool pv         = lhs.prefixValid;
    lhs.prefixValid = rhs.prefixValid;
    rhs.prefixValid = pv;

    uint64_t blah = lhs.numNodes;
    lhs.numNodes  = rhs.numNodes;
    rhs.numNodes  = blah;

    blah         = lhs.numEdges;
    lhs.numEdges = rhs.numEdges;
    rhs.numEdges = blah;
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
    numEdges = 0;
    edgeEnd  = 0;
    maxEdges = graph.sizeEdges();
    maxNodes = numNodes;

    if (UseNumaAlloc) {
      nodeData.allocateBlocked(numNodes);
      edgeIndData.allocateBlocked(numNodes);
      edgeDst.allocateBlocked(numEdges);
      edgeData.allocateBlocked(numEdges);
      pfxsum.allocateInterleaved(numNodes);
      this->outOfLineAllocateBlocked(numNodes);
    } else {
      nodeData.allocateInterleaved(numNodes);
      edgeIndData.allocateInterleaved(numNodes);
      edgeDst.allocateInterleaved(numEdges);
      edgeData.allocateInterleaved(numEdges);
      pfxsum.allocateInterleaved(numNodes);
      this->outOfLineAllocateInterleaved(numNodes);
    }
    resetPrefixSum();
  }

  void allocateFrom(uint64_t nNodes, uint64_t nEdges) {
    numNodes = nNodes;
    numEdges = 0;
    edgeEnd  = 0;
    maxEdges = nEdges;
    maxNodes = nNodes;

    if (UseNumaAlloc) {
      nodeData.allocateBlocked(maxNodes);
      edgeIndData.allocateBlocked(maxNodes);
      edgeDst.allocateBlocked(maxEdges);
      edgeData.allocateBlocked(maxEdges);
      prefixSumCache.allocateBlocked(maxNodes);
      this->outOfLineAllocateBlocked(maxNodes);
    } else {
      nodeData.allocateInterleaved(maxNodes);
      edgeIndData.allocateInterleaved(maxNodes);
      edgeDst.allocateInterleaved(maxEdges);
      edgeData.allocateInterleaved(maxEdges);
      prefixSumCache.allocateInterleaved(maxNodes);
      this->outOfLineAllocateInterleaved(maxNodes);
    }
    resetPrefixSum();
  }

  void destroyAndAllocateFrom(uint64_t nNodes, uint64_t nEdges) {
    numNodes = nNodes;
    numEdges = 0;
    edgeEnd  = 0;
    maxEdges = nEdges;
    maxNodes = nNodes;

    deallocate();
    if (UseNumaAlloc) {
      nodeData.allocateBlocked(numNodes);
      edgeIndData.allocateBlocked(numNodes);
      edgeDst.allocateBlocked(numEdges);
      edgeData.allocateBlocked(numEdges);
      prefixSumCache.allocateBlocked(numNodes);
      this->outOfLineAllocateBlocked(numNodes);
    } else {
      nodeData.allocateInterleaved(numNodes);
      edgeIndData.allocateInterleaved(numNodes);
      edgeDst.allocateInterleaved(numEdges);
      edgeData.allocateInterleaved(numEdges);
      prefixSumCache.allocateInterleaved(numNodes);
      this->outOfLineAllocateInterleaved(numNodes);
    }
    resetPrefixSum();
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

    prefixSumCache.deallocate();
    prefixSumCache.destroy();
  }

  void constructEdge(uint64_t e, uint64_t dst,
                     const typename EdgeData::value_type& val) {
    edgeData.set(e, val);
    edgeDst[e] = dst;
  }

  void constructEdge(uint64_t e, uint64_t dst) { edgeDst[e] = dst; }

  void fixEndEdge(uint64_t n, uint64_t e) { edgeIndData[n].second = e; }
  void fixStartEdge(uint64_t n, uint64_t e) { edgeIndData[n].first = e; }

  /**
   * Perform an in-memory transpose of the graph, replacing the original
   * CSR to CSC
   */
  template <bool ComputePFXSum = true>
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
      edgeDst_old.allocateBlocked(edgeEnd);
      edgeData_new.allocateBlocked(maxEdges);
    } else {
      edgeIndData_old.allocateInterleaved(numNodes);
      edgeIndData_temp.allocateInterleaved(numNodes);
      edgeDst_old.allocateInterleaved(edgeEnd);
      edgeData_new.allocateInterleaved(maxEdges);
    }

    uint64_t numNodes_temp = numNodes.load(std::memory_order_relaxed);
    uint64_t edgeEnd_temp  = edgeEnd.load(std::memory_order_relaxed);
    // Copy old node->index location + initialize the temp array
    galois::do_all(
        galois::iterate(UINT64_C(0), numNodes_temp),
        [&](uint64_t n) {
          edgeIndData_old[n]         = edgeIndData[n];
          edgeIndData_temp[n].first  = 0;
          edgeIndData_temp[n].second = 0;
        },
        galois::no_stats(), galois::loopname("TRANSPOSE_EDGEINTDATA_COPY"));

    // get destination of edge, copy to array, and
    galois::do_all(
        galois::iterate(UINT64_C(0), edgeEnd_temp),
        [&](uint64_t e) {
          auto dst       = edgeDst[e];
          edgeDst_old[e] = dst;
          // counting outgoing edges in the tranpose graph by
          // counting incoming edges in the original graph
          __sync_add_and_fetch(&edgeIndData_temp[dst].second, 1);
        },
        galois::no_stats(), galois::loopname("TRANSPOSE_EDGEINTDATA_INC"));

    // TODO is it worth doing parallel prefix sum?
    // prefix sum calculation of the edge index array
    edgeIndData_temp[0].first = 0;
    for (uint64_t n = 1; n < numNodes_temp; ++n) {
      edgeIndData_temp[n].second += edgeIndData_temp[n - 1].second;
      edgeIndData_temp[n].first = edgeIndData_temp[n - 1].second;
    }

    // copy over the new tranposed edge index data
    galois::do_all(
        galois::iterate(UINT64_C(0), numNodes_temp),
        [&](uint64_t n) { edgeIndData[n] = edgeIndData_temp[n]; },
        galois::no_stats(), galois::loopname("TRANSPOSE_EDGEINTDATA_SET"));

    /* AdityaAtulTewari edit: Elided since this was stored in above loop.
    // edgeIndData_temp[i] will now hold number of edges that all nodes
    // before the ith node have
    if (numNodes >= 1) {
      edgeIndData_temp[0] = 0;
      galois::do_all(
          galois::iterate(UINT64_C(1), numNodes),
          [&](uint64_t n) { edgeIndData_temp[n] = edgeIndData[n - 1]; },
          galois::no_stats(), galois::loopname("TRANSPOSE_EDGEINTDATA_TEMP"));
    }
    */

    galois::do_all(
        galois::iterate(UINT64_C(0), numNodes_temp),
        [&](uint64_t src) {
          // e = start index into edge array for a particular node
          uint64_t e = edgeIndData_old[src].first;

          // get all outgoing edges of a particular node in the
          // non-transpose and convert to incoming
          while (e < edgeIndData_old[src].second) {
            // destination nodde
            auto dst = edgeDst_old[e];
            // location to save edge
            auto e_new =
                __sync_fetch_and_add(&(edgeIndData_temp[dst].first), 1);
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
          galois::iterate(UINT64_C(0), edgeEnd_temp),
          [&](uint64_t e) { edgeDataCopy(edgeData, edgeData_new, e, e); },
          galois::no_stats(), galois::loopname("TRANSPOSE_EDGEDATA_SET"));
    }
    edgeEnd.store(numEdges, std::memory_order_relaxed);

    resetPrefixSum();
    if (ComputePFXSum) {
      computePrefixSum();
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
                    LS_LC_CSR_64_Graph::size_of_out_of_line::value,
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
    resetPrefixSum();
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
                    LS_LC_CSR_64_Graph::size_of_out_of_line::value,
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
    resetPrefixSum();
  }

  /**
   * Returns the reference to the edgeIndData LargeArray
   * (a prefix sum of edges)
   *
   * @returns reference to LargeArray prefixSumCache
   */
  const EdgeDst& getEdgePrefixSum() {
    if (!prefixValid)
      computePrefixSum();
    return prefixSumCache;
  }

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
                   [&](uint64_t n) { edgeIndData[n].second = prefix_sum[n]; });

    galois::do_all(galois::iterate((uint64_t)0, numNodes), [&](uint64_t n) {
      if (n == 0) {
        edgeIndData[n].first = 0;
        if (edgeIndData[n].second > 0) {
          std::copy(edges_id[n].begin(), edges_id[n].end(), edgeDst.begin());
          std::copy(edges_data[n].begin(), edges_data[n].end(),
                    edgeData.begin());
        }
      } else {
        edgeIndData[n].first = edgeIndData[n - 1].second;
        if (edgeIndData[n].second - edgeIndData[n].first > 0) {
          std::copy(edges_id[n].begin(), edges_id[n].end(),
                    edgeDst.begin() + edgeIndData[n].first);
          std::copy(edges_data[n].begin(), edges_data[n].end(),
                    edgeData.begin() + edgeIndData[n].second);
        }
      }
    });

    resetPrefixSum();
    initializeLocalRanges();
  }
  void constructFrom(
      uint64_t numNodes, uint64_t numEdges, std::vector<uint64_t>& prefix_sum,
      galois::gstl::Vector<galois::PODResizeableArray<uint64_t>>& edges_id,
      std::vector<std::vector<EdgeTy>>& edges_data) {
    allocateFrom(numNodes, numEdges);
    constructNodes();

    galois::do_all(galois::iterate((uint64_t)0, numNodes),
                   [&](uint64_t n) { edgeIndData[n].second = prefix_sum[n]; });

    galois::do_all(galois::iterate((uint64_t)0, numNodes), [&](uint64_t n) {
      if (n == 0) {
        edgeIndData[n].first = 0;
        if (edgeIndData[n].second > 0) {
          std::copy(edges_id[n].begin(), edges_id[n].end(), edgeDst.begin());
          std::copy(edges_data[n].begin(), edges_data[n].end(),
                    edgeData.begin());
        }
      } else {
        edgeIndData[n].first = edgeIndData[n - 1].second;
        if (edgeIndData[n].second - edgeIndData[n].first > 0) {
          std::copy(edges_id[n].begin(), edges_id[n].end(),
                    edgeDst.begin() + edgeIndData[n].first);
          std::copy(edges_data[n].begin(), edges_data[n].end(),
                    edgeData.begin() + edgeIndData[n].first);
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

// used to determine if a instance is this template
template <typename Type>
struct is_LS_LC_CSR_64_Graph : std::false_type {};

template <typename NodeTy, typename EdgeTy, bool HasNoLockable,
          bool UseNumaAlloc, bool HasOutOfLineLockable, typename FileEdgeTy>
struct is_LS_LC_CSR_64_Graph<
    LS_LC_CSR_64_Graph<NodeTy, EdgeTy, HasNoLockable, UseNumaAlloc,
                       HasOutOfLineLockable, FileEdgeTy>> : std::true_type {};

} // namespace galois::graphs

#endif
