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

#ifndef GALOIS_GRAPHS_OCGRAPH_H
#define GALOIS_GRAPHS_OCGRAPH_H

#include <string>
#include <type_traits>

#include <boost/iterator/counting_iterator.hpp>
#include <boost/utility.hpp>

#include "galois/config.h"
#include "galois/graphs/Details.h"
#include "galois/substrate/PageAlloc.h"
#include "galois/LazyObject.h"
#include "galois/LargeArray.h"
#include "galois/optional.h"

namespace galois {
namespace graphs {

/**
 * Binds the segment parameter of an out-of-core graph so that it can be used in
 * place of a non out-of-core graph.
 */
template <typename Graph>
class BindSegmentGraph : private boost::noncopyable {
  typedef typename Graph::segment_type segment_type;

  Graph& graph;
  segment_type segment;

public:
  explicit BindSegmentGraph(Graph& g) : graph(g) {}
  BindSegmentGraph(Graph& g, segment_type s) : graph(g), segment(s) {}

  void setSegment(const segment_type& s) { segment = s; }

  typedef typename Graph::GraphNode GraphNode;
  typedef typename Graph::edge_data_type edge_data_type;
  typedef typename Graph::node_data_type node_data_type;
  typedef typename Graph::edge_data_reference edge_data_reference;
  typedef typename Graph::node_data_reference node_data_reference;
  typedef typename Graph::edge_iterator edge_iterator;
  typedef typename Graph::in_edge_iterator in_edge_iterator;
  typedef typename Graph::iterator iterator;
  typedef typename Graph::const_iterator const_iterator;
  typedef typename Graph::local_iterator local_iterator;
  typedef typename Graph::const_local_iterator const_local_iterator;

  node_data_reference getData(GraphNode N,
                              MethodFlag mflag = MethodFlag::WRITE) {
    return graph.getData(N, mflag);
  }

  edge_data_reference getEdgeData(edge_iterator ni,
                                  MethodFlag mflag = MethodFlag::UNPROTECTED) {
    return graph.getEdgeData(segment, ni, mflag);
  }

  GraphNode getEdgeDst(edge_iterator ni) {
    return graph.getEdgeDst(segment, ni);
  }

  size_t size() const { return graph.size(); }
  size_t sizeEdges() const { return graph.sizeEdges(); }

  iterator begin() const { return graph.begin(); }
  iterator end() const { return graph.end(); }

  local_iterator local_begin() const { return graph.local_begin(); }
  local_iterator local_end() const { return graph.local_end(); }

  edge_iterator edge_begin(GraphNode N, MethodFlag mflag = MethodFlag::WRITE) {
    return graph.edge_begin(segment, N, mflag);
  }

  edge_iterator edge_end(GraphNode N, MethodFlag mflag = MethodFlag::WRITE) {
    return graph.edge_end(segment, N, mflag);
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

  edge_data_reference
  getInEdgeData(edge_iterator ni, MethodFlag mflag = MethodFlag::UNPROTECTED) {
    return graph.getInEdgeData(segment, ni, mflag);
  }

  GraphNode getInEdgeDst(in_edge_iterator ni) {
    return graph.getInEdgeDst(segment, ni);
  }

  in_edge_iterator in_edge_begin(GraphNode N,
                                 MethodFlag mflag = MethodFlag::WRITE) {
    return graph.in_edge_begin(segment, N, mflag);
  }

  in_edge_iterator in_edge_end(GraphNode N,
                               MethodFlag mflag = MethodFlag::WRITE) {
    return graph.in_edge_end(segment, N, mflag);
  }

  internal::InEdgesIterator<BindSegmentGraph>
  in_edges(GraphNode N, MethodFlag mflag = MethodFlag::WRITE) {
    return internal::InEdgesIterator<BindSegmentGraph>(*this, N, mflag);
  }

  size_t idFromNode(GraphNode N) { return graph.idFromNode(N); }

  GraphNode nodeFromId(size_t N) { return graph.nodeFromId(N); }
};

//! Like {@link FileGraph} but allows partial loading of the graph.
class OCFileGraph : private boost::noncopyable {
public:
  typedef uint32_t GraphNode;
  typedef boost::counting_iterator<uint32_t> iterator;
  typedef boost::counting_iterator<uint64_t> edge_iterator;
  typedef uint64_t* edge_offset_iterator;

  template <typename EdgeTy>
  struct EdgeReference {
    typedef typename LazyObject<EdgeTy>::reference type;
  };

private:
  struct PageSizeConf;

  class Block {
    friend class OCFileGraph;
    void* m_mapping;
    size_t m_length;
    char* m_data;
    size_t m_begin;
    size_t m_sizeof_data;

    void unload();
    void load(int fd, offset_t offset, size_t begin, size_t len,
              size_t sizeof_data);

  public:
    Block() : m_mapping(0) {}

    char* get(size_t index) const {
      char* p = m_data + (m_sizeof_data * (index - m_begin));
      assert(p < reinterpret_cast<char*>(m_mapping) + m_length);
      assert(m_mapping <= p);
      return p;
    }
  };

  struct Segment {
    Block outs;
    Block edgeData;
    bool loaded;

    Segment() : loaded(false) {}

    void unload() {
      outs.unload();
      edgeData.unload();
      loaded = false;
    }
  };

  void* masterMapping;
  int masterFD;
  size_t masterLength;
  uint64_t numEdges;
  uint64_t numNodes;
  uint64_t* outIdx;

public:
  typedef Segment segment_type;

  OCFileGraph()
      : masterMapping(0), masterFD(-1), numEdges(0), numNodes(0), outIdx(0) {}
  ~OCFileGraph();

  iterator begin() const { return iterator(0); }
  iterator end() const { return iterator(numNodes); }
  size_t size() const { return numNodes; }
  size_t sizeEdges() const { return numEdges; }
  edge_iterator edge_begin(GraphNode n) const {
    return edge_iterator(n == 0 ? 0 : outIdx[n - 1]);
  }
  edge_iterator edge_end(GraphNode n) const { return edge_iterator(outIdx[n]); }
  edge_offset_iterator edge_offset_begin() const { return outIdx; }
  edge_offset_iterator edge_offset_end() const { return outIdx + numNodes; }

  template <typename EdgeTy>
  typename EdgeReference<EdgeTy>::type getEdgeData(
      const segment_type& s, edge_iterator it,
      typename std::enable_if<!std::is_same<void, EdgeTy>::value>::type* = 0) {
    EdgeTy* p = reinterpret_cast<EdgeTy*>(s.edgeData.get(*it));
    return *p;
  }

  template <typename EdgeTy>
  typename EdgeReference<EdgeTy>::type getEdgeData(
      const segment_type&, edge_iterator,
      typename std::enable_if<std::is_same<void, EdgeTy>::value>::type* = 0) {
    return 0;
  }

  GraphNode getEdgeDst(const segment_type& s, edge_iterator it) {
    uint32_t* p = reinterpret_cast<uint32_t*>(s.outs.get(*it));
    return *p;
  }

  void unload(segment_type& s) {
    if (!s.loaded)
      return;

    s.outs.unload();
    s.edgeData.unload();
    s.loaded = false;
  }

  void load(segment_type& s, edge_iterator begin, edge_iterator end,
            size_t sizeof_data);

  void fromFile(const std::string& fname);
};

struct read_oc_immutable_edge_graph_tag {};

template <typename NodeTy, typename EdgeTy, bool HasNoLockable = false,
          // bool UseNumaAlloc=false, // XXX: implement this
          bool HasOutOfLineLockable = false>
class OCImmutableEdgeGraph
    : private internal::LocalIteratorFeature<false>,
      private internal::OutOfLineLockableFeature<HasOutOfLineLockable &&
                                                 !HasNoLockable> {
public:
  template <bool _has_id>
  struct with_id {
    typedef OCImmutableEdgeGraph type;
  };

  template <typename _node_data>
  struct with_node_data {
    typedef OCImmutableEdgeGraph<_node_data, EdgeTy, HasNoLockable,
                                 HasOutOfLineLockable>
        type;
  };

  template <typename _edge_data>
  struct with_edge_data {
    typedef OCImmutableEdgeGraph<NodeTy, _edge_data, HasNoLockable,
                                 HasOutOfLineLockable>
        type;
  };

  template <bool _has_no_lockable>
  struct with_no_lockable {
    typedef OCImmutableEdgeGraph<NodeTy, EdgeTy, _has_no_lockable,
                                 HasOutOfLineLockable>
        type;
  };

  template <bool _use_numa_alloc>
  struct with_numa_alloc {
    typedef OCImmutableEdgeGraph type;
  };

  template <bool _has_out_of_line_lockable>
  struct with_out_of_line_lockable {
    typedef OCImmutableEdgeGraph<NodeTy, EdgeTy, HasNoLockable,
                                 _has_out_of_line_lockable>
        type;
  };

  typedef read_oc_immutable_edge_graph_tag read_tag;

private:
  typedef internal::NodeInfoBase<NodeTy,
                                 !HasNoLockable && !HasOutOfLineLockable>
      NodeInfo;
  typedef LargeArray<NodeInfo> NodeData;

  NodeData nodeData;
  OCFileGraph outGraph;
  OCFileGraph inGraphStorage;
  OCFileGraph* inGraph;

  uint64_t numNodes;
  uint64_t numEdges;

public:
  typedef int tt_is_segmented;

  typedef typename OCFileGraph::GraphNode GraphNode;
  typedef EdgeTy edge_data_type;
  typedef edge_data_type file_edge_data_type;
  typedef NodeTy node_data_type;
  typedef typename OCFileGraph::template EdgeReference<EdgeTy>::type
      edge_data_reference;
  typedef typename NodeInfo::reference node_data_reference;
  typedef typename OCFileGraph::edge_iterator edge_iterator;
  typedef edge_iterator in_edge_iterator;
  typedef typename OCFileGraph::iterator iterator;
  typedef iterator const_iterator;
  typedef boost::counting_iterator<GraphNode> local_iterator;
  typedef local_iterator const_local_iterator;

  class segment_type {
    template <typename, typename, bool, bool>
    friend class OCImmutableEdgeGraph;
    OCFileGraph::segment_type out;
    OCFileGraph::segment_type in;
    iterator nodeBegin;
    iterator nodeEnd;

  public:
    //! Returns true if segment has been loaded into memory
    bool loaded() const { return out.loaded; }
    //! Returns true if segment represents a non-empty range
    explicit operator bool() { return nodeBegin != nodeEnd; }
    size_t size() const { return std::distance(nodeBegin, nodeEnd); }
    bool containsNode(size_t n) const { // XXX: hack
      return *nodeBegin <= n && n < *nodeEnd;
    }
  };

private:
  galois::optional<segment_type> memorySegment;

  segment_type computeSegment(size_t startNode, size_t numEdges) {
    typedef typename OCFileGraph::edge_offset_iterator edge_offset_iterator;

    segment_type ret;

    edge_offset_iterator outStart = outGraph.edge_offset_begin();
    edge_offset_iterator outEnd   = outGraph.edge_offset_end();
    std::advance(outStart, startNode);
    if (outStart == outEnd) {
      ret.nodeBegin = ret.nodeEnd = iterator(0);
      return ret;
    }
    edge_offset_iterator outNext =
        std::lower_bound(outStart + 1, outEnd, *outStart + numEdges);
    ptrdiff_t outNodes = std::distance(outStart, outNext);

    edge_offset_iterator inStart = inGraph->edge_offset_begin();
    edge_offset_iterator inEnd   = inGraph->edge_offset_end();
    std::advance(inStart, startNode);
    edge_offset_iterator inNext =
        std::lower_bound(inStart + 1, inEnd, *inStart + numEdges);
    ptrdiff_t inNodes = std::distance(inStart, inNext);

    ptrdiff_t nodes = std::min(outNodes, inNodes);

    ret.nodeBegin = iterator(startNode);
    ret.nodeEnd   = iterator(startNode + nodes);
    return ret;
  }

  void load(segment_type& seg, size_t sizeof_data) {
    outGraph.load(seg.out, outGraph.edge_begin(*seg.nodeBegin),
                  outGraph.edge_end(seg.nodeEnd[-1]), sizeof_data);
    if (inGraph != &outGraph)
      inGraph->load(seg.in, inGraph->edge_begin(*seg.nodeBegin),
                    inGraph->edge_end(seg.nodeEnd[-1]), sizeof_data);
    else
      seg.in = seg.out;
  }

  template <bool _A1 = HasNoLockable, bool _A2 = HasOutOfLineLockable>
  void acquireNode(GraphNode N, MethodFlag mflag,
                   typename std::enable_if<!_A1 && !_A2>::type* = 0) {
    galois::runtime::acquire(&nodeData[N], mflag);
  }

  template <bool _A1 = HasOutOfLineLockable, bool _A2 = HasNoLockable>
  void acquireNode(GraphNode N, MethodFlag mflag,
                   typename std::enable_if<_A1 && !_A2>::type* = 0) {
    this->outOfLineAcquire(idFromNode(N), mflag);
  }

  template <bool _A1 = HasOutOfLineLockable, bool _A2 = HasNoLockable>
  void acquireNode(GraphNode, MethodFlag,
                   typename std::enable_if<_A2>::type* = 0) {}

public:
  ~OCImmutableEdgeGraph() {
    if (memorySegment) {
      outGraph.unload(memorySegment->out);
      if (inGraph != &outGraph)
        inGraph->unload(memorySegment->in);
    }
  }

  void keepInMemory() {
    memorySegment = galois::optional<segment_type>(computeSegment(0, numEdges));
    load(*memorySegment, LazyObject<EdgeTy>::size_of::value);
  }

  /**
   * Returns a segment starting from the beginning of the graph with either
   * (1) some number of nodes with all their edges but no more than numEdges
   * else (2) one node and all its edges.
   */
  segment_type nextSegment(size_t edges) {
    if (memorySegment)
      return *memorySegment;
    else
      return computeSegment(0, edges);
  }

  /**
   * Returns the next segment after cur.
   */
  segment_type nextSegment(const segment_type& cur, size_t edges) {
    return computeSegment(*cur.nodeEnd, edges);
  }

  void load(segment_type& seg) {
    if (memorySegment)
      return;

    load(seg, LazyObject<EdgeTy>::size_of::value);
  }

  void unload(segment_type& seg) {
    if (memorySegment)
      return;

    outGraph.unload(seg.out);
    if (inGraph != &outGraph)
      inGraph->unload(seg.in);
  }

  iterator begin(const segment_type& cur) { return cur.nodeBegin; }
  iterator end(const segment_type& cur) { return cur.nodeEnd; }

  node_data_reference getData(GraphNode N,
                              MethodFlag mflag = MethodFlag::WRITE) {
    // galois::runtime::checkWrite(mflag, false);
    NodeInfo& NI = nodeData[N];
    acquireNode(N, mflag);
    return NI.getData();
  }

  edge_data_reference
  getEdgeData(const segment_type& segment, edge_iterator ni,
              MethodFlag GALOIS_UNUSED(mflag) = MethodFlag::UNPROTECTED) {
    // galois::runtime::checkWrite(mflag, false);
    return outGraph.getEdgeData<EdgeTy>(segment.out, ni);
  }

  GraphNode getEdgeDst(const segment_type& segment, edge_iterator ni) {
    return outGraph.getEdgeDst(segment.out, ni);
  }

  size_t size() const { return numNodes; }
  size_t sizeEdges() const { return numEdges; }

  iterator begin() const { return outGraph.begin(); }
  iterator end() const { return outGraph.end(); }

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

  edge_iterator edge_begin(const segment_type& segment, GraphNode N,
                           MethodFlag mflag = MethodFlag::WRITE) {
    acquireNode(N, mflag);
    if (galois::runtime::shouldLock(mflag)) {
      for (edge_iterator ii = outGraph.edge_begin(N), ee = outGraph.edge_end(N);
           ii != ee; ++ii) {
        acquireNode(outGraph.getEdgeDst(segment.out, *ii), mflag);
      }
    }
    return outGraph.edge_begin(N);
  }

  edge_iterator edge_end(const segment_type&, GraphNode N,
                         MethodFlag mflag = MethodFlag::WRITE) {
    acquireNode(N, mflag);
    return outGraph.edge_end(N);
  }

  edge_data_reference
  getInEdgeData(const segment_type& segment, edge_iterator ni,
                MethodFlag GALOIS_UNUSED(mflag) = MethodFlag::UNPROTECTED) {
    // galois::runtime::checkWrite(mflag, false);
    return inGraph->getEdgeData<EdgeTy>(segment.in, ni);
  }

  GraphNode getInEdgeDst(const segment_type& segment, in_edge_iterator ni) {
    return inGraph->getEdgeDst(segment.in, ni);
  }

  in_edge_iterator in_edge_begin(const segment_type& segment, GraphNode N,
                                 MethodFlag mflag = MethodFlag::WRITE) {
    acquireNode(N, mflag);
    if (galois::runtime::shouldLock(mflag)) {
      for (in_edge_iterator ii = inGraph->edge_begin(N),
                            ee = inGraph->edge_end(N);
           ii != ee; ++ii) {
        acquireNode(inGraph->getEdgeDst(segment.in, ii), mflag);
      }
    }
    return inGraph->edge_begin(N);
  }

  in_edge_iterator in_edge_end(const segment_type&, GraphNode N,
                               MethodFlag mflag = MethodFlag::WRITE) {
    acquireNode(N, mflag);
    return inGraph->edge_end(N);
  }

  size_t idFromNode(GraphNode N) { return N; }

  GraphNode nodeFromId(size_t N) { return N; }

  //! Assumes that the graph is symmetric
  void createFrom(const std::string& fname) {
    outGraph.fromFile(fname);
    numNodes = outGraph.size();
    numEdges = outGraph.sizeEdges();
    nodeData.create(numNodes);
    inGraph = &outGraph;
    this->outOfLineAllocateInterleaved(numNodes);
    for (size_t i = 0; i < numNodes; ++i)
      this->outOfLineConstructAt(i);
  }

  void createFrom(const std::string& fname, const std::string& transpose) {
    outGraph.fromFile(fname);
    inGraphStorage.fromFile(transpose);
    numNodes = outGraph.size();
    if (numNodes != inGraphStorage.size())
      GALOIS_DIE(
          "graph does not have the same number of nodes as its transpose");
    numEdges = outGraph.sizeEdges();
    nodeData.create(numNodes);
    inGraph = &inGraphStorage;
    this->outOfLineAllocateInterleaved(numNodes);
    for (size_t i = 0; i < numNodes; ++i)
      this->outOfLineConstructAt(i);
  }
};

template <typename GraphTy, typename... Args>
void readGraphDispatch(GraphTy& graph, read_oc_immutable_edge_graph_tag,
                       Args&&... args) {
  graph.createFrom(std::forward<Args>(args)...);
}

} // namespace graphs
} // namespace galois

#endif
