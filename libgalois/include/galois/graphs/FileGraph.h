/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
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

/**
 * @file FileGraph.h
 *
 * Contains FileGraph and FileGraphWriter class declarations.
 *
 * @todo finish up doxygen
 */

#ifndef GALOIS_GRAPH_FILEGRAPH_H
#define GALOIS_GRAPH_FILEGRAPH_H

#include <cstring>
#include <deque>
#include <type_traits>
#include <vector>

#include <boost/iterator/counting_iterator.hpp>
#include <boost/iterator/transform_iterator.hpp>

#include "galois/config.h"
#include "galois/Endian.h"
#include "galois/MethodFlags.h"
#include "galois/LargeArray.h"
#include "galois/graphs/Details.h"
#include "galois/graphs/GraphHelpers.h"
#include "galois/runtime/Context.h"
#include "galois/substrate/CacheLineStorage.h"
#include "galois/substrate/CompilerSpecific.h"
#include "galois/substrate/NumaMem.h"
#include "galois/Reduction.h"

namespace galois {
namespace graphs {

// XXX(ddn): Refactor to eliminate OCFileGraph

//! Graph that mmaps Galois gr files for access
class FileGraph {
public:
  //! type of a node
  using GraphNode = uint64_t;

private:
  struct Convert32 : public std::unary_function<uint32_t, uint32_t> {
    uint32_t operator()(uint32_t x) const { return convert_le32toh(x); }
  };

  struct Convert64 : public std::unary_function<uint64_t, uint64_t> {
    uint64_t operator()(uint64_t x) const { return convert_le64toh(x); }
  };

  struct mapping {
    void* ptr;
    size_t len;
  };

  std::deque<mapping> mappings;
  std::deque<int> fds;

  //! The size of edge data (on 1 edge)
  uint64_t sizeofEdge;
  //! Number of nodes in this (sub)graph
  uint64_t numNodes;
  //! Number of edges in this (sub)graph
  uint64_t numEdges;

  //! Array specifying where a node's edges begin in memory
  uint64_t* outIdx;
  //! Array storing outgoing edge destinations
  void* outs;
  //! Array storing edge data (if it exists)
  char* edgeData;

  //! Galois gr version of read in graph
  int graphVersion;

  //! adjustments to node index when we load only part of a graph
  uint64_t nodeOffset;
  //! adjustments to edge index when we load only part of a graph
  uint64_t edgeOffset;

  // graph reading speed variables
  galois::GAccumulator<uint64_t> numBytesReadIndex, numBytesReadEdgeDst,
      numBytesReadEdgeData;

  /**
   * Construct a file graph by moving in structures from the passed in file graph.
   */
  void move_assign(FileGraph&&);
  /**
   * Get the local edge id of the edge with a specific source and destination
   * if it exists.
   *
   * @param src Global source id of edge to find
   * @param dst Global destination id of edge to find
   * @returns the local edge id of the edge (src, dst) if it exists, otherwise
   * return ~0
   */
  uint64_t getEdgeIdx(GraphNode src, GraphNode dst);

  /**
   * Gets a pointer to the first neighbor of node N.
   *
   * @param N global node id of neighbor begin to get
   * @returns pointer to global id of first neighbor of node N
   */
  void* raw_neighbor_begin(GraphNode N);
  /**
   * Gets a pointer to the end of node N's neighbors in the edge destination
   * array.
   *
   * @param N global node id of neighbor end to get
   * @returns pointer to end of node N's neighbors in edge destination array.
   */
  void* raw_neighbor_end(GraphNode N);

  /**
   * Given an mmap'd version of the graph, initialize graph from that block of
   * memory
   */
  void fromMem(void* m, uint64_t nodeOffset, uint64_t edgeOffset, uint64_t);

  /**
   * Loads a graph from another file graph
   *
   * @param g FileGraph to load from
   * @param sizeofEdgeData Size of edge data (for 1 edge)
   *
   * @returns Pointer to the edge data array of the newly loaded file graph
   */
  void* fromGraph(FileGraph& g, size_t sizeofEdgeData);

  /**
   * Finds the first node N such that
   *
   *  N * nodeSize +
   *  (sum_{i=0}^{N-1} E[i]) * edgeSize
   *    >=
   *  targetSize
   *
   *  in range [lb, ub). Returns ub if unsuccessful.
   *
   * @param nodeSize Weight of nodes
   * @param edgeSize Weight of edges
   * @param targetSize Size that returned node id should attempt to hit
   * @param lb Lower bound of nodes to consider
   * @param ub Upper bound of nodes to consider
   *
   * @returns A node id that hits the target size (or gets close to it)
   */
  size_t findIndex(size_t nodeSize, size_t edgeSize, size_t targetSize,
                   size_t lb, size_t ub);

  void fromFileInterleaved(const std::string& filename, size_t sizeofEdgeData);

  /**
   * Page in a portion of the loaded graph data based based on division of labor
   * by nodes.
   *
   * @param id ID of unit of thread/unit of execution that will page in pages
   * @param total Total number of threads/units of execution to split page in
   * work among
   * @param sizeofEdgeData Size of the loaded edge data
   */
  void pageInByNode(size_t id, size_t total, size_t sizeofEdgeData);

protected:
  /**
   * Copies graph connectivity information from arrays. Returns a pointer to
   * array to populate with edge data.
   *
   * @param outIdx Out index information in an array
   * @param numNodes number of nodes
   * @param outs edge destination array
   * @param numEdges number of edges
   * @param edgeData array of edge data
   * @param sizeofEdgeData The size of the edge data
   * @param nodeOffset how many nodes from the beginning will this graph start
   * from
   * @param edgeOffset how many edges from the beginning will this edge start 
   * from
   * @param converted whether values in arrays are in host byte ordering 
   * (false) or in FileGraph byte ordering (true)
   * @param oGraphVersion Galois graph version to use
   * @return pointer to begining of edgeData in graph
   */
  void* fromArrays(uint64_t* outIdx, uint64_t numNodes, void* outs,
                   uint64_t numEdges, char* edgeData, size_t sizeofEdgeData,
                   uint64_t nodeOffset, uint64_t edgeOffset, bool converted,
                   int oGraphVersion = 1);

public:
  /**
   * Reset the num bytes counters
   */
  void reset_byte_counters() {
    numBytesReadEdgeDst.reset();
    numBytesReadIndex.reset();
    numBytesReadEdgeData.reset();
  }

  /**
   * Return all bytes read
   */
  uint64_t num_bytes_read() {
    return numBytesReadEdgeDst.reduce() + numBytesReadEdgeData.reduce() +
           numBytesReadIndex.reduce();
  }

  // Node Handling

  //! Checks if a node is in the graph (already added)
  bool containsNode(const GraphNode n) const {
    return n + nodeOffset < numNodes;
  }

  // Edge Handling

  //! Get edge data of an edge between 2 nodes
  template <typename EdgeTy>
  EdgeTy& getEdgeData(GraphNode src, GraphNode dst) {
    assert(sizeofEdge == sizeof(EdgeTy));
    numBytesReadEdgeData += sizeof(EdgeTy);
    return reinterpret_cast<EdgeTy*>(edgeData)[getEdgeIdx(src, dst)];
  }

  //! Edge iterators (boost iterator)
  using edge_iterator = boost::counting_iterator<uint64_t>;

  /**
   * Returns the index to the beginning of global node N's outgoing edges
   * in the outgoing edges array.
   *
   * @param N global node id of edge begin to get
   * @returns Iterator to first edge of node N
   */
  edge_iterator edge_begin(GraphNode N);
  /**
   * Returns the index to the end of global node N's outgoing edges
   * in the outgoing edges array.
   *
   * @param N global node id of edge end to get
   * @returns Iterator to end of node N's edges
   */
  edge_iterator edge_end(GraphNode N);

  /**
   * Returns the edges of node N as a range that can be iterated through
   * by C++ foreach.
   */
  runtime::iterable<NoDerefIterator<edge_iterator>> edges(GraphNode N) {
    return internal::make_no_deref_range(edge_begin(N), edge_end(N));
  }

  /**
   * Returns the edges of node N as a range that can be iterated through
   * by C++ foreach.
   */
  runtime::iterable<NoDerefIterator<edge_iterator>> out_edges(GraphNode N) {
    return edges(N);
  }

  /**
   * Sorts outgoing edges of a node. Comparison function is over EdgeTy.
   */
  template <typename EdgeTy, typename CompTy>
  void sortEdgesByEdgeData(GraphNode N,
                           const CompTy& comp = std::less<EdgeTy>()) {
    if (graphVersion == 1) {
      typedef LargeArray<uint32_t> EdgeDst;
      typedef LargeArray<EdgeTy> EdgeData;

      typedef internal::EdgeSortIterator<GraphNode, uint64_t, EdgeDst, EdgeData,
                                         Convert32>
          edge_sort_iterator;

      EdgeDst edgeDst(outs, numEdges);
      EdgeData ed(edgeData, numEdges);

      edge_sort_iterator begin(
          std::distance((uint32_t*)outs, (uint32_t*)raw_neighbor_begin(N)),
          &edgeDst, &ed);
      edge_sort_iterator end(
          std::distance((uint32_t*)outs, (uint32_t*)raw_neighbor_end(N)),
          &edgeDst, &ed);
      std::sort(begin, end,
                internal::EdgeSortCompWrapper<EdgeSortValue<GraphNode, EdgeTy>,
                                              CompTy>(comp));
    } else if (graphVersion == 2) {
      typedef LargeArray<uint64_t> EdgeDst;
      typedef LargeArray<EdgeTy> EdgeData;

      typedef internal::EdgeSortIterator<GraphNode, uint64_t, EdgeDst, EdgeData,
                                         Convert64>
          edge_sort_iterator;

      EdgeDst edgeDst(outs, numEdges);
      EdgeData ed(edgeData, numEdges);

      edge_sort_iterator begin(
          std::distance((uint64_t*)outs, (uint64_t*)raw_neighbor_begin(N)),
          &edgeDst, &ed);
      edge_sort_iterator end(
          std::distance((uint64_t*)outs, (uint64_t*)raw_neighbor_end(N)),
          &edgeDst, &ed);
      std::sort(begin, end,
                internal::EdgeSortCompWrapper<EdgeSortValue<GraphNode, EdgeTy>,
                                              CompTy>(comp));
    } else {
      GALOIS_DIE("unknown file version at sortEdgesByEdgeData", graphVersion);
    }
  }

  /**
   * Sorts outgoing edges of a node.
   * Comparison function is over <code>EdgeSortValue<EdgeTy></code>.
   */
  template <typename EdgeTy, typename CompTy>
  void sortEdges(GraphNode N, const CompTy& comp) {
    if (graphVersion == 1) {
      typedef LargeArray<uint32_t> EdgeDst;
      typedef LargeArray<EdgeTy> EdgeData;
      typedef internal::EdgeSortIterator<GraphNode, uint64_t, EdgeDst, EdgeData,
                                         Convert32>
          edge_sort_iterator;

      EdgeDst edgeDst(outs, numEdges);
      EdgeData ed(edgeData, numEdges);

      edge_sort_iterator begin(
          std::distance((uint32_t*)outs, (uint32_t*)raw_neighbor_begin(N)),
          &edgeDst, &ed);
      edge_sort_iterator end(
          std::distance((uint32_t*)outs, (uint32_t*)raw_neighbor_end(N)),
          &edgeDst, &ed);
      std::sort(begin, end, comp);
    } else if (graphVersion == 2) {
      typedef LargeArray<uint64_t> EdgeDst;
      typedef LargeArray<EdgeTy> EdgeData;
      typedef internal::EdgeSortIterator<GraphNode, uint64_t, EdgeDst, EdgeData,
                                         Convert64>
          edge_sort_iterator;

      EdgeDst edgeDst(outs, numEdges);
      EdgeData ed(edgeData, numEdges);

      edge_sort_iterator begin(
          std::distance((uint64_t*)outs, (uint64_t*)raw_neighbor_begin(N)),
          &edgeDst, &ed);
      edge_sort_iterator end(
          std::distance((uint64_t*)outs, (uint64_t*)raw_neighbor_end(N)),
          &edgeDst, &ed);
      std::sort(begin, end, comp);
    } else {
      GALOIS_DIE("unknown file version at sortEdgesByEdgeData", graphVersion);
    }
  }

  // template<typename EdgeTy>
  // const EdgeTy& getEdgeData(edge_iterator it) const {
  //  assert(edgeData);
  //  return reinterpret_cast<const EdgeTy*>(edgeData)[*it];
  //}

  //! Get edge data given an edge iterator
  template <typename EdgeTy>
  EdgeTy& getEdgeData(edge_iterator it) {
    assert(edgeData);
    numBytesReadEdgeData += sizeof(EdgeTy);
    return reinterpret_cast<EdgeTy*>(edgeData)[*it];
  }

  /**
   * Gets the destination of some edge.
   *
   * @param it local edge id of edge destination to get
   * @returns a global node id representing the destination of the edge
   */
  GraphNode getEdgeDst(edge_iterator it);

  //! iterator over neighbors
  typedef boost::transform_iterator<Convert32, uint32_t*> neighbor_iterator;
  //! iterator over node ids
  typedef boost::transform_iterator<Convert32, uint32_t*> node_id_iterator;
  //! edge iterator
  typedef boost::transform_iterator<Convert64, uint64_t*> edge_id_iterator;
  //! uint64 boost counting iterator
  typedef boost::counting_iterator<uint64_t> iterator;

  /**
   * Gets an iterator to the first neighbor of node N
   * 
   * @warning only version 1 support, do not use with version 2
   */
  neighbor_iterator neighbor_begin(GraphNode N) {
    return boost::make_transform_iterator((uint32_t*)raw_neighbor_begin(N),
                                          Convert32());
  }

  /**
   * Gets an iterator to the end of node N's neighbors
   * 
   * @warning only version 1 support, do not use with version 2
   */
  neighbor_iterator neighbor_end(GraphNode N) {
    return boost::make_transform_iterator((uint32_t*)raw_neighbor_end(N),
                                          Convert32());
  }

  template <typename EdgeTy>
  EdgeTy* edge_data_begin() const {
    assert(edgeData);
    return reinterpret_cast<EdgeTy*>(edgeData);
  }

  template <typename EdgeTy>
  EdgeTy* edge_data_end() const {
    assert(edgeData);
    assert(sizeof(EdgeTy) == sizeofEdge);
    EdgeTy* r = reinterpret_cast<EdgeTy*>(edgeData);
    return &r[numEdges];
  }

  /**
   * Gets the first node of the loaded graph.
   *
   * @returns An iterator to the first node of the graph. Note it is a GLOBAL id.
   */
  iterator begin() const;
  /**
   * Gets the end of the nodes of the loaded graph.
   *
   * @returns An iterator to the end of the nodes of the graph (of the
   * loaded part of the graph).
   */
  iterator end() const;

  //! pair specifying a node range
  typedef std::pair<iterator, iterator> NodeRange;
  //! pair specifying an edge range
  typedef std::pair<edge_iterator, edge_iterator> EdgeRange;
  //! pair of a NodeRange and an EdgeRange
  typedef std::pair<NodeRange, EdgeRange> GraphRange;

  /**
   * Given a division and a total number of divisions, return a range for that
   * particular division to work on. (i.e. this divides labor among divisions
   * depending on how much weight is given to nodes/edges).
   *
   * @param nodeSize Weight of nodes
   * @param edgeSize Weight of edges
   * @param id Division id
   * @param total Total number of divisions
   *
   * @returns A node range and an edge range specifying division "id"'s assigned
   * nodes/edges
   */
  GraphRange divideByNode(size_t nodeSize, size_t edgeSize, size_t id,
                          size_t total);

  /**
   * Divides nodes only considering edges.
   *
   * IMPORTANT: Note that it may potentially not return all nodes in the graph
   * (it will return up to the last node with edges).
   *
   * @param nodeSize Weight of nodes
   * @param edgeSize Weight of edges
   * @param id Division id
   * @param total Total number of divisions
   *
   * @returns A node range and an edge range specifying division "id"'s assigned
   * nodes/edges
   */
  GraphRange divideByEdge(size_t nodeSize, size_t edgeSize, size_t id,
                          size_t total);
  /**
   * Returns an iterator to the beginning of the node destination
   * array.
   *
   * @returns iterator to beginning of the node destination array of the
   * loaded graph (local)
   * @todo implement version 2 support
   */
  node_id_iterator node_id_begin() const;
  /**
   * Returns an iterator to the end of the node destination
   * array.
   *
   * @returns iterator to end of the node destination array of the loaded
   * graph (local)
   * @todo implement version 2 support
   */
  node_id_iterator node_id_end() const;
  /**
   * Returns an iterator to the beginning of the array specifying
   * the index into the destination array where a particular node's
   * edges begin.
   *
   * @return iterator to beginning of edge index array of the loaded graph
   */
  edge_id_iterator edge_id_begin() const;
  /**
   * Returns an iterator to the end of the array specifying
   * the index into the destination array where a particular node's
   * edges begin.
   *
   * @return iterator to end of edge index array of the loaded graph
   */
  edge_id_iterator edge_id_end() const;

  /**
   * Determines if an edge with source N1 and destination N2 existed
   * in the currently loaded (local) graph.
   *
   * @param N1 global node id of neighbor 1 (source)
   * @param N2 global node id of neighbor 2 (destination)
   *
   * @returns true if edge (N1, N2) exists locally, false otherwise
   */
  bool hasNeighbor(GraphNode N1, GraphNode N2);

  //! Returns the number of nodes in the (sub)graph
  size_t size() const { return numNodes; }

  //! Returns the number of edges in the (sub)graph
  size_t sizeEdges() const { return numEdges; }

  //! Returns the size of an edge
  size_t edgeSize() const { return sizeofEdge; }

  /**
   * Default file graph constructor which initializes fields to null values.
   */
  FileGraph();

  /**
   * Construct graph from another FileGraph
   *
   * @param o Other filegraph to initialize from.
   */
  FileGraph(const FileGraph&);
  /**
   * Copy constructor operator for FileGraph
   */
  FileGraph& operator=(const FileGraph&);
  /**
   * Move constructor for FileGraph
   */
  FileGraph(FileGraph&&);
  /**
   * Move constructor operator for FileGraph
   */
  FileGraph& operator=(FileGraph&&);
  /**
   * Destructor. Un-mmaps mmap'd things and closes opened files.
   */
  ~FileGraph();

  /**
   * Given a file name, mmap the entire file into memory. Should
   * be a graph with some specific layout.
   *
   * @param filename Graph file to load
   */
  void fromFile(const std::string& filename);

  /**
   * Loads/mmaps particular portions of a graph corresponding to a node
   * range and edge range into memory.
   *
   * Note that it makes the object work on a LOCAL scale (i.e. there are
   * now local ids corresponding to the subgraph). Most functions will
   * still handle global ids, though. (see below)
   *
   * @param filename File to load
   * @param nrange Node range to load
   * @param erange Edge range to load
   * @param numaMap if true, does interleaved numa allocation for data 
   * structures
   */
  void partFromFile(const std::string& filename, NodeRange nrange,
                    EdgeRange erange, bool numaMap = false);

  /**
   * Reads graph connectivity information from file. Tries to balance memory
   * evenly across system.  Cannot be called during parallel execution.
   *
   * Edge data version.
   */
  template <typename EdgeTy>
  void fromFileInterleaved(
      const std::string& filename,
      typename std::enable_if<!std::is_void<EdgeTy>::value>::type* = 0) {
    fromFileInterleaved(filename, sizeof(EdgeTy));
  }

  /**
   * Reads graph connectivity information from file. Tries to balance memory
   * evenly across system.  Cannot be called during parallel execution.
   *
   * No edge data version.
   */
  template <typename EdgeTy>
  void fromFileInterleaved(
      const std::string& filename,
      typename std::enable_if<std::is_void<EdgeTy>::value>::type* = 0) {
    fromFileInterleaved(filename, 0);
  }

  /**
   * Reads graph connectivity information from graph but not edge data. Returns
   * a pointer to array to populate with edge data.
   */
  template <typename T>
  T* fromGraph(FileGraph& g) {
    return reinterpret_cast<T*>(fromGraph(g, sizeof(T)));
  }

  /**
   * Write current contents of mappings to a file
   *
   * @param file File to write to
   * @todo perform host -> le on data
   */
  void toFile(const std::string& file);
};

/**
 * Simplifies writing graphs.
 *
 * Writer your file in rounds:
 * <ol>
 *  <li>setNumNodes(), setNumEdges(), setSizeofEdgeData()</li>
 *  <li>phase1(), for each node, incrementDegree(Node x)</li>
 *  <li>phase2(), add neighbors for each node, addNeighbor(Node src, Node
 *    dst)</li>
 *  <li>finish(), use as FileGraph</li>
 * </ol>
 */
class FileGraphWriter : public FileGraph {
  std::vector<uint64_t> outIdx;
  std::vector<uint32_t> starts;
  std::vector<uint32_t> outs;
  std::vector<uint64_t> starts64;
  std::vector<uint64_t> outs64;

  size_t sizeofEdgeData;
  size_t numNodes;
  size_t numEdges;
public:
  //! Constructor: initializes nodes, edges, and edge data to 0
  FileGraphWriter() : sizeofEdgeData(0), numNodes(0), numEdges(0) {}

  //! Set number of nodes to write to n
  //! @param n number of nodes to set to
  void setNumNodes(size_t n) { numNodes = n; }
  //! Set number of edges to write to n
  //! @param n number of edges to set to
  void setNumEdges(size_t n) { numEdges = n; }
  //! Set the size of the edge data to write to n
  //! @param n size of edge data to write
  void setSizeofEdgeData(size_t n) { sizeofEdgeData = n; }

  //! Marks the transition to next phase of parsing: counting the degree of
  //! nodes
  void phase1() { outIdx.resize(numNodes); }

  //! Increments degree of id by delta
  void incrementDegree(size_t id, int delta = 1) {
    assert(id < numNodes);
    outIdx[id] += delta;
  }

  //! Marks the transition to next phase of parsing, adding edges
  void phase2() {
    if (numNodes == 0)
      return;

    // Turn counts into partial sums
    auto prev = outIdx.begin();
    for (auto ii = outIdx.begin() + 1, ei = outIdx.end(); ii != ei;
         ++ii, ++prev) {
      *ii += *prev;
    }
    assert(outIdx[numNodes - 1] == numEdges);

    if (numNodes <= std::numeric_limits<uint32_t>::max()) {
      // version 1
      starts.resize(numNodes);
      outs.resize(numEdges);
    } else {
      // version 2
      starts64.resize(numNodes);
      outs64.resize(numEdges);
    }
  }

  //! Adds a neighbor between src and dst
  size_t addNeighbor(size_t src, size_t dst) {
    size_t base = src ? outIdx[src - 1] : 0;

    if (numNodes <= std::numeric_limits<uint32_t>::max()) {
      // version 1
      size_t idx = base + starts[src]++;
      assert(idx < outIdx[src]);
      outs[idx] = dst;
      return idx;
    } else {
      // version 2
      size_t idx = base + (starts64)[src]++;
      assert(idx < outIdx[src]);
      outs64[idx] = dst;
      return idx;
    }
  }

  /**
   * Finish making graph. Returns pointer to block of memory that should be
   * used to store edge data.
   */
  template <typename T>
  T* finish() {
    void* ret;
    if (numNodes <= std::numeric_limits<uint32_t>::max()) {
      // version 1
      ret = fromArrays(&outIdx[0], numNodes, &outs[0], numEdges, nullptr,
                       sizeofEdgeData, 0, 0, false, 1);
      starts.clear();
      outs.clear();
    } else {
      // version 2
      ret = fromArrays(&outIdx[0], numNodes, &outs64[0], numEdges, nullptr,
                       sizeofEdgeData, 0, 0, false, 2);
    }

    outIdx.clear();
    return reinterpret_cast<T*>(ret);
  }
};

/**
 * Adds reverse edges to a graph. Reverse edges have edge data copied from the
 * original edge. The new graph is placed in the out parameter.  The previous
 * out is destroyed.
 */
template <typename EdgeTy>
void makeSymmetric(FileGraph& in_graph, FileGraph& out) {
  typedef FileGraph::GraphNode GNode;
  typedef LargeArray<EdgeTy> EdgeData;
  typedef typename EdgeData::value_type edge_value_type;

  FileGraphWriter g;
  EdgeData edgeData;

  size_t numEdges = 0;

  for (FileGraph::iterator ii = in_graph.begin(), ei = in_graph.end(); ii != ei;
       ++ii) {
    GNode src = *ii;
    for (FileGraph::edge_iterator jj = in_graph.edge_begin(src),
                                  ej = in_graph.edge_end(src);
         jj != ej; ++jj) {
      GNode dst = in_graph.getEdgeDst(jj);
      numEdges += 1;
      if (src != dst)
        numEdges += 1;
    }
  }

  g.setNumNodes(in_graph.size());
  g.setNumEdges(numEdges);
  g.setSizeofEdgeData(EdgeData::has_value ? sizeof(edge_value_type) : 0);

  g.phase1();
  for (FileGraph::iterator ii = in_graph.begin(), ei = in_graph.end(); ii != ei;
       ++ii) {
    GNode src = *ii;
    for (FileGraph::edge_iterator jj = in_graph.edge_begin(src),
                                  ej = in_graph.edge_end(src);
         jj != ej; ++jj) {
      GNode dst = in_graph.getEdgeDst(jj);
      g.incrementDegree(src);
      if (src != dst)
        g.incrementDegree(dst);
    }
  }

  g.phase2();
  edgeData.create(numEdges);
  for (FileGraph::iterator ii = in_graph.begin(), ei = in_graph.end(); ii != ei;
       ++ii) {
    GNode src = *ii;
    for (FileGraph::edge_iterator jj = in_graph.edge_begin(src),
                                  ej = in_graph.edge_end(src);
         jj != ej; ++jj) {
      GNode dst = in_graph.getEdgeDst(jj);
      if (EdgeData::has_value) {
        edge_value_type& data = in_graph.getEdgeData<edge_value_type>(jj);
        edgeData.set(g.addNeighbor(src, dst), data);
        if (src != dst)
          edgeData.set(g.addNeighbor(dst, src), data);
      } else {
        g.addNeighbor(src, dst);
        if (src != dst)
          g.addNeighbor(dst, src);
      }
    }
  }

  edge_value_type* rawEdgeData = g.finish<edge_value_type>();
  if (EdgeData::has_value)
    std::uninitialized_copy(std::make_move_iterator(edgeData.begin()),
                            std::make_move_iterator(edgeData.end()),
                            rawEdgeData);

  out = std::move(g);
}

/**
 * Permutes a graph.
 *
 * Permutation array, P, conforms to: P[i] = j where i is a node index from the
 * original graph and j is a node index in the permuted graph. New, permuted
 * graph is placed in the out parameter. The previous out is destroyed.
 *
 * @param in_graph original graph
 * @param p permutation array
 * @param out permuted graph
 */
template <typename EdgeTy, typename PTy>
void permute(FileGraph& in_graph, const PTy& p, FileGraph& out) {
  typedef FileGraph::GraphNode GNode;
  typedef LargeArray<EdgeTy> EdgeData;
  typedef typename EdgeData::value_type edge_value_type;

  FileGraphWriter g;
  EdgeData edgeData;

  size_t numEdges = in_graph.sizeEdges();
  g.setNumNodes(in_graph.size());
  g.setNumEdges(numEdges);
  g.setSizeofEdgeData(EdgeData::has_value ? sizeof(edge_value_type) : 0);

  g.phase1();
  for (FileGraph::iterator ii = in_graph.begin(), ei = in_graph.end(); ii != ei;
       ++ii) {
    GNode src = *ii;
    for (FileGraph::edge_iterator jj = in_graph.edge_begin(src),
                                  ej = in_graph.edge_end(src);
         jj != ej; ++jj) {
      g.incrementDegree(p[src]);
    }
  }

  g.phase2();
  edgeData.create(numEdges);
  for (FileGraph::iterator ii = in_graph.begin(), ei = in_graph.end(); ii != ei;
       ++ii) {
    GNode src = *ii;
    for (FileGraph::edge_iterator jj = in_graph.edge_begin(src),
                                  ej = in_graph.edge_end(src);
         jj != ej; ++jj) {
      GNode dst = in_graph.getEdgeDst(jj);
      if (EdgeData::has_value) {
        edge_value_type& data = in_graph.getEdgeData<edge_value_type>(jj);
        edgeData.set(g.addNeighbor(p[src], p[dst]), data);
      } else {
        g.addNeighbor(p[src], p[dst]);
      }
    }
  }

  edge_value_type* rawEdgeData = g.finish<edge_value_type>();
  if (EdgeData::has_value)
    std::uninitialized_copy(std::make_move_iterator(edgeData.begin()),
                            std::make_move_iterator(edgeData.end()),
                            rawEdgeData);

  out = std::move(g);
}

} // namespace graphs
} // namespace galois
#endif
