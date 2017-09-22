/** Basic serialized graphs -*- C++ -*-
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
 * This file contains low-level representations of graphs, closely
 * tied with their serialized form in the Galois system. These graphs
 * are very basic (e.g., they don't support concurrency) and are
 * intended to be converted to/from more specialized graph data
 * structures.  More full featured graphs are available in LCGraph.h.
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#ifndef GALOIS_GRAPH_FILEGRAPH_H
#define GALOIS_GRAPH_FILEGRAPH_H

#include "galois/Endian.h"
#include "galois/MethodFlags.h"
#include "galois/LargeArray.h"
#include "galois/graphs/Details.h"
#include "galois/runtime/Context.h"
#include "galois/substrate/CacheLineStorage.h"
#include "galois/substrate/CompilerSpecific.h"
#include "galois/substrate/NumaMem.h"
#include "galois/Accumulator.h"

#include <boost/iterator/counting_iterator.hpp>
#include <boost/iterator/transform_iterator.hpp>

#include "galois/graphs/GraphHelpers.h"

#include <type_traits>
#include <deque>
#include <vector>
#include <string.h>

namespace galois {
namespace graphs {

//XXX(ddn): Refactor to eliminate OCFileGraph
//! Graph serialized to a file
class FileGraph {
public:
  typedef uint64_t GraphNode;

private:
  struct Convert32: public std::unary_function<uint32_t, uint32_t> {
    uint32_t operator()(uint32_t x) const {
      return convert_le32toh(x);
    }
  };
  
  struct Convert64: public std::unary_function<uint64_t,uint64_t> {
    uint64_t operator()(uint64_t x) const {
      return convert_le64toh(x);
    }
  };

  struct mapping {
    void* ptr;
    size_t len;
  };

  std::deque<mapping> mappings;
  std::deque<int> fds;

  uint64_t sizeofEdge;
  uint64_t numNodes;
  uint64_t numEdges;

  uint64_t* outIdx;
  void* outs;
  char* edgeData;

  int graphVersion;

  //! adjustments to node index when we load only part of a graph
  uint64_t nodeOffset;
  //! adjustments to edge index when we load only part of a graph
  uint64_t edgeOffset;

  // graph reading speed variables
  galois::GAccumulator<uint64_t> numBytesReadIndex, numBytesReadEdgeDst, numBytesReadEdgeData;


  void move_assign(FileGraph&&);
  uint64_t getEdgeIdx(GraphNode src, GraphNode dst);
  void* raw_neighbor_begin(GraphNode N);
  void* raw_neighbor_end(GraphNode N);

  //! Initializes a graph from block of memory
  void fromMem(void* m, uint64_t nodeOffset, uint64_t edgeOffset, uint64_t);

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
   */
  size_t findIndex(size_t nodeSize, size_t edgeSize, size_t targetSize, size_t lb, size_t ub);
  
  void fromFileInterleaved(const std::string& filename, size_t sizeofEdgeData);

  void pageInByNode(size_t id, size_t total, size_t sizeofEdgeData);

protected:
  /**
   * Copies graph connectivity information from arrays. Returns a pointer to
   * array to populate with edge data.
   *
   * @param converted
   *   whether values in arrays are in host byte ordering (false) or in
   *   FileGraph byte ordering (true)
   * @return pointer to begining of edgeData in graph
   */
  void* fromArrays(uint64_t* outIdx, uint64_t numNodes,
      void* outs, uint64_t numEdges,
      char* edgeData, size_t sizeofEdgeData,
      uint64_t nodeOffset, uint64_t edgeOffset,
      bool converted, int oGraphVersion=1);

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
     return numBytesReadEdgeDst.reduce() + numBytesReadEdgeData.reduce() + numBytesReadIndex.reduce();
  }

  // Node Handling

  //! Checks if a node is in the graph (already added)
  bool containsNode(const GraphNode n) const {
    return n + nodeOffset < numNodes;
  }

  // Edge Handling
  template<typename EdgeTy>
  EdgeTy& getEdgeData(GraphNode src, GraphNode dst) {
    assert(sizeofEdge == sizeof(EdgeTy));
    numBytesReadEdgeData += sizeof(EdgeTy);
    return reinterpret_cast<EdgeTy*>(edgeData)[getEdgeIdx(src, dst)];
  }

  // Iterators
  typedef boost::counting_iterator<uint64_t> edge_iterator;
  edge_iterator edge_begin(GraphNode N);
  edge_iterator edge_end(GraphNode N);

  runtime::iterable<NoDerefIterator<edge_iterator>> edges(GraphNode N) {
    return internal::make_no_deref_range(edge_begin(N), edge_end(N));
  }

  runtime::iterable<NoDerefIterator<edge_iterator>> out_edges(GraphNode N) {
    return edges(N);
  }

  /**
   * Sorts outgoing edges of a node. Comparison function is over EdgeTy.
   */
  template<typename EdgeTy, typename CompTy>
  void sortEdgesByEdgeData(GraphNode N, const CompTy& comp = std::less<EdgeTy>()) {
    if (graphVersion == 1) {
      typedef LargeArray<uint32_t> EdgeDst;
      typedef LargeArray<EdgeTy> EdgeData;
  
      typedef internal::EdgeSortIterator<GraphNode, uint64_t,
                                       EdgeDst,EdgeData,Convert32> edge_sort_iterator;
  
      EdgeDst edgeDst(outs, numEdges);
      EdgeData ed(edgeData, numEdges);

      edge_sort_iterator begin(std::distance((uint32_t*)outs, 
                                 (uint32_t*)raw_neighbor_begin(N)), 
                               &edgeDst, &ed);
      edge_sort_iterator end(std::distance((uint32_t*)outs, 
                               (uint32_t*)raw_neighbor_end(N)), 
                             &edgeDst, &ed);
      std::sort(begin, end, 
                internal::EdgeSortCompWrapper<EdgeSortValue<GraphNode,EdgeTy>,
                                            CompTy>(comp));
    } else if (graphVersion == 2) {
      typedef LargeArray<uint64_t> EdgeDst;
      typedef LargeArray<EdgeTy> EdgeData;
  
      typedef internal::EdgeSortIterator<GraphNode, uint64_t,
                                       EdgeDst,EdgeData,Convert64> edge_sort_iterator;
  
      EdgeDst edgeDst(outs, numEdges);
      EdgeData ed(edgeData, numEdges);

      edge_sort_iterator begin(std::distance((uint64_t*)outs, 
                                 (uint64_t*)raw_neighbor_begin(N)), 
                               &edgeDst, &ed);
      edge_sort_iterator end(std::distance((uint64_t*)outs, 
                               (uint64_t*)raw_neighbor_end(N)), 
                             &edgeDst, &ed);
      std::sort(begin, end, 
                internal::EdgeSortCompWrapper<EdgeSortValue<GraphNode,EdgeTy>,
                                            CompTy>(comp));
    } else {
      GALOIS_DIE("unknown file version at sortEdgesByEdgeData", graphVersion);
    }
  }

  /**
   * Sorts outgoing edges of a node. 
   * Comparison function is over <code>EdgeSortValue<EdgeTy></code>.
   */
  template<typename EdgeTy, typename CompTy>
  void sortEdges(GraphNode N, const CompTy& comp) {
    if (graphVersion == 1) {
      typedef LargeArray<uint32_t> EdgeDst;
      typedef LargeArray<EdgeTy> EdgeData;
      typedef internal::EdgeSortIterator<GraphNode, uint64_t, EdgeDst, EdgeData,
                                       Convert32> edge_sort_iterator;

      EdgeDst edgeDst(outs, numEdges);
      EdgeData ed(edgeData, numEdges);

      edge_sort_iterator begin(std::distance((uint32_t*)outs, 
                                 (uint32_t*)raw_neighbor_begin(N)), 
                               &edgeDst, &ed);
      edge_sort_iterator end(std::distance((uint32_t*)outs, 
                                 (uint32_t*)raw_neighbor_end(N)), 
                             &edgeDst, &ed);
      std::sort(begin, end, comp);
    } else if (graphVersion == 2) {
      typedef LargeArray<uint64_t> EdgeDst;
      typedef LargeArray<EdgeTy> EdgeData;
      typedef internal::EdgeSortIterator<GraphNode, uint64_t, EdgeDst, EdgeData,
                                       Convert64> edge_sort_iterator;

      EdgeDst edgeDst(outs, numEdges);
      EdgeData ed(edgeData, numEdges);

      edge_sort_iterator begin(std::distance((uint64_t*)outs, 
                                 (uint64_t*)raw_neighbor_begin(N)), 
                               &edgeDst, &ed);
      edge_sort_iterator end(std::distance((uint64_t*)outs, 
                               (uint64_t*)raw_neighbor_end(N)), 
                             &edgeDst, &ed);
      std::sort(begin, end, comp);
    } else {
      GALOIS_DIE("unknown file version at sortEdgesByEdgeData", graphVersion);
    }
  }

  //template<typename EdgeTy> 
  //const EdgeTy& getEdgeData(edge_iterator it) const {
  //  assert(edgeData);
  //  return reinterpret_cast<const EdgeTy*>(edgeData)[*it];
  //}

  template<typename EdgeTy>
  EdgeTy& getEdgeData(edge_iterator it) {
    assert(edgeData);
    numBytesReadEdgeData += sizeof(EdgeTy);
    return reinterpret_cast<EdgeTy*>(edgeData)[*it];
  }

  GraphNode getEdgeDst(edge_iterator it);

  typedef boost::transform_iterator<Convert32, uint32_t*> neighbor_iterator;
  typedef boost::transform_iterator<Convert32, uint32_t*> node_id_iterator;
  typedef boost::transform_iterator<Convert64, uint64_t*> edge_id_iterator;
  typedef boost::counting_iterator<uint64_t> iterator;
  
  // TODO ONLY VERSION 1 SUPPORT, DO NOT USE WITH VERSION 2
  neighbor_iterator neighbor_begin(GraphNode N) {
    return boost::make_transform_iterator((uint32_t*)raw_neighbor_begin(N), 
                                          Convert32());
  }

  // TODO ONLY VERSION 1 SUPPORT, DO NOT USE WITH VERSION 2
  neighbor_iterator neighbor_end(GraphNode N) {
    return boost::make_transform_iterator((uint32_t*)raw_neighbor_end(N), 
                                          Convert32());
  }

  template<typename EdgeTy>
  EdgeTy* edge_data_begin() const {
    assert(edgeData);
    return reinterpret_cast<EdgeTy*>(edgeData);
  }

  template<typename EdgeTy>
  EdgeTy* edge_data_end() const {
    assert(edgeData);
    assert(sizeof(EdgeTy) == sizeofEdge);
    EdgeTy* r = reinterpret_cast<EdgeTy*>(edgeData);
    return &r[numEdges];
  }

  iterator begin() const;
  iterator end() const;

  typedef std::pair<iterator, iterator> NodeRange;
  typedef std::pair<edge_iterator, edge_iterator> EdgeRange;
  typedef std::pair<NodeRange, EdgeRange> GraphRange;

  //! Divides nodes into balanced ranges 
  GraphRange divideByNode(size_t nodeSize, size_t edgeSize, size_t id, size_t total);

  //! Divides edges into balanced ranges
  GraphRange divideByEdge(size_t nodeSize, size_t edgeSize, size_t id, size_t total);

  node_id_iterator node_id_begin() const;
  node_id_iterator node_id_end() const;
  edge_id_iterator edge_id_begin() const;
  edge_id_iterator edge_id_end() const;

  bool hasNeighbor(GraphNode N1, GraphNode N2);

  //! Returns the number of nodes in the graph
  size_t size() const { return numNodes; }

  //! Returns the number of edges in the graph
  size_t sizeEdges() const { return numEdges; }

  //! Returns the size of an edge
  size_t edgeSize() const { return sizeofEdge; }

  FileGraph();
  FileGraph(const FileGraph&);
  FileGraph& operator=(const FileGraph&);
  FileGraph(FileGraph&&);
  FileGraph& operator=(FileGraph&&);
  ~FileGraph();

  //! Reads graph from file
  void fromFile(const std::string& filename);

  /**
   * Reads a subgraph corresponding to given range of edges from file.
   *
   * An example of use:
   *
   * \snippet test/filegraph.cpp Reading part of graph
   */
  void partFromFile(const std::string& filename, NodeRange nrange, 
                    EdgeRange erange, bool numaMap=false);

  /**
   * Reads graph connectivity information from file. Tries to balance memory
   * evenly across system.  Cannot be called during parallel execution.
   */
  template<typename EdgeTy>
  void fromFileInterleaved(const std::string& filename, 
      typename std::enable_if<!std::is_void<EdgeTy>::value>::type* = 0) {
    fromFileInterleaved(filename, sizeof(EdgeTy));
  }

  template<typename EdgeTy>
  void fromFileInterleaved(const std::string& filename, 
      typename std::enable_if<std::is_void<EdgeTy>::value>::type* = 0) {
    fromFileInterleaved(filename, 0);
  }

  /** 
   * Reads graph connectivity information from graph but not edge data. Returns
   * a pointer to array to populate with edge data.
   */
  template<typename T>
  T* fromGraph(FileGraph& g) {
    return reinterpret_cast<T*>(fromGraph(g, sizeof(T)));
  }

  //! Writes graph to file
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
class FileGraphWriter: public FileGraph {
  std::vector<uint64_t> outIdx;

  std::vector<uint32_t> starts;
  std::vector<uint32_t> outs;
  std::vector<uint64_t> starts64;
  std::vector<uint64_t> outs64;

  size_t sizeofEdgeData;
  size_t numNodes;
  size_t numEdges;

public:
  FileGraphWriter(): sizeofEdgeData(0), numNodes(0), numEdges(0) { }

  void setNumNodes(size_t n) { numNodes = n; }
  void setNumEdges(size_t n) { numEdges = n; }
  void setSizeofEdgeData(size_t n) { sizeofEdgeData = n; }
  
  //! Marks the transition to next phase of parsing, counting the degree of
  //! nodes
  void phase1() { 
    outIdx.resize(numNodes);
  }

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
    for (auto ii = outIdx.begin() + 1, ei = outIdx.end(); ii != ei; ++ii, ++prev) {
      *ii += *prev;
    }
    assert(outIdx[numNodes-1] == numEdges);

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
  template<typename T> 
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
 * original edge. New graph is placed in out parameter.  The previous graph in
 * out is destroyed.
 */
template<typename EdgeTy>
void makeSymmetric(FileGraph& in, FileGraph& out) {
  typedef FileGraph::GraphNode GNode;
  typedef LargeArray<EdgeTy> EdgeData;
  typedef typename EdgeData::value_type edge_value_type;

  FileGraphWriter g;
  EdgeData edgeData;

  size_t numEdges = 0;

  for (FileGraph::iterator ii = in.begin(), ei = in.end(); ii != ei; ++ii) {
    GNode src = *ii;
    for (FileGraph::edge_iterator jj = in.edge_begin(src), ej = in.edge_end(src); jj != ej; ++jj) {
      GNode dst = in.getEdgeDst(jj);
      numEdges += 1;
      if (src != dst)
        numEdges += 1;
    }
  }

  g.setNumNodes(in.size());
  g.setNumEdges(numEdges);
  g.setSizeofEdgeData(EdgeData::has_value ? sizeof(edge_value_type) : 0);

  g.phase1();
  for (FileGraph::iterator ii = in.begin(), ei = in.end(); ii != ei; ++ii) {
    GNode src = *ii;
    for (FileGraph::edge_iterator jj = in.edge_begin(src), ej = in.edge_end(src); jj != ej; ++jj) {
      GNode dst = in.getEdgeDst(jj);
      g.incrementDegree(src);
      if (src != dst)
        g.incrementDegree(dst);
    }
  }

  g.phase2();
  edgeData.create(numEdges);
  for (FileGraph::iterator ii = in.begin(), ei = in.end(); ii != ei; ++ii) {
    GNode src = *ii;
    for (FileGraph::edge_iterator jj = in.edge_begin(src), ej = in.edge_end(src); jj != ej; ++jj) {
      GNode dst = in.getEdgeDst(jj);
      if (EdgeData::has_value) {
        edge_value_type& data = in.getEdgeData<edge_value_type>(jj);
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
    std::uninitialized_copy(std::make_move_iterator(edgeData.begin()), std::make_move_iterator(edgeData.end()), rawEdgeData);

  out = std::move(g);
}

/**
 * Permutes a graph.
 *
 * Permutation array, P, conforms to: P[i] = j where i is a node index from the
 * original graph and j is a node index in the permuted graph. New, permuted
 * graph is placed in the out parameter. The previous graph in out is destroyed.
 *
 * @param in original graph
 * @param p permutation array
 * @param out permuted graph
 */
template<typename EdgeTy,typename PTy>
void permute(FileGraph& in, const PTy& p, FileGraph& out) {
  typedef FileGraph::GraphNode GNode;
  typedef LargeArray<EdgeTy> EdgeData;
  typedef typename EdgeData::value_type edge_value_type;

  FileGraphWriter g;
  EdgeData edgeData;

  size_t numEdges = in.sizeEdges();
  g.setNumNodes(in.size());
  g.setNumEdges(numEdges);
  g.setSizeofEdgeData(EdgeData::has_value ? sizeof(edge_value_type) : 0);

  g.phase1();
  for (FileGraph::iterator ii = in.begin(), ei = in.end(); ii != ei; ++ii) {
    GNode src = *ii;
    for (FileGraph::edge_iterator jj = in.edge_begin(src), ej = in.edge_end(src); jj != ej; ++jj) {
      g.incrementDegree(p[src]);
    }
  }

  g.phase2();
  edgeData.create(numEdges);
  for (FileGraph::iterator ii = in.begin(), ei = in.end(); ii != ei; ++ii) {
    GNode src = *ii;
    for (FileGraph::edge_iterator jj = in.edge_begin(src), ej = in.edge_end(src); jj != ej; ++jj) {
      GNode dst = in.getEdgeDst(jj);
      if (EdgeData::has_value) {
        edge_value_type& data = in.getEdgeData<edge_value_type>(jj);
        edgeData.set(g.addNeighbor(p[src], p[dst]), data);
      } else {
        g.addNeighbor(p[src], p[dst]);
      }
    }
  }

  edge_value_type* rawEdgeData = g.finish<edge_value_type>();
  if (EdgeData::has_value)
    std::uninitialized_copy(std::make_move_iterator(edgeData.begin()), std::make_move_iterator(edgeData.end()), rawEdgeData);

  out = std::move(g);
}

}
}
#endif
