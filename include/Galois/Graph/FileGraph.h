/** Basic serialized graphs -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2012, The University of Texas at Austin. All rights reserved.
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
 * This file contains low-level representations of graphs, closely tied with
 * their serialized form in the Galois system. These graphs are very basic
 * (e.g., they don't support concurrency) and are intended to be converted
 * to/from more specialized graph data structures.  More full featured graphs
 * are available in LCGraph.h. 
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#ifndef GALOIS_GRAPH_FILEGRAPH_H
#define GALOIS_GRAPH_FILEGRAPH_H

#include "Galois/Endian.h"
#include "Galois/MethodFlags.h"
#include "Galois/LargeArray.h"
#include "Galois/Graph/Util.h"
#include "Galois/Runtime/Context.h"
#include "Galois/Runtime/ll/CacheLineStorage.h"

#include <boost/iterator/counting_iterator.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/type_traits/is_pod.hpp>
#include <boost/utility.hpp>

#include <map>
#include <vector>
#include <fstream>

#include <string.h>

namespace Galois {
namespace Graph {

//! Graph serialized to a file
class FileGraph: boost::noncopyable {
public:
  typedef uint32_t GraphNode;

protected:
  void* volatile masterMapping;
  size_t masterLength;
  uint64_t sizeEdgeTy;
  int masterFD;

  uint64_t* outIdx;
  uint32_t* outs;

  char* edgeData;

  uint64_t numEdges;
  uint64_t numNodes;

  uint64_t getEdgeIdx(GraphNode src, GraphNode dst) const;
  uint32_t* raw_neighbor_begin(GraphNode N) const;
  uint32_t* raw_neighbor_end(GraphNode N) const;

  struct Convert32: public std::unary_function<uint32_t, uint32_t> {
    uint32_t operator()(uint32_t x) const {
      return le32toh(x);
    }
  };
  
  struct Convert64: public std::unary_function<uint64_t,uint64_t> {
    uint64_t operator()(uint64_t x) const {
      return le64toh(x);
    }
  };

  //! Initialize graph from block of memory
  void parse(void* m);

  //! Read graph connectivity information from memory
  void structureFromMem(void* mem, size_t len, bool clone);

  void* structureFromArrays(uint64_t* outIdxs, uint64_t numNodes,
      uint32_t* outs, uint64_t numEdges, size_t sizeofEdgeData);

  void* structureFromGraph(FileGraph& g, size_t sizeofEdgeData);


public:
  // Node Handling

  //! Check if a node is in the graph (already added)
  bool containsNode(const GraphNode n) const;

  // Edge Handling
  template<typename EdgeTy>
  EdgeTy& getEdgeData(GraphNode src, GraphNode dst) {
    assert(sizeEdgeTy == sizeof(EdgeTy));
    return reinterpret_cast<EdgeTy*>(edgeData)[getEdgeIdx(src, dst)];
  }

  // Iterators
  typedef boost::counting_iterator<uint64_t> edge_iterator;
  edge_iterator edge_begin(GraphNode N) const;
  edge_iterator edge_end(GraphNode N) const;

  EdgesWithNoFlagIterator<FileGraph> out_edges(GraphNode N) {
    return EdgesWithNoFlagIterator<FileGraph>(*this, N);
  }

  /**
   * Sorts outgoing edges of a node. Comparison function is over EdgeTy.
   */
  template<typename EdgeTy, typename CompTy>
  void sortEdgesByEdgeData(GraphNode N, const CompTy& comp = std::less<EdgeTy>()) {
    typedef LargeArrayWrapper<GraphNode> EdgeDst;
    typedef LargeArrayWrapper<EdgeTy> EdgeData;
    typedef EdgeSortIterator<GraphNode,uint64_t,EdgeDst,EdgeData> edge_sort_iterator;

    EdgeDst edgeDst(outs, numEdges);
    EdgeData ed(edgeData, numEdges);

    edge_sort_iterator begin(std::distance(outs, raw_neighbor_begin(N)), &edgeDst, &ed);
    edge_sort_iterator end(std::distance(outs, raw_neighbor_end(N)), &edgeDst, &ed);

    std::sort(begin, end, EdgeSortCompWrapper<EdgeSortValue<GraphNode,EdgeTy>,CompTy>(comp));
  }

  /**
   * Sorts outgoing edges of a node. Comparison function is over <code>EdgeSortValue<EdgeTy></code>.
   */
  template<typename EdgeTy, typename CompTy>
  void sortEdges(GraphNode N, const CompTy& comp) {
    typedef LargeArrayWrapper<GraphNode> EdgeDst;
    typedef LargeArrayWrapper<EdgeTy> EdgeData;
    typedef EdgeSortIterator<GraphNode,uint64_t,EdgeDst,EdgeData> edge_sort_iterator;

    EdgeDst edgeDst(outs, numEdges);
    EdgeData ed(edgeData, numEdges);

    edge_sort_iterator begin(std::distance(outs, raw_neighbor_begin(N)), &edgeDst, &ed);
    edge_sort_iterator end(std::distance(outs, raw_neighbor_end(N)), &edgeDst, &ed);

    std::sort(begin, end, comp);
  }

  template<typename EdgeTy> 
  EdgeTy& getEdgeData(edge_iterator it) const {
    return reinterpret_cast<EdgeTy*>(edgeData)[*it];
  }

  GraphNode getEdgeDst(edge_iterator it) const;

  typedef boost::transform_iterator<Convert32, uint32_t*> neighbor_iterator;
  typedef boost::transform_iterator<Convert32, uint32_t*> node_id_iterator;
  typedef boost::transform_iterator<Convert64, uint64_t*> edge_id_iterator;
  
  neighbor_iterator neighbor_begin(GraphNode N) const {
    return boost::make_transform_iterator(raw_neighbor_begin(N), Convert32());
  }

  neighbor_iterator neighbor_end(GraphNode N) const {
    return boost::make_transform_iterator(raw_neighbor_end(N), Convert32());
  }

  node_id_iterator node_id_begin() const;
  node_id_iterator node_id_end() const;
  edge_id_iterator edge_id_begin() const;
  edge_id_iterator edge_id_end() const;

  template<typename EdgeTy>
  EdgeTy& getEdgeData(neighbor_iterator it) {
    return reinterpret_cast<EdgeTy*>(edgeData)[std::distance(outs, it.base())];
  }

  template<typename EdgeTy>
  EdgeTy* edge_data_begin() const {
    return reinterpret_cast<EdgeTy*>(edgeData);
  }

  template<typename EdgeTy>
  EdgeTy* edge_data_end() const {
    assert(sizeof(EdgeTy) == sizeEdgeTy);
    EdgeTy* r = reinterpret_cast<EdgeTy*>(edgeData);
    return &r[numEdges];
  }

  bool hasNeighbor(GraphNode N1, GraphNode N2) const;

  typedef boost::counting_iterator<uint64_t> iterator;

  //! Iterate over nodes in graph (not thread safe)
  iterator begin() const;

  iterator end() const;

  //! The number of nodes in the graph
  unsigned int size() const;

  //! The number of edges in the graph
  unsigned int sizeEdges() const;

  //! sizeof an edge
  size_t sizeEdge() const { return sizeEdgeTy; }

  FileGraph();
  ~FileGraph();

  //! Read graph connectivity information from file
  void structureFromFile(const std::string& filename);

  //! Read graph connectivity information from arrays.
  //! Return a pointer to array to populate with edge data.
  template<typename T>
  T* structureFromArrays(uint64_t* outIdxs, uint64_t numNodes,
      uint32_t* outs, uint64_t numEdges) {
    return reinterpret_cast<T*>(structureFromArrays(outIdx, numNodes, outs, numEdges, sizeof(T)));
  }

  //! Read graph connectivity information from arrays.
  //! Return a pointer to array to populate with edge data.
  template<typename T>
  T* structureFromGraph(FileGraph& g) {
    return reinterpret_cast<T*>(structureFromGraph(g, sizeof(T)));
  }

  //! Write graph connectivity information to file
  void structureToFile(const std::string& file);

  void swap(FileGraph& other);
  void cloneFrom(FileGraph& other);
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
  uint64_t *outIdx; // outIdxs
  uint32_t *starts;
  uint32_t *outs; // outs
  size_t sizeofEdgeData;

public:
  FileGraphWriter(): outIdx(0), starts(0), outs(0), sizeofEdgeData(0) { }

  ~FileGraphWriter() { 
    if (outIdx)
      delete [] outIdx;
    if (starts)
      delete [] starts;
    if (outs)
      delete [] outs;
  }

  void setNumNodes(uint64_t n) { this->numNodes = n; }
  void setNumEdges(uint64_t n) { this->numEdges = n; }
  void setSizeofEdgeData(size_t n) { sizeofEdgeData = n; }
  
  //! Marks the transition to next phase of parsing, counting the degree of
  //! nodes
  void phase1() { 
    assert(!outIdx);
    outIdx = new uint64_t[this->numNodes];
    memset(outIdx, 0, sizeof(*outIdx) * this->numNodes);
  }

  //! Increments degree of id by delta
  void incrementDegree(size_t id, int delta = 1) {
    assert(id < this->numNodes);
    outIdx[id] += delta;
  }

  //! Marks the transition to next phase of parsing, adding edges
  void phase2() {
    if (this->numNodes == 0)
      return;

    // Turn counts into partial sums
    uint64_t* prev = outIdx;
    for (uint64_t *ii = outIdx + 1, *ei = outIdx + this->numNodes; ii != ei; ++ii, ++prev) {
      *ii += *prev;
    }
    assert(outIdx[this->numNodes-1] == this->numEdges);

    starts = new uint32_t[this->numNodes];
    memset(starts, 0, sizeof(*starts) * this->numNodes);

    outs = new uint32_t[this->numEdges];
  }

  //! Adds a neighbor between src and dst
  size_t addNeighbor(size_t src, size_t dst) {
    size_t base = src ? outIdx[src-1] : 0;
    size_t idx = base + starts[src]++;
    assert(idx < outIdx[src]);
    outs[idx] = dst;
    return idx;
  }

  /** 
   * Finish making graph. Returns pointer to block of memory that should be
   * used to store edge data.
   */
  template<typename T>
  T* finish() { 
    void* ret = structureFromArrays(outIdx, this->numNodes, outs, this->numEdges, sizeofEdgeData);
    delete [] outIdx;
    outIdx = 0;
    delete [] starts;
    starts = 0;
    delete [] outs;
    outs = 0;
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
  typedef LargeArray<EdgeTy,boost::is_pod<EdgeTy>::value> EdgeData;
  typedef typename EdgeData::value_type edge_value_type;

  FileGraphWriter g;
  EdgeData edgeData;

  size_t numEdges = in.sizeEdges() * 2;
  g.setNumNodes(in.size());
  g.setNumEdges(numEdges);
  g.setSizeofEdgeData(EdgeData::has_value ? sizeof(edge_value_type) : 0);

  g.phase1();
  for (FileGraph::iterator ii = in.begin(), ei = in.end(); ii != ei; ++ii) {
    GNode src = *ii;
    for (FileGraph::edge_iterator jj = in.edge_begin(src), ej = in.edge_end(src); jj != ej; ++jj) {
      GNode dst = in.getEdgeDst(jj);
      g.incrementDegree(src);
      g.incrementDegree(dst);
    }
  }

  g.phase2();
  edgeData.allocate(numEdges);
  for (FileGraph::iterator ii = in.begin(), ei = in.end(); ii != ei; ++ii) {
    GNode src = *ii;
    for (FileGraph::edge_iterator jj = in.edge_begin(src), ej = in.edge_end(src); jj != ej; ++jj) {
      GNode dst = in.getEdgeDst(jj);
      if (EdgeData::has_value) {
        edge_value_type& data = in.getEdgeData<edge_value_type>(jj);
        edgeData.set(g.addNeighbor(src, dst), data);
        edgeData.set(g.addNeighbor(dst, src), data);
      } else {
        g.addNeighbor(src, dst);
        g.addNeighbor(dst, src);
      }
    }
  }

  edge_value_type* rawEdgeData = g.finish<edge_value_type>();
  if (EdgeData::has_value)
    std::copy(edgeData.begin(), edgeData.end(), rawEdgeData);

  out.swap(g);
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
  typedef LargeArray<EdgeTy,boost::is_pod<EdgeTy>::value> EdgeData;
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
  edgeData.allocate(numEdges);
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
    std::copy(edgeData.begin(), edgeData.end(), rawEdgeData);

  out.swap(g);
}

template<typename GraphTy,typename... Args>
void structureFromFile(GraphTy& g, const std::string& fname, Args&&... args) {
  FileGraph graph;
  graph.structureFromFile(fname);
  g.structureFromGraph(graph, std::forward<Args>(args)...);
}

//! Local computation graph (i.e., graph structure does not change)
//! THIS GRAPH SHOULD GO AWAY
template<typename NodeTy, typename EdgeTy>
class LC_FileGraph : public FileGraph {

  struct gNode : public Galois::Runtime::Lockable {
    NodeTy data;
    gNode() {}
  };

  //null if type is void
  Galois::Runtime::LL::CacheLineStorage<gNode>* nodeData;

public:
  GALOIS_ATTRIBUTE_DEPRECATED
  LC_FileGraph(): nodeData(0) { }

  ~LC_FileGraph() {
    if (nodeData)
      delete[] nodeData;
  }
  
  NodeTy& getData(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    Galois::Runtime::acquire(&nodeData[N].data, mflag);
    return nodeData[N].data.data;
  }

  EdgeTy& getEdgeData(GraphNode src, GraphNode dst, MethodFlag mflag = MethodFlag::ALL) {
    return FileGraph::getEdgeData<EdgeTy>(src, dst, mflag);
  }

  EdgeTy& getEdgeData(FileGraph::edge_iterator it, MethodFlag mflag = MethodFlag::ALL) {
    return FileGraph::getEdgeData<EdgeTy>(it, mflag);
  }

  EdgeTy& getEdgeData(FileGraph::neighbor_iterator it, MethodFlag mflag = MethodFlag::ALL) {
    return FileGraph::getEdgeData<EdgeTy>(it, mflag);
  }

  neighbor_iterator neighbor_begin(GraphNode N, MethodFlag mflag = MethodFlag::ALL) const {
    return FileGraph::neighbor_begin(N);
  }

  neighbor_iterator neighbor_end(GraphNode N, MethodFlag mflag = MethodFlag::ALL) const {
    return FileGraph::neighbor_end(N);
  }

  //! Loads node data from file
  void nodeDataFromFile(const char* filename) {
    emptyNodeData();
    std::ifstream file(filename);
    for (uint64_t i = 0; i < size(); ++i)
      file >> nodeData[i];
  }
  
  //! Initializes node data for the graph to default values
  void emptyNodeData(NodeTy init = NodeTy()) {
    nodeData = new Galois::Runtime::LL::CacheLineStorage<gNode>[size()];
    for (uint64_t i = 0; i < size(); ++i)
      nodeData[i].data.data = init;
  }

  void swap(LC_FileGraph& other) {
    std::swap(nodeData, other.nodeData);
    FileGraph::swap(other);
  }

  void cloneFrom(LC_FileGraph& other) {
    nodeData = other.nodeData;
    FileGraph::cloneFrom(other);
  }

  template<typename GTy>
  void copyGraph(GTy& graph) {
    structureFromGraph(graph);
    emptyNodeData();
    int i = 0;
    for (typename GTy::iterator ii = graph.begin(), ee = graph.end(); ii != ee; ++ii, ++i)
      nodeData[i].data.data = graph.getData(*ii);
  }
};

//! Local computation graph (i.e., graph structure does not change)
template<typename EdgeTy>
class LC_FileGraph<void, EdgeTy> : public FileGraph {
  struct gNode : public Galois::Runtime::Lockable {
    gNode() {}
  };

  //null if type is void
  Galois::Runtime::LL::CacheLineStorage<gNode>* nodeData;

public:
  GALOIS_ATTRIBUTE_DEPRECATED
  LC_FileGraph(): nodeData(0) {}
  ~LC_FileGraph() {
    if (nodeData)
      delete[] nodeData;
  }
  
  EdgeTy& getEdgeData(GraphNode src, GraphNode dst, MethodFlag mflag = MethodFlag::ALL) {
    return FileGraph::getEdgeData<EdgeTy>(src, dst, mflag);
  }
  EdgeTy& getEdgeData(FileGraph::edge_iterator it, MethodFlag mflag = MethodFlag::ALL) {
    return FileGraph::getEdgeData<EdgeTy>(it, mflag);
  }
  EdgeTy& getEdgeData(FileGraph::neighbor_iterator it, MethodFlag mflag = MethodFlag::ALL) {
    return FileGraph::getEdgeData<EdgeTy>(it, mflag);
  }
  neighbor_iterator neighbor_begin(GraphNode N, MethodFlag mflag = MethodFlag::ALL) const {
    return FileGraph::neighbor_begin(N);
  }
  neighbor_iterator neighbor_end(GraphNode N, MethodFlag mflag = MethodFlag::ALL) const {
    return FileGraph::neighbor_end(N);
  }
};

template<typename NodeTy>
class LC_FileGraph<NodeTy, void>: public FileGraph { 
  struct gNode : public Galois::Runtime::Lockable {
    NodeTy data;
    gNode() {}
  };

  //null if type is void
  Galois::Runtime::LL::CacheLineStorage<gNode>* nodeData;

public:
  GALOIS_ATTRIBUTE_DEPRECATED
  LC_FileGraph(): nodeData(0) {}
  ~LC_FileGraph() {
    if (nodeData)
      delete[] nodeData;
  }
  
  NodeTy& getData(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    Galois::Runtime::acquire(&nodeData[N].data, mflag);
    return nodeData[N].data.data;
  }

  void nodeDataFromFile(const char* filename) {
    emptyNodeData();
    std::ifstream file(filename);
    for (uint64_t i = 0; i < numNodes; ++i)
      file >> nodeData[i];
  }
  
  void emptyNodeData(NodeTy init = NodeTy()) {
    nodeData = new Galois::Runtime::LL::CacheLineStorage<gNode>[numNodes];
    for (uint64_t i = 0; i < numNodes; ++i)
      nodeData[i].data.data = init;
  }

  neighbor_iterator neighbor_begin(GraphNode N, MethodFlag mflag = MethodFlag::ALL) const {
    return FileGraph::neighbor_begin(N);
  }

  neighbor_iterator neighbor_end(GraphNode N, MethodFlag mflag = MethodFlag::ALL) const {
    return FileGraph::neighbor_end(N);
  }
};

template<>
class LC_FileGraph<void, void>: public FileGraph { 
public:
  GALOIS_ATTRIBUTE_DEPRECATED
  LC_FileGraph() { }
  neighbor_iterator neighbor_begin(GraphNode N, MethodFlag mflag = MethodFlag::ALL) const {
    return FileGraph::neighbor_begin(N);
  }
  neighbor_iterator neighbor_end(GraphNode N, MethodFlag mflag = MethodFlag::ALL) const {
    return FileGraph::neighbor_end(N);
  }
};

}
}
#endif
