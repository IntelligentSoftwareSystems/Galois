/** Basic serialized graphs -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
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
 * There are two main classes, ::FileGraph and ::LC_FileGraph. The former
 * represents the pure structure of a graph (i.e., whether an edge exists between
 * two nodes) and cannot be modified. The latter allows values to be stored on
 * nodes and edges, but the structure of the graph cannot be modified.
 *
 * An example of use:
 * 
 * \code
 * typedef Galois::Graph::LC_FileGraph<int,int> Graph;
 * 
 * // Create graph
 * Graph g;
 * g.structureFromFile(inputfile);
 *
 * // Traverse graph
 * for (Graph::active_iterator i = g.active_begin(), iend = g.active_end();
 *      i != iend;
 *      ++i) {
 *   Graph::GraphNode src = *i;
 *   for (Graph::neighbor_iterator j = g.neighbor_begin(src),
 *                                 jend = g.neighbor_end(src);
 *        j != jend;
 *        ++j) {
 *     Graph::GraphNode dst = *j;
 *     int edgeData = g.getEdgeData(src, dst);
 *     int nodeData = g.getData(dst);
 *   }
 * }
 * \endcode
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#ifndef GALOIS_GRAPHS_FILEGRAPH_H
#define GALOIS_GRAPHS_FILEGRAPH_H

// TODO(ddn): needed until we move method flags out of Graph.h
#include "Galois/Graphs/Graph.h"
#include <boost/iterator/counting_iterator.hpp>

using namespace GaloisRuntime;

namespace Galois {
namespace Graph {

//! Graph serialized to a file
class FileGraph {
public:
  typedef uint32_t GraphNode;

protected:
  void* masterMapping;
  size_t masterLength;
  uint64_t sizeEdgeTy;
  int masterFD;

  uint64_t* outIdx;
  uint32_t* outs;

  char* edgeData;

  uint64_t numEdges;
  uint64_t numNodes;

  uint64_t getEdgeIdx(GraphNode src, GraphNode dst) {
    for (neighbor_iterator ii = neighbor_begin(src),
	   ee = neighbor_end(src); ii != ee; ++ii)
      if (*ii == dst)
	return std::distance(outs, ii);
    return ~(uint64_t)0;
  }

public:
  // Node Handling

  //! Check if a node is in the graph (already added)
  bool containsNode(const GraphNode n) {
    return n < numNodes;
  }

  // Edge Handling
  template<typename EdgeTy>
  EdgeTy& getEdgeData(GraphNode src, GraphNode dst, MethodFlag mflag = ALL) {
    assert(sizeEdgeTy == sizeof(EdgeTy));
    return ((EdgeTy*)edgeData)[getEdgeIdx(src,dst)];
  }

  void prefetch_edges(GraphNode N) {
    __builtin_prefetch(neighbor_begin(N, NONE));
  }

  template<typename EdgeTy>
  void prefetch_edgedata(GraphNode N) {
    __builtin_prefetch(
        &((EdgeTy*)edgeData)[std::distance(outs, neighbor_begin(N))]);
  }

  void prefetch_pre(GraphNode N) {
    if (N != 0)
      __builtin_prefetch(&outIdx[N-1]);
    __builtin_prefetch(&outIdx[N]);
  }

  // General Things

  typedef uint32_t* neighbor_iterator;

  neighbor_iterator neighbor_begin(GraphNode N, MethodFlag mflag = ALL) {
    if (N == 0)
      return &outs[0];
    else
      return &outs[outIdx[N-1]];
  }

  neighbor_iterator neighbor_end(GraphNode N, MethodFlag mflag = ALL) {
    return &outs[outIdx[N]];
  }

  bool has_neighbor(GraphNode N1, GraphNode N2, MethodFlag mflag = ALL) {
    return std::find(neighbor_begin(N1), neighbor_end(N1), N2) != neighbor_end(N1);
  }

  typedef boost::counting_iterator<uint64_t> active_iterator;

  //! Iterate over nodes in graph (not thread safe)
  active_iterator active_begin() {
    return active_iterator(0);
  }

  active_iterator active_end() {
    return active_iterator(numNodes);
  }

  //! The number of nodes in the graph
  unsigned int size() {
    return numNodes;
  }

  FileGraph();
  ~FileGraph();

  //! Read graph connectivity information from file
  bool structureFromFile(const char* filename);
};

//! Local computation graph (i.e., graph structure does not change)
template<typename NodeTy, typename EdgeTy>
class LC_FileGraph : public FileGraph {

  struct gNode : public GaloisRuntime::Lockable {
    NodeTy data;
    gNode() {}
  };

  //null if type is void
  cache_line_storage<gNode>* NodeData;

public:
  LC_FileGraph() :NodeData(0) {}
  ~LC_FileGraph() {
    if (NodeData)
      delete[] NodeData;
  }
  
  NodeTy& getData(GraphNode N, MethodFlag mflag = ALL) {
    GaloisRuntime::acquire(&NodeData[N].data, mflag);
    return NodeData[N].data.data;
  }

  EdgeTy& getEdgeData(GraphNode src, GraphNode dst, MethodFlag mflag = ALL) {
    return FileGraph::getEdgeData<EdgeTy>(src, dst, mflag);
  }

  //! Loads node data from file
  void nodeDataFromFile(const char* filename) {
    emptyNodeData();
    std::ifstream file(filename);
    for (uint64_t i = 0; i < numNodes; ++i)
      file >> NodeData[i];
  }
  
  //! Initializes node data for the graph to default values
  void emptyNodeData(NodeTy init = NodeTy()) {
    NodeData = new cache_line_storage<gNode>[numNodes];
    for (uint64_t i = 0; i < numNodes; ++i)
      NodeData[i].data.data = init;
  }

  void prefetch_edgedata(GraphNode N) {
    FileGraph::prefetch_edgedata<EdgeTy>(N);
  }

  void prefetch_neighbors(GraphNode N) {
    for (neighbor_iterator ii = neighbor_begin(N, NONE),
        ee = neighbor_begin(N,NONE); ii != ee; ++ii)
      __builtin_prefetch(&NodeData[*ii].data.data);
  }

};

template<typename NodeTy>
class LC_FileGraph<NodeTy, void>: public FileGraph { 

  struct gNode : public GaloisRuntime::Lockable {
    NodeTy data;
    gNode() {}
  };

  //null if type is void
  cache_line_storage<gNode>* NodeData;

public:
  LC_FileGraph() :NodeData(0) {}
  ~LC_FileGraph() {
    if (NodeData)
      delete[] NodeData;
  }
  
  NodeTy& getData(GraphNode N, MethodFlag mflag = ALL) {
    GaloisRuntime::acquire(&NodeData[N].data, mflag);
    return NodeData[N].data.data;
  }

  void nodeDataFromFile(const char* filename) {
    emptyNodeData();
    std::ifstream file(filename);
    for (uint64_t i = 0; i < numNodes; ++i)
      file >> NodeData[i];
  }
  
  void emptyNodeData(NodeTy init = NodeTy()) {
    NodeData = new cache_line_storage<gNode>[numNodes];
    for (uint64_t i = 0; i < numNodes; ++i)
      NodeData[i].data.data = init;
  }

  void prefetch_neighbors(GraphNode N) {
    for (neighbor_iterator ii = neighbor_begin(N, NONE),
        ee = neighbor_begin(N,NONE); ii != ee; ++ii)
      __builtin_prefetch(&NodeData[*ii].data.data);
  }
};

template<>
class LC_FileGraph<void, void>: public FileGraph { 
};

}
}
#endif
