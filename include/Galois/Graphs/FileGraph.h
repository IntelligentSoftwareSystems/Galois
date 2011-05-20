// simple graph -*- C++ -*-
/*
Galois, a framework to exploit amorphous data-parallelism in irregular
programs.

Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS SOFTWARE
AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY, FITNESS FOR ANY
PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF PERFORMANCE, AND ANY
WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF DEALING OR USAGE OF TRADE.
NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH RESPECT TO THE USE OF THE
SOFTWARE OR DOCUMENTATION. Under no circumstances shall University be liable
for incidental, special, indirect, direct or consequential damages or loss of
profits, interruption of business, or related expenses which may arise from use
of Software or Documentation, including but not limited to those resulting from
defects in Software and/or Documentation, or loss or inaccuracy of data of any
kind.
*/

#include <boost/iterator/counting_iterator.hpp>

//#include "Galois/Graphs/Graph.h"

using namespace GaloisRuntime;

namespace Galois {
namespace Graph {

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

  // Check if a node is in the graph (already added)
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
    __builtin_prefetch(&((EdgeTy*)edgeData)[std::distance(outs, neighbor_begin(N))]);
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

  //These are not thread safe!!
  typedef boost::counting_iterator<uint64_t> active_iterator;

  active_iterator active_begin() {
    return active_iterator(0);
  }

  active_iterator active_end() {
    return active_iterator(numNodes);
  }
  // The number of nodes in the graph
  unsigned int size() {
    return numNodes;
  }

  FileGraph();
  ~FileGraph();

  bool structureFromFile(const char* filename);
};

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
    if (shouldLock(mflag))
      GaloisRuntime::acquire(&NodeData[N].data);
    return NodeData[N].data.data;
  }

  EdgeTy& getEdgeData(GraphNode src, GraphNode dst, MethodFlag mflag = ALL) {
    return FileGraph::getEdgeData<EdgeTy>(src, dst, mflag);
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

  void prefetch_edgedata(GraphNode N) {
    FileGraph::prefetch_edgedata<EdgeTy>(N);
  }

  void prefetch_neighbors(GraphNode N) {
    for (neighbor_iterator ii = neighbor_begin(N, NONE), ee = neighbor_begin(N,NONE); ii != ee; ++ii)
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
    if (shouldLock(mflag))
      GaloisRuntime::acquire(&NodeData[N].data);
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
    for (neighbor_iterator ii = neighbor_begin(N, NONE), ee = neighbor_begin(N,NONE); ii != ee; ++ii)
      __builtin_prefetch(&NodeData[*ii].data.data);
  }
};

}
}
// vim:sw=2:ts=8:sts=2
