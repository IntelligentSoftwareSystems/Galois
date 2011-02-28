// simple graph -*- C++ -*-

#include <boost/iterator/counting_iterator.hpp>

//#include "Galois/Graphs/Graph.h"

using namespace GaloisRuntime;

namespace Galois {
namespace Graph {

class FileGraph {
public:
  typedef uint64_t GraphNode;

protected:
  void* masterMapping;
  size_t masterLength;
  uint64_t sizeEdgeTy;
  int masterFD;

  uint64_t* outIdx;
  uint64_t* outs;

  char* edgeData;

  uint64_t numNodes;
  uint64_t numEdges;

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

  // General Things

  typedef uint64_t* neighbor_iterator;

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
  gNode* NodeData;

public:
  LC_FileGraph() :NodeData(0) {}
  ~LC_FileGraph() {
    if (NodeData)
      delete[] NodeData;
  }
  
  NodeTy& getData(GraphNode N, MethodFlag mflag = ALL) {
    if (shouldLock(mflag))
      GaloisRuntime::acquire(&NodeData[N]);
    return NodeData[N].data;
  }

  EdgeTy& getEdgeData(GraphNode src, GraphNode dst, MethodFlag mflag = ALL) {
    return FileGraph::getEdgeData<EdgeTy>(src,dst, mflag);
  }

  bool nodeDataFromFile(const char* filename) {
    emptyNodeData();
    std::ifstream file(filename);
    for (uint64_t i = 0; i < numNodes; ++i)
      file >> NodeData[i];
  }
  
  void emptyNodeData(NodeTy init = NodeTy()) {
    NodeData = new gNode[numNodes];
    for (uint64_t i = 0; i < numNodes; ++i)
      NodeData[i].data = init;
  }


};

}
}
// vim: sw=2:ts=8
