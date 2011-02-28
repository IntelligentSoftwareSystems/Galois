// simple graph -*- C++ -*-

#include <boost/iterator/counting_iterator.hpp>

namespace Galois {
namespace Graph {

class FileGraph {
public:
  typedef uint64_t GraphNode;

private:
  void* masterMapping;
  size_t masterLength;
  int masterFD;

  uint64_t* outIdx;
  uint64_t* outs;
  uint64_t numNodes;
  uint64_t numEdges;

public:
  // Node Handling

  // Check if a node is in the graph (already added)
  bool containsNode(const GraphNode n) {
    return n < numNodes;
  }

  // General Things

  typedef uint64_t* neighbor_iterator;

  neighbor_iterator neighbor_begin(GraphNode N) {
    if (N == 0)
      return &outs[0];
    else
      return &outs[outIdx[N-1]];
  }

  neighbor_iterator neighbor_end(GraphNode N) {
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

  bool fromFile(const char* filename);

};

}
}
// vim: sw=2:ts=8
