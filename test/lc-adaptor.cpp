#include "galois/graphs/LC_Adaptor_Graph.h"
#include <boost/iterator/counting_iterator.hpp>

struct CSRArrays {
  int* outIdx;
  int* outs;
  int* nodeData;
  int numNodes;
  int numEdges;
};

class MyGraph : public galois::graphs::LC_Adaptor_Graph<int, void, MyGraph, int, boost::counting_iterator<int>, int*> {
  CSRArrays m_instance;

public:
  MyGraph(const CSRArrays& i): m_instance(i) { }

  size_t get_id(GraphNode n) const { return n; }

  node_data_reference get_data(GraphNode n) {
    return m_instance.nodeData[n];
  }

  edge_data_reference get_edge_data(edge_iterator n) {
    return {};
  }

  GraphNode get_edge_dst(edge_iterator n) {
    return *n;
  }

  int get_size() const { return m_instance.numNodes; }
  int get_size_edges() const { return m_instance.numEdges; }

  iterator get_begin() const { return iterator(0); }
  iterator get_end() const { return iterator(m_instance.numNodes); }

  edge_iterator get_edge_begin(GraphNode n) {
    return n == 0 ? &m_instance.outs[0] : &m_instance.outs[m_instance.outIdx[n-1]];
  }
  edge_iterator get_edge_end(GraphNode n) {
    return &m_instance.outs[m_instance.outIdx[n]];
  }
};

int main() {
  CSRArrays arrays;
  MyGraph g(arrays);
  return 0;
}
