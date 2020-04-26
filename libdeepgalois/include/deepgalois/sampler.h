#pragma once
#include "deepgalois/gtypes.h"

namespace deepgalois {
class Sampler {
public:
  Sampler() : m(1000) {}
  ~Sampler() {}

  // sample a subgraph sg of size n from graph g
  void subgraph_sample(size_t n, Graph &g, Graph &sg, VertexList &vertex_set, mask_t *masks);

  // !API function for user-defined selection strategy
  virtual void select_vertices(Graph &g, VertexList &vertex_set, size_t n, size_t m);

  galois::runtime::iterable<galois::NoDerefIterator<Graph::edge_iterator> > neighbor_sampler(Graph &g, GNode v);

  Graph::edge_iterator sampled_edge_begin(Graph &g, GNode v) { return g.edge_begin(v); }

  Graph::edge_iterator sampled_edge_end(Graph &g, GNode v) { return g.edge_end(v); }

protected:
  size_t m;
  // Utility function to randomly select k items from [begin, end)
  VertexList selectVertex(GNode begin, GNode end, size_t k);
  // Utility function to find ceiling of r in arr[l..h]
  inline int findCeil(std::vector<unsigned> arr, unsigned r, unsigned l, unsigned h);
  // Utility function to select one element from n elements given a frequency (probability) distribution
  size_t selectOneVertex(size_t n, std::vector<unsigned> dist);
  // Given a subset of vertices and a graph g, generate a subgraph sg from the graph g
  void generate_subgraph(VertexList &vertex_set, Graph &g, Graph &sub);
};

}
