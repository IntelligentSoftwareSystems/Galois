#pragma once
#include "deepgalois/gtypes.h"

namespace deepgalois {
class Sampler {
public:
  Sampler() : m_(1000) {}
  ~Sampler() {}

  // sample a subgraph sg of size n from graph g
  void subgraph_sample(size_t n, Graph &sg, mask_t* masks);

  // !API function for user-defined selection strategy
  virtual void select_vertices(size_t nv, size_t n, int m, Graph* g, VertexList vertices, VertexSet &vertex_set);

  galois::runtime::iterable<galois::NoDerefIterator<edge_iterator> > neighbor_sampler(Graph &g, VertexID v);

  edge_iterator sampled_edge_begin(Graph &g, VertexID v) { return g.edge_begin(v); }

  edge_iterator sampled_edge_end(Graph &g, VertexID v) { return g.edge_end(v); }

  void set_masked_graph(size_t begin, size_t end, size_t count, mask_t* masks, Graph* g);

protected:
  int m_;
  size_t count_;
  size_t begin_;
  size_t end_;
  VertexList vertices_;
  mask_t *masks_;
  Graph *masked_graph;
  Graph *graph;

  // Given a subset of vertices and a graph g, generate a subgraph sg from the graph g
  void generate_subgraph(VertexSet &vertex_set, Graph &g, Graph &sub);
  void generate_masked_graph(size_t n, mask_t* masks, Graph* g, Graph& mg);
  void get_masked_degrees(size_t n, mask_t* masks, Graph* g, std::vector<uint32_t> &degrees);
  void update_masks(size_t n, VertexSet vertices, mask_t* masks);
  inline VertexList reindexing_vertice(size_t n, VertexSet vertex_set);
};

}
