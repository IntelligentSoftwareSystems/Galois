#ifndef GALOIS_USE_DIST

#pragma once
#include "deepgalois/gtypes.h"

namespace deepgalois {
#define ETA 1.5          // length factor of DB in sampling
#define SAMPLE_CLIP 3000 // clip degree in sampling
#define DEFAULT_SIZE_FRONTIER 3000
#define DEFAULT_SIZE_SUBG 9000

class Sampler {
public:
  typedef int db_t;
  Sampler() : m_(DEFAULT_SIZE_FRONTIER) {}
  ~Sampler() {}

  // sample a subgraph sg of size n from graph g
  void subgraph_sample(size_t n, Graph& sg, mask_t* masks, unsigned tid = 0);

  // !API function for user-defined selection strategy
  virtual void select_vertices(size_t nv, size_t n, int m, Graph* g,
                               VertexList vertices, VertexSet& vertex_set);
  virtual void select_vertices(size_t n, int m, VertexSet& vertex_set,
                               unsigned tid);

  // galois::runtime::iterable<galois::NoDerefIterator<edge_iterator> >
  // neighbor_sampler(Graph &g, VertexID v);

  edge_iterator sampled_edge_begin(Graph& g, VertexID v) {
    return g.edge_begin(v);
  }

  edge_iterator sampled_edge_end(Graph& g, VertexID v) { return g.edge_end(v); }

  void set_masked_graph(size_t begin, size_t end, size_t count, mask_t* masks,
                        Graph* g);

protected:
  int m_;
  size_t count_;
  size_t begin_;
  size_t end_;
  int avg_deg;
  int subg_deg;
  VertexList vertices_;
  std::vector<index_t> node_train;
  mask_t* masks_;
  Graph* masked_graph;
  Graph* graph;

  // Given a subset of vertices and a graph g, generate a subgraph sg from the
  // graph g
  void generate_subgraph(VertexSet& vertex_set, Graph& g, Graph& sub);
  void generate_masked_graph(size_t n, mask_t* masks, Graph* g, Graph& mg);
  void get_masked_degrees(size_t n, mask_t* masks, Graph* g,
                          std::vector<uint32_t>& degrees);
  void update_masks(size_t n, VertexSet vertices, mask_t* masks);
  inline VertexList reindexing_vertice(size_t n, VertexSet vertex_set);
  void check_DB(std::vector<db_t>& DB0, std::vector<db_t>& DB1,
                std::vector<db_t>& DB2, size_t size);
};

} // namespace deepgalois

#endif
