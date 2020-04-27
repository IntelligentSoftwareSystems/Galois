#pragma once
#include "deepgalois/gtypes.h"

namespace deepgalois {
class Sampler {
public:
  Sampler() : m_(1000) {}
  ~Sampler() {}

  // sample a subgraph sg of size n from graph g
  void subgraph_sample(size_t n, Graph &sg, mask_t *masks);

  // !API function for user-defined selection strategy
  virtual void select_vertices(size_t nv, size_t n, int m, Graph &g, VertexList vertices, VertexList &vertex_set);

  galois::runtime::iterable<galois::NoDerefIterator<Graph::edge_iterator> > neighbor_sampler(Graph &g, GNode v);

  Graph::edge_iterator sampled_edge_begin(Graph &g, GNode v) { return g.edge_begin(v); }

  Graph::edge_iterator sampled_edge_end(Graph &g, GNode v) { return g.edge_end(v); }

  void set_masked_graph(size_t begin, size_t end, size_t count, mask_t *masks, Graph *g) {
    begin_ = begin;
    end_ = end;
    count_ = count;
    masks_ = masks;
    graph = g;
    generate_masked_graph(count, masks, *g, masked_graph);
    size_t idx = 0;
    vertices_.resize(count);
    for (size_t i = begin; i < end; i++) {
      if (masks_[i] == 1) vertices_[idx++] = i;
    }
  }

protected:
  int m_;
  size_t count_;
  size_t begin_;
  size_t end_;
  VertexList vertices_;
  mask_t *masks_;
  Graph masked_graph;
  Graph *graph;

  // Utility function to randomly select k items from [begin, end)
  template <typename T = uint32_t>
  T* select_k_items(T k, T begin, T end);

  // Utility function to find ceiling of r in arr[l..h]
  template <typename T = int>
  inline T findCeil(std::vector<T> arr, T r, T l, T h);

  // Utility function to select one element from n elements given a frequency (probability) distribution
  template <typename T = int>
  T select_one_item(T n, std::vector<T> dist);

  // Given a subset of vertices and a graph g, generate a subgraph sg from the graph g
  void generate_subgraph(VertexList &vertex_set, Graph &g, Graph &sub);
  void generate_masked_graph(size_t n, mask_t *masks, Graph &g, Graph &mg);
};

}
