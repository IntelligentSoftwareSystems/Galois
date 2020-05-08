#ifndef GALOIS_USE_DIST

#pragma once
#include "deepgalois/GraphTypes.h"

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

  //! sample a subgraph sg of size n from graph g
  //! sg is overwritten/is output
  void subgraph_sample(size_t n, Graph& sg, mask_t* masks, unsigned tid = 0);

  //! API function for user-defined selection strategy
  // TODO how to expose this?
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

  //! Given a mask, construct the graph with only those vertices ans ave as the
  //! masked graph in this class for the sampler.
  void initializeMaskedGraph(size_t count, mask_t* masks, Graph* g);

protected:
  int m_;
  size_t count_;

  //! averaged degree of masked graph
  int avg_deg;
  //! average degree cut off to a clip
  int subg_deg;
  //! list  of vertices active in the graph being maintained (masked_graph)
  //VertexList vertices_;
  //! List of training nodes; sampling set
  std::vector<index_t> node_train;
  mask_t* masks_;
  //! masked original graph; typically to the training set
  Graph* masked_graph;
  Graph* graph;

  //! Reindex a graph to only contain those in the vertex set
  void reindexSubgraph(VertexSet& keptVertices, Graph& g, Graph& reindexed);
  //! Given a graph, return a graph with edges to unmasked vertices removed in
  //! mg
  void getMaskedGraph(size_t n, mask_t* masks, Graph* g, Graph& mg);
  void get_masked_degrees(size_t n, mask_t* masks, Graph* g,
                          std::vector<uint32_t>& degrees);
  //! Set masks bitset with IDs in the vertices VertexSet
  void getMasks(size_t n, VertexSet vertices, mask_t* masks);
  inline VertexList reindexVertices(size_t n, VertexSet vertex_set);
  void checkGSDB(std::vector<db_t>& DB0, std::vector<db_t>& DB1,
                std::vector<db_t>& DB2, size_t size);
};

} // namespace deepgalois

#endif
