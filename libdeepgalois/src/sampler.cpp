#include "deepgalois/utils.h"
#include "deepgalois/sampler.h"
#include <time.h> 
#include <vector>

inline unsigned getDegree(Graph &g, GNode v) {
	return std::distance(g.edge_begin(v), g.edge_end(v));
}

namespace deepgalois {

void Sampler::get_masked_degrees(size_t n, mask_t *masks, Graph &g, std::vector<uint32_t> &degrees) {
  assert(degrees.size() == n);
  galois::do_all(galois::iterate(size_t(0), n), [&](const GNode src) {
    if (masks[src] == 1) {
      for (const auto e : g.edges(src)) {
        const auto dst = g.getEdgeDst(e);
        if (masks[dst] == 1) degrees[src] ++;
      }
    }
  }, galois::loopname("update_degrees"));
}

void Sampler::generate_masked_graph(size_t n, mask_t *masks, Graph &g, Graph *sub) {
  std::vector<uint32_t> degrees(n, 0);
  get_masked_degrees(n, masks, g, degrees);
  auto offsets = deepgalois::parallel_prefix_sum(degrees);
  size_t ne = offsets[n];
#ifndef GALOIS_USE_DIST
  sub = new Graph();
  sub->allocateFrom(n, ne);
  sub->constructNodes();
  galois::do_all(galois::iterate((size_t)0, n), [&](const GNode src) {
    g.fixEndEdge(src, offsets[src+1]);
    if (masks[src] == 1) {
      auto idx = offsets[src];
      for (const auto e : g.edges(src)) {
        const auto dst = g.getEdgeDst(e);
        if (masks[dst] == 1) g.constructEdge(idx++, dst, 0);
      }
    }
  }, galois::loopname("gen_subgraph"));
#endif
}

// !API function for user-defined selection strategy
// Select n vertices from vertices and put them in vertex_set.
// nv: number of vertices in the original graph;
// n: number of vertices in the subgraph;
// m: number of vertices in the frontier.
void Sampler::select_vertices(size_t nv, size_t n, int m, Graph &g, VertexList vertices, VertexSet &vertex_set) {
  assert(nv == vertices.size());
  auto frontier_indices = deepgalois::select_k_items(m, 0, (int)nv); // randomly select m vertices from vertices as frontier
  VertexList frontier(m);
  for (int i = 0; i < m; i++)
    frontier[i] = vertices[frontier_indices[i]];
  vertex_set.insert(frontier.begin(), frontier.end());
  int *degrees = new int[m];
  galois::do_all(galois::iterate(size_t(0), g.size()), [&](const auto i) {
    degrees[i] = (int)getDegree(g, frontier[i]);
  }, galois::loopname("compute_degrees"));
  for (size_t i = 0; i < n - m; i++) {
    auto pos = select_one_item((int)m, degrees);
    auto u = frontier[pos];
    auto degree = degrees[pos];
    auto neighbor_id = rand() % degree; // randomly select a neighbor
    auto dst = g.getEdgeDst(g.edge_begin(u) + neighbor_id);
    frontier[pos] = dst;
    degrees[pos] = getDegree(g, frontier[pos]);
    vertex_set.insert(u);
  }
  assert(n == vertex_set.size());
}

void Sampler::update_masks(size_t n, VertexSet vertices, mask_t *masks) {
	std::fill(masks, masks+n, 0);
	for (auto v : vertices) masks[v] = 1;
}

inline VertexList Sampler::reindexing_vertice(VertexSet vertex_set) {
  VertexList new_ids(vertex_set.size(), 0);
  int vid = 0;
  for (auto v : vertex_set) {
    new_ids[v] = vid++; // reindex
  }
  return new_ids;
}

// Given a subset of vertices and a graph g, generate a subgraph sg from the graph g
void Sampler::generate_subgraph(VertexSet &vertex_set, Graph &g, Graph &sub) {
  //auto n = g.size(); // old graph size
  auto nv = vertex_set.size(); // new graph (subgraph) size
  VertexList new_ids = reindexing_vertice(vertex_set);
  std::vector<uint32_t> degrees(nv, 0); // degrees of vertices in the subgraph
  for (auto v : vertex_set) {
	degrees[new_ids[v]] = std::distance(g.edge_begin(v), g.edge_end(v));
  }
  auto offsets = deepgalois::parallel_prefix_sum(degrees);
  auto ne = offsets[nv];
#ifndef GALOIS_USE_DIST
  sub.allocateFrom(nv, ne);
  sub.constructNodes();
  VertexList old_ids(vertex_set.begin(), vertex_set.end()); // vertex ID mapping
  galois::do_all(galois::iterate((size_t)0, nv), [&](const auto i) {
    g.fixEndEdge(i, offsets[i+1]);
    unsigned j = 0;
    auto old_id = old_ids[i];
    for (auto e : g.edges(old_id)) {
      g.constructEdge(offsets[i]+j, g.getEdgeDst(e), 0);
      j ++;
    }
  }, galois::loopname("construct_graph"));
#endif
}

void Sampler::subgraph_sample(size_t n, Graph&sg, mask_t *masks) {
  VertexSet vertex_set; // n = 9000 by default
  select_vertices(count_, n, m_, *masked_graph, vertices_, vertex_set); // m = 1000 by default
  update_masks(graph->size(), vertex_set, masks); // set masks for vertices in the vertex_set
  generate_masked_graph(n, masks, *masked_graph, &sg); // remove edges whose destination is not masked
  generate_subgraph(vertex_set, *masked_graph, sg);
}

} // end namespace

