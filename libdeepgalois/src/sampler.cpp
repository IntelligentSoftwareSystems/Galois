#include "deepgalois/utils.h"
#include "deepgalois/sampler.h"
#include <time.h> 
#include <vector>

inline unsigned getDegree(Graph *g, GNode v) {
	return std::distance(g->edge_begin(v), g->edge_end(v));
}

namespace deepgalois {

void Sampler::set_masked_graph(size_t begin, size_t end, size_t count, mask_t *masks, Graph *g) {
  galois::gPrint("Set masked graph: begin=", begin, ", end=", end, ", count=", count, "\n");
  begin_ = begin;
  end_ = end;
  count_ = count;
  masks_ = masks;
  graph = g;
#ifndef GALOIS_USE_DIST
  masked_graph = new Graph();
#endif
  generate_masked_graph(g->size(), masks, g, *masked_graph);
  size_t idx = 0;
  vertices_.resize(count);
  for (size_t i = begin; i < end; i++) {
    if (masks_[i] == 1) vertices_[idx++] = i;
  }
}

void Sampler::get_masked_degrees(size_t n, mask_t *masks, Graph *g, std::vector<uint32_t> &degrees) {
  assert(degrees.size() == n);
  galois::do_all(galois::iterate(size_t(0), n), [&](const GNode src) {
    if (masks[src] == 1) {
      for (const auto e : g->edges(src)) {
        const auto dst = g->getEdgeDst(e);
        if (masks[dst] == 1) degrees[src] ++;
      }
    }
  }, galois::loopname("update_degrees"));
}

void Sampler::generate_masked_graph(size_t n, mask_t* masks, Graph* g, Graph& sub) {
  std::vector<uint32_t> degrees(n, 0);
  get_masked_degrees(n, masks, g, degrees);
  auto offsets = deepgalois::parallel_prefix_sum(degrees);
  size_t ne = offsets[n];
  galois::gPrint("Generate masked graph: num_vertices=", n, ", num_edges=", ne, "\n");
#ifndef GALOIS_USE_DIST
  sub.allocateFrom(n, ne);
  sub.constructNodes();
  galois::do_all(galois::iterate((size_t)0, n), [&](const GNode src) {
    sub.fixEndEdge(src, offsets[src+1]);
    if (masks[src] == 1) {
      auto idx = offsets[src];
      for (const auto e : g->edges(src)) {
        const auto dst = g->getEdgeDst(e);
        if (masks[dst] == 1) sub.constructEdge(idx++, dst, 0);
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
void Sampler::select_vertices(size_t nv, size_t n, int m, Graph *g, VertexList vertices, VertexSet &vertex_set) {
  galois::gPrint("Select a vertex set of size ", n, " from ", nv, " vertices, graph size: ", g->size(), "\n");
  assert(nv == vertices.size());
  auto frontier_indices = deepgalois::select_k_items(m, 0, (int)nv); // randomly select m vertices from vertices as frontier
  VertexList frontier(m);
  for (int i = 0; i < m; i++)
    frontier[i] = vertices[frontier_indices[i]];
  vertex_set.insert(frontier.begin(), frontier.end());
  galois::gPrint("vertex_set size: ", vertex_set.size(), "\n");
  int *degrees = new int[m];
  galois::do_all(galois::iterate(size_t(0), size_t(m)), [&](const auto i) {
    degrees[i] = (int)getDegree(g, frontier[i]);
  }, galois::loopname("compute_degrees"));
  for (size_t i = 0; i < n - m; i++) {
    auto pos = select_one_item((int)m, degrees);
    auto u = frontier[pos];
    auto degree = degrees[pos];
    int j =0;
    for (; j < degree; j ++) {
      auto neighbor_id = rand() % degree; // randomly select a neighbor
      auto dst = g->getEdgeDst(g->edge_begin(u) + neighbor_id);
      if (vertex_set.find(dst) == vertex_set.end()) {
        frontier[pos] = dst;
        degrees[pos] = getDegree(g, frontier[pos]);
        vertex_set.insert(dst);
        break;
      }
    }
    if (j == degree) galois::gPrint("Not found from ", degree, " neighbors\n");
  }
  assert(n == vertex_set.size());
  galois::gPrint("Done selection, vertex_set size: ", vertex_set.size(), ", set: ( ");
  unsigned counter = 0;
  for (int i : vertex_set) {
    counter ++;
    if (counter > 16 && counter < n-16) continue;
    galois::gPrint(i, " ");
  }
  galois::gPrint(" )\n");
}

void Sampler::update_masks(size_t n, VertexSet vertices, mask_t *masks) {
  galois::gPrint("Updating masks, size = ", vertices.size(), "\n");
  std::fill(masks, masks+n, 0);
  for (auto v : vertices) masks[v] = 1;
}

inline VertexList Sampler::reindexing_vertice(size_t n, VertexSet vertex_set) {
  VertexList new_ids(n, 0);
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
  VertexList new_ids = reindexing_vertice(graph->size(), vertex_set);
  std::vector<uint32_t> degrees(nv, 0); // degrees of vertices in the subgraph
  for (auto v : vertex_set) {
	degrees[new_ids[v]] = std::distance(g.edge_begin(v), g.edge_end(v));
  }
  auto offsets = deepgalois::parallel_prefix_sum(degrees);
  auto ne = offsets[nv];
  galois::gPrint("Generate subgraph: num_vertices=", nv, ", num_edges=", ne, "\n");
#ifndef GALOIS_USE_DIST
  sub.allocateFrom(nv, ne);
  sub.constructNodes();
  VertexList old_ids(vertex_set.begin(), vertex_set.end()); // vertex ID mapping
  galois::do_all(galois::iterate((size_t)0, nv), [&](const auto i) {
    sub.fixEndEdge(i, offsets[i+1]);
    unsigned j = 0;
    auto old_id = old_ids[i];
    for (auto e : g.edges(old_id)) {
      sub.constructEdge(offsets[i]+j, g.getEdgeDst(e), 0);
      j ++;
    }
  }, galois::loopname("construct_graph"));
#endif
}

void Sampler::subgraph_sample(size_t n, Graph&sg, mask_t *masks) {
  VertexSet vertex_set; // n = 9000 by default
  select_vertices(count_, n, m_, masked_graph, vertices_, vertex_set); // m = 1000 by default
  update_masks(graph->size(), vertex_set, masks); // set masks for vertices in the vertex_set
#ifndef GALOIS_USE_DIST
  Graph masked_sg;
  generate_masked_graph(graph->size(), masks, masked_graph, masked_sg); // remove edges whose destination is not masked
  generate_subgraph(vertex_set, masked_sg, sg);
#endif
}

} // end namespace

