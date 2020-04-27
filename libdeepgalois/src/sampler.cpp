#include "deepgalois/sampler.h"
#include <time.h> 
#include <vector>

inline unsigned getDegree(Graph &g, GNode v) {
	return std::distance(g.edge_begin(v), g.edge_end(v));
}

namespace deepgalois {

// Utility function to randomly select k vertices from [begin, end)
template <typename T = int>
T* Sampler::select_k_items(T k, T begin, T end) {
    auto i = begin;
  
    // reservoir[] is the output array. Initialize  
    // it with first k vertices 
    T *reservoir = new T[k];
    for (; i < k; i++) reservoir[i] = i;
  
    // Use a different seed value so that we don't get  
    // same result each time we run this program  
    srand(time(NULL));  
  
    // Iterate from the (k+1)th element to nth element  
    for (; i < end; i++) {  
        // Pick a random index from 0 to i.  
        auto j = rand() % (i + 1);  
  
        // If the randomly picked index is smaller than k,  
        // then replace the element present at the index  
        // with new element from stream  
        if (j < k) reservoir[j] = i;
    }
	return reservoir;
}

// Utility function to find ceiling of r in arr[l..h]
template <typename T = int>
inline T Sampler::findCeil(std::vector<T> arr, T r, T l, T h) {  
	T mid;
	while (l < h) {
		mid = l + ((h - l) >> 1); // Same as mid = (l+h)/2
		(r > arr[mid]) ? (l = mid + 1) : (h = mid);
	}
	return (arr[l] >= r) ? l : -1;  
} 

// Utility function to select one element from n elements given a frequency (probability) distribution
// https://www.geeksforgeeks.org/random-number-generator-in-arbitrary-probability-distribution-fashion/
template <typename T = int>
T Sampler::select_one_item(T n, std::vector<T> dist) {
	std::vector<T> offsets(n);
	offsets[0] = dist[0];
	// compute the prefix sum of the distribution
	for (T i = 1; i < n; ++i) offsets[i] = offsets[i-1] + dist[i];
	// offsets[n-1] is sum of all frequencies
	T sum = offsets[n-1];
	T r = (rand() % sum) + 1;
	// find which range r falls into, and return the index of the range
	return findCeil(offsets, r, 0, n - 1);
}

// Given a subset of vertices and a graph g, generate a subgraph sg from the graph g
void Sampler::generate_subgraph(VertexList &vertex_set, Graph &g, Graph &sub) {
	auto nv = vertex_set.size();
	size_t ne = 0;
	std::vector<unsigned> offsets(nv+1);
	offsets[0] = 0;
	size_t i = 0;
	VertexList vertices(nv);
	for (auto v : vertex_set) {
		vertices[i] = v;
		offsets[i+1] = offsets[i] + getDegree(g, v);
		i++;
	}
	// TODO: need to remove edges whose has endpoint not belong to the selected vertex subset
	sub.allocateFrom(nv, ne);
	sub.constructNodes();
	for (i = 0; i < nv; i++) {
		g.fixEndEdge(i, offsets[i+1]);
		for (unsigned offset = 0; offset < offsets[i+1]-offsets[i]; offset ++) {
			g.constructEdge(offsets[i]+offset, g.getEdgeDst(g.edge_begin(vertices[i])+offset), 0);
		}
	}
}

void Sampler::generate_masked_graph(size_t n, mask_t *masks, Graph &g, Graph &sub) {
  std::vector<uint32_t> degrees(n, 0);
  galois::do_all(galois::iterate(g), [&](const GNode src) {
    if (masks[src] == 1) {
      for (const auto e : g.edges(src)) {
        const auto dst = g.getEdgeDst(e);
        if (masks[dst] == 1) degrees[src] ++;
      }
    }
  }, galois::loopname("update_degrees"));
  std::vector<uint32_t> offsets(n+1);
  offsets[0] = 0;
  for (size_t i = 0; i < n; i ++) {
    offsets[i+1] = offsets[i] + degrees[i];
  }
  size_t ne = offsets[n];
  sub.allocateFrom(n, ne);
  sub.constructNodes();
  galois::do_all(galois::iterate(sub), [&](const GNode src) {
    g.fixEndEdge(src, offsets[src+1]);
    if (masks[src] == 1) {
      auto idx = offsets[src];
      for (const auto e : g.edges(src)) {
        const auto dst = g.getEdgeDst(e);
        if (masks[dst] == 1) g.constructEdge(idx++, dst, 0);
      }
    }
  }, galois::loopname("gen_subgraph"));
}

// !API function for user-defined selection strategy
// Select n vertices from vertices and put them in vertex_set.
// nv: number of vertices in the original graph;
// n: number of vertices in the subgraph;
// m: number of vertices in the frontier.
void Sampler::select_vertices(size_t nv, size_t n, int m, Graph &g, VertexList vertices, VertexList &vertex_set) {
  assert(nv == vertices.size());
  assert(n == vertex_set.size());
  auto frontier_indices = select_k_items(m, 0, (int)nv); // randomly select m vertices from vertices as frontier
  VertexList frontier(m);
  for (int i = 0; i < m; i++) vertex_set[i] = frontier[i] = vertices[frontier_indices[i]];
  std::vector<int> degrees(m);
  galois::do_all(galois::iterate(g.begin(), g.end()), [&](const auto i) {
    degrees[i] = getDegree(g, frontier[i]);
  }, galois::loopname("compute_degrees"));
  for (size_t i = 0; i < n - m; i++) {
    auto pos = select_one_item((int)m, degrees);
    auto u = frontier[pos];
    auto degree = degrees[pos];
    auto neighbor_id = rand() % degree; // randomly select a neighbor
    auto dst = g.getEdgeDst(g.edge_begin(u) + neighbor_id);
    frontier[pos] = dst;
    degrees[pos] = getDegree(g, frontier[pos]);
    vertex_set.push_back(u);
  }
}

void update_masks(size_t n, VertexList vertices, mask_t *masks) {
	std::fill(masks, masks+n, 0);
	for (auto v : vertices) masks[v] = 1;
}

void Sampler::subgraph_sample(size_t n, Graph&sg, mask_t *masks) {
  VertexList vertex_set(n);
  select_vertices(count_, n, m_, masked_graph, vertices_, vertex_set); 
  generate_subgraph(vertex_set, masked_graph, sg);
  update_masks(graph->size(), vertex_set, masks);
}

} // end namespace

