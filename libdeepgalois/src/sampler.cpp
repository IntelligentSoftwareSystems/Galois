#include "deepgalois/sampler.h"
#include <time.h> 
#include <vector>

inline unsigned getDegree(Graph &g, GNode v) {
	return std::distance(g.edge_begin(v), g.edge_end(v));
}

namespace deepgalois {

// Utility function to randomly select k items from [begin, end)
VertexList Sampler::selectVertex(GNode begin, GNode end, size_t k) {
    auto i = begin;
  
    // reservoir[] is the output array. Initialize  
    // it with first k vertices 
    VertexList reservoir(k);
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
inline int Sampler::findCeil(std::vector<unsigned> arr, unsigned r, unsigned l, unsigned h) {  
	unsigned mid;
	while (l < h) {
		mid = l + ((h - l) >> 1); // Same as mid = (l+h)/2
		(r > arr[mid]) ? (l = mid + 1) : (h = mid);
	}
	return (arr[l] >= r) ? l : -1;  
} 

// Utility function to select one element from n elements given a frequency (probability) distribution
// https://www.geeksforgeeks.org/random-number-generator-in-arbitrary-probability-distribution-fashion/
size_t Sampler::selectOneVertex(size_t n, std::vector<unsigned> dist) {
	std::vector<unsigned> offsets(n);
	offsets[0] = dist[0];
	// compute the prefix sum of the distribution
	for (size_t i = 1; i < n; ++i) offsets[i] = offsets[i-1] + dist[i];
	// offsets[n-1] is sum of all frequencies
	unsigned sum = offsets[n-1];
	unsigned r = (rand() % sum) + 1;
	// find which range r falls into,
	// and return the index of the range
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

// !API function for user-defined selection strategy
// Select n vertices from graph g and put them in vertex_set.
// n: number of vertices in the subgraph;
// m: number of vertices in the frontier.
void Sampler::select_vertices(Graph &g, VertexList &vertex_set, size_t n, size_t m) {
	assert(n == vertex_set.size());
    auto num_vertices = g.size(); // number of vertices in the original input graph
    auto frontier = selectVertex(0, num_vertices, m); // randomly select m vertices from g as frontier
	for (size_t i = 0; i < m; i++) vertex_set[i] = frontier[i];
	std::vector<unsigned> degrees(m);
	//std::vector<float> probabilities(m);
	//unsigned sum_degree = 0;
	for (size_t i = 0; i < m; i++) {
		degrees[i] = getDegree(g, frontier[i]);
		//sum_degree += degrees[i];
	}
	for (size_t i = 0; i < n - m; i++) {
		//for (size_t i = 0; i < m; i++)
		//	probabilities[i] = (float)degrees[i] / (float)sum_degree;
		auto pos = selectOneVertex(m, degrees);
		GNode u = frontier[pos];
		auto degree = degrees[pos];
		auto neighbor_id = rand() % degree;
		frontier[pos] = g.getEdgeDst(g.edge_begin(u) + neighbor_id);
		degrees[pos] = getDegree(g, frontier[pos]);
		//sum_degree -= degree;
		//sum_degree += degrees[pos];
		vertex_set.push_back(u);
	}
}

void Sampler::subgraph_sampler(Graph &g, Graph&sg, size_t n) {
	VertexList vertex_set(n);
	select_vertices(g, vertex_set, n, m); 
	generate_subgraph(vertex_set, g, sg);
}

} // end namespace

