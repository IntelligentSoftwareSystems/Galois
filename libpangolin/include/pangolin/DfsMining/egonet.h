#ifndef EGONET_H
#define EGONET_H
#include "pangolin/types.h"

class Egonet {
public:
	Egonet() {}
	Egonet(unsigned c, unsigned k) { allocate(c, k); }
	~Egonet() {}

	void allocate(unsigned c, unsigned k) {
		//std::cout << "Allocating memory for egonet: c=" << c << ", k=" << k << "\n";
		length = c;
		max_level = k;
		cur_level = 1;
		degrees.resize(k);
		sizes.resize(k);
		std::fill(sizes.begin(), sizes.end(), 0);
		for (unsigned i = 1; i < k; i ++) degrees[i].resize(length);
		adj.resize(length*length);
	}

	size_t size() const { return sizes[cur_level]; }
	size_t size(unsigned level) const { return sizes[level]; }

	unsigned get_adj(VertexId vid) const { return adj[vid]; }
	unsigned get_degree(unsigned i) const { return degrees[cur_level][i]; }
	unsigned get_degree(unsigned level, unsigned i) const { return degrees[level][i]; }

	void set_size(unsigned level, unsigned size) { sizes[level] = size; }
	void set_adj(VertexId vid, unsigned value) { adj[vid] = value; }
	void set_degree(unsigned level, unsigned i, unsigned degree) { degrees[level][i] = degree; }
	void set_cur_level(unsigned level) { cur_level = level; }
	void inc_degree(unsigned level, unsigned i) { degrees[level][i] ++; }

	VertexId getEdgeDst(VertexId vid) const { return adj[vid]; }
	EdgeId edge_begin(VertexId vid) { return vid * length; }
	EdgeId edge_end(VertexId vid) { return vid * length + get_degree(vid); }
	EdgeId edge_end(unsigned level, VertexId vid) { return vid * length + get_degree(level, vid); }

protected:
	unsigned length;    // the core value of this subgraph
	unsigned max_level; // maximum number of levels
	unsigned cur_level; // current level
	UintList adj;       // truncated list of neighbors
	IndexLists degrees; // degrees[level]: degrees of the vertices in the egonet
	UintList sizes;     // sizes[level]: no. of embeddings (i.e. no. of vertices in the the current level)
};

#endif
