#ifndef EGONET_H
#define EGONET_H
#include "types.h"

class Egonet {
public:
	Egonet() {}
	Egonet(unsigned c, unsigned k) {
		allocate(c, k);
	}
	~Egonet() {}
	void allocate(unsigned c, unsigned k) {
		//std::cout << "Allocating memory for egonet...\n";
		length = c;
		max_level = k;
		cur_level = 1;
		degrees.resize(k);
		sizes.resize(k);
		//labels.resize(length);
		std::fill(sizes.begin(), sizes.end(), 0);
		//std::fill(labels.begin(), labels.end(), 0);
		for (unsigned i = 1; i < k; i ++) degrees[i].resize(length);
		adj.resize(length*length);
	}
	size_t size() const { return sizes[cur_level]; }
	size_t size(unsigned level) const { return sizes[level]; }
	unsigned get_adj(VertexId vid) const { return adj[vid]; }
	unsigned get_degree(unsigned i) const { return degrees[cur_level][i]; }
	unsigned get_degree(unsigned level, unsigned i) const { return degrees[level][i]; }
	//unsigned get_cur_level() const { return cur_level; }
	//BYTE get_label(VertexId vid) const { return labels[vid]; }
	void set_size(unsigned level, unsigned size) { sizes[level] = size; }
	void set_adj(VertexId vid, unsigned value) { adj[vid] = value; }
	//void set_label(VertexId vid, BYTE value) { labels[vid] = value; }
	void set_degree(unsigned level, unsigned i, unsigned degree) { degrees[level][i] = degree; }
	void set_cur_level(unsigned level) { cur_level = level; }
	void inc_degree(unsigned level, unsigned i) { degrees[level][i] ++; }

	VertexId getEdgeDst(VertexId vid) const { return adj[vid]; }
	VertexId edge_begin(VertexId vid) { return vid * length; }
	VertexId edge_end(VertexId vid) { return vid * length + get_degree(vid); }
	VertexId edge_end(unsigned level, VertexId vid) { return vid * length + get_degree(level, vid); }
	//void increase_level() { cur_level ++; }

protected:
	unsigned length;
	unsigned max_level;
	UintList adj;//truncated list of neighbors
	IndexLists degrees;//degrees[level]: degrees of the vertices in the egonet
	UintList sizes; //sizes[level]: no. of embeddings (i.e. no. of vertices in the the current level)
	//ByteList labels; //labels[i] is the label of each vertex i that shows its current level
	unsigned cur_level;
};

//typedef galois::substrate::PerThreadStorage<Egonet> Egonets;

#endif
