#ifndef EGONET_H_
#define EGONET_H_

#include "Mining/types.h"
typedef unsigned IndexTy;
typedef unsigned VertexId;
typedef unsigned char BYTE;
typedef unsigned long Ulong;
typedef galois::gstl::Vector<BYTE> ByteList;
typedef galois::gstl::Vector<unsigned> UintList;
typedef galois::gstl::Vector<Ulong> UlongList;
typedef galois::gstl::Vector<VertexId> VertexList;
typedef galois::gstl::Vector<UintList> IndexLists;
typedef galois::gstl::Vector<ByteList> ByteLists;
typedef galois::gstl::Vector<VertexList> VertexLists;
typedef galois::gstl::Set<VertexId> VertexSet;
typedef galois::substrate::PerThreadStorage<UintList> Lists;

class EmbeddingList {
public:
	EmbeddingList() {}
	EmbeddingList(int core, unsigned k) {
		allocate(core, k);
	}
	void allocate(int core, unsigned k) {
		max_level = k;
		cur_level = k-1;
		sizes.resize(k);
		label.resize(core);
		for (unsigned i = 0; i < k; i ++) sizes[i] = 0;
		vid_lists.resize(k);
		for (unsigned i = 2; i < k; i ++) vid_lists[i].resize(core);
	}
	~EmbeddingList() {}
	size_t size() const { return sizes[cur_level]; }
	size_t size(unsigned level) { return sizes[level]; }
	unsigned get_vertex(unsigned level, unsigned i) { return vid_lists[level][i]; }
	BYTE get_label(unsigned vid) { return label[vid]; }
	unsigned get_level() { return cur_level; }
	void set_size(unsigned level, unsigned size) { sizes[level] = size; }
	void set_vertex(unsigned level, unsigned i, unsigned value) { vid_lists[level][i] = value; }
	void set_label(unsigned vid, BYTE value) { label[vid] = value; }
	void set_level(unsigned level) { cur_level = level; }
protected:
	VertexLists vid_lists;
	UintList sizes; //sizes[level]: no. of embeddings (i.e. no. of vertices in the the current level)
	ByteList label;//label[i] is the label of each vertex i that shows its current level
	unsigned max_level;
	unsigned cur_level;
};
typedef galois::substrate::PerThreadStorage<EmbeddingList> EmbeddingLists;

class Egonet {
public:
	Egonet() {}
	Egonet(unsigned c, unsigned k) {
		allocate(c, k);
	}
	~Egonet() {}
	void allocate(int core, unsigned k) {
		max_size = k;
		degrees.resize(k);
		for (unsigned i = 2; i < k; i ++) degrees[i].resize(core);
		adj.resize(core*core);
		//emb_list.allocate(core, k);
	}
	unsigned get_adj(unsigned vid) { return adj[vid]; }
	unsigned get_degree(unsigned level, unsigned i) { return degrees[level][i]; }
	void set_adj(unsigned vid, unsigned value) { adj[vid] = value; }
	void set_degree(unsigned level, unsigned i, unsigned degree) { degrees[level][i] = degree; }
	void inc_degree(unsigned level, unsigned i) { degrees[level][i] ++; }
	//unsigned get_vertex(unsigned level, unsigned i) { return emb_list.get_vertex(level, i); }
	//void set_vertex(unsigned level, unsigned i, unsigned value) { emb_list.set_vertex(level, i, value); }
protected:
	unsigned max_size;
	UintList adj;//truncated list of neighbors
	IndexLists degrees;//degrees[level]: degrees of the vertices in the egonet
	//EmbeddingList emb_list;
};
typedef galois::substrate::PerThreadStorage<Egonet> Egonets;

class DfsMiner {
public:
	DfsMiner(Graph *g, unsigned c, unsigned size = 3) {
		graph = g;
		core = c;
		max_size = size;
		total_num.reset();
		for (int i = 0; i < numThreads; i++) {
			egonets.getLocal(i)->allocate(core, size);
			emb_lists.getLocal(i)->allocate(core, size);
		}
	}
	virtual ~DfsMiner() {}
	// construct the subgraph induced by vertex u's neighbors
	void mksub(const GNode &u, Egonet &egonet, EmbeddingList &emb_list, UintList ids) {
		//UintList ids(graph->size(), (unsigned)-1);
		ids.resize(graph->size());
		for (unsigned i = 0; i < ids.size(); i ++) ids[i] = (unsigned)-1;
		unsigned level = max_size-1;
		if(debug) printf("\n\n=======================\ninit: u = %d, level = %d\n", u, level);
		for (unsigned i = 0; i < emb_list.size(level); i ++) emb_list.set_label(i, 0);
		unsigned j = 0;
		for (auto e : graph->edges(u)) {
			auto v = graph->getEdgeDst(e);
			ids[v] = j;
			emb_list.set_label(j, level);
			emb_list.set_vertex(level, j, j);
			egonet.set_degree(level, j, 0);//new degrees
			if(debug) printf("init: v[%d] = %d\n", j, v);
			j ++;
		}
		if(debug) printf("init: num_neighbors = %d\n", j);
		emb_list.set_size(level, j); // number of neighbors of u. Since u is in level k, u's neighbors are in level k-1
		unsigned i = 0;
		//reodering adjacency list and computing new degrees
		for (auto e0 : graph->edges(u)) {
			auto v = graph->getEdgeDst(e0);
			// intersection of two neighbor lists
			for (auto e : graph->edges(v)) {
				auto dst = graph->getEdgeDst(e); // dst is the neighbor's neighbor
				unsigned new_id = ids[dst];
				if (new_id != (unsigned)-1) { // if dst is also a neighbor of u
					//if (max_size == 3) total_num += 1; //listing here!!!
					//else {
						unsigned degree = egonet.get_degree(level, i);
						egonet.set_adj(core * i + degree, new_id); // relabel
						egonet.set_degree(level, i, degree+1);
					//}
				}
			}
			if(debug) printf("vertex %d, number of common neighbors: %d\n", v, egonet.get_degree(level, i));
			i ++;
		}
	}
	// each task extends from a vertex, level starts from k-1 and decreases until level=2
	void dfs_extend(unsigned level, Egonet &egonet, EmbeddingList &emb_list) {
		//emb_list.set_level(level);
		if(debug) printf("debug: level = %d\n", level);
		if (level == 2) {
			for(unsigned i = 0; i < emb_list.size(level); i++) { //list all edges
				unsigned u = emb_list.get_vertex(level, i);
				unsigned begin = u * core;
				unsigned end = begin + egonet.get_degree(level, u);
				for (unsigned j = begin; j < end; j ++) {
					total_num += 1; //listing here!!!
				}
			}
			return;
		}
		// compute the subgraphs induced on the neighbors of each node in current level,
		// and then recurse on such a subgraph
		for(unsigned i = 0; i < emb_list.size(level); i ++) {
			// for each vertex u in current level
			// a new induced subgraph G[∆G(u)] is built
			unsigned u = emb_list.get_vertex(level, i);
			if(debug) printf("debug: u = %d\n", u);
			emb_list.set_size(level-1, 0);
			unsigned begin = u * core;
			unsigned end = begin + egonet.get_degree(level, u);
			// extend one vertex v which is a neighbor of u
			for (unsigned edge = begin; edge < end; edge ++) {
				// for each out-neighbor v of node u in G, set its label to level-1
				// if the label was equal to level. We thus have that if a label of a
				// node v is equal to level-1 it means that node v is in the new subgraph
				unsigned v = egonet.get_adj(edge);
				// update info of v
				// relabeling vertices and forming U'.
				if(debug) printf("\tdebug: v = %d, label = %d\n", v, emb_list.get_label(v));
				if (emb_list.get_label(v) == level) {
					unsigned pos = emb_list.size(level-1);
					emb_list.set_vertex(level-1, pos, v);
					emb_list.set_size(level-1, pos+1);
					emb_list.set_label(v, level-1);
					egonet.set_degree(level-1, v, 0);//new degrees
				}
			}
			// for each out-neighbor v of u
			// reodering adjacency list and computing new degrees
			unsigned new_size = emb_list.size(level-1);
			if(debug) printf("debug: u = %d, new_size = %d\n", u, new_size);
			for (unsigned j = 0; j < new_size; j ++) {
				unsigned v = emb_list.get_vertex(level-1, j);
				begin = v * core;
				end = begin + egonet.get_degree(level, v);
				// move all the out-neighbors of v with label equal to level − 1 
				// in the first part of ∆(v) (by swapping nodes),
				// and compute the out-degree of node v in the new subgraph
				// updating degrees(v). The first degrees(v) nodes in ∆(v) are
				// thus the out-neighbors of v in the new subgraph.
				for (unsigned k = begin; k < end; k ++) {
					unsigned dst = egonet.get_adj(k);
					if (emb_list.get_label(dst) == level-1)
						egonet.inc_degree(level-1, v);
					else {
						egonet.set_adj(k--, egonet.get_adj(--end));
						egonet.set_adj(end, dst);
					}
				}
			}
			dfs_extend(level-1, egonet, emb_list);
			for (unsigned j = 0; j < emb_list.size(level-1); j ++) {//restoring labels
				unsigned v = emb_list.get_vertex(level-1, j);
				emb_list.set_label(v, level);
			}
		}
	}
	void process() {
		//galois::do_all(galois::iterate((size_t)0, graph->size()),
		//galois::for_each(galois::iterate(graph->begin(), graph->end()),
		galois::do_all(galois::iterate(graph->begin(), graph->end()),
			[&](const size_t &u) {
			//[&](const size_t &u, auto &ctx) {
				Egonet *egonet = egonets.getLocal();
				EmbeddingList *emb_list = emb_lists.getLocal();
				UintList *id_list = id_lists.getLocal();
				mksub(u, *egonet, *emb_list, *id_list);
				//if (max_size > 3)
					dfs_extend(max_size-1, *egonet, *emb_list);
			},
			galois::chunk_size<CHUNK_SIZE>(), galois::steal(), 
			galois::loopname("KclSolver")
		);
	}
	Ulong get_total_count() { return total_num.reduce(); }
	void print_output() {
		std::cout << "\n\ttotal_num_cliques = " << get_total_count() << "\n\n";
	}

protected:
	Graph *graph;
	int core;
	unsigned max_size;
	UlongAccu total_num;
	EmbeddingLists emb_lists;
	Egonets egonets;
	Lists id_lists;
};

#endif
