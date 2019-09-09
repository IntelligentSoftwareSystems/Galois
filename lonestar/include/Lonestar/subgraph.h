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
#ifdef ALGO_EDGE
#define BOTTOM 1
#else
#define BOTTOM 2
#endif

struct Edge {
	IndexTy src;
	IndexTy dst;
	Edge() : src(0), dst(0) {}
	Edge(IndexTy from, IndexTy to) : src(from), dst(to) {}
	std::string to_string() const {
		std::stringstream ss;
		ss << "e(" << src << "," << dst << ")";
		return ss.str();
	}
};

class EdgeList {
using iterator = typename galois::gstl::Vector<Edge>::iterator;
public:
	EdgeList() {}
	EdgeList(Graph& graph, bool is_dag = false) {
		init(graph, is_dag);
	}
	~EdgeList() {}
	bool empty() const { return edgelist.empty(); }
	iterator begin() { return edgelist.begin(); }
	iterator end() { return edgelist.end(); }
	size_t size() const { return edgelist.size(); }
	void resize (size_t n) { edgelist.resize(n); }
	void init(Graph& graph, bool is_dag = false) {
		size_t num_edges = graph.sizeEdges();
		if (!is_dag) num_edges = num_edges / 2;
		edgelist.resize(num_edges);
		if(is_dag) {
			galois::do_all(galois::iterate(graph.begin(), graph.end()),
				[&](const GNode& src) {
					for (auto e : graph.edges(src)) {
						GNode dst = graph.getEdgeDst(e);
						add_edge(*e, src, dst);
					}
				},
				galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::no_conflicts(),
				galois::loopname("Init-edgelist")
			);
		} else {
			size_t num_vertices = graph.size();
			UintList num_init_emb(num_vertices);
			galois::do_all(galois::iterate(graph.begin(), graph.end()),
				[&](const GNode& src) {
					num_init_emb[src] = 0;
					for (auto e : graph.edges(src)) {
						GNode dst = graph.getEdgeDst(e);
						if (src < dst) num_init_emb[src] ++;
					}
				},
				galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::no_conflicts(),
				galois::loopname("Init-edgelist-alloc")
			);
			UintList indices(num_vertices + 1);
			unsigned total = 0;
			for (size_t n = 0; n < num_vertices; n++) {
				indices[n] = total;
				total += num_init_emb[n];
			}
			indices[num_vertices] = total;
			galois::do_all(galois::iterate(graph.begin(), graph.end()),
				[&](const GNode& src) {
					IndexTy start = indices[src];
					for (auto e : graph.edges(src)) {
						GNode dst = graph.getEdgeDst(e);
						if (src < dst) {
							add_edge(start, src, dst);
							start ++;
						}
					}
				},
				galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::no_conflicts(),
				galois::loopname("Init-edgelist-insert")
			);
		}
	}
private:
	//std::vector<Edge> edgelist;
	galois::gstl::Vector<Edge> edgelist;
	void add_edge(unsigned pos, IndexTy src, IndexTy dst) {
		edgelist[pos] = Edge(src, dst);
	}
};

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
		for (unsigned i = BOTTOM; i < k; i ++) vid_lists[i].resize(core);
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
		for (unsigned i = BOTTOM; i < k; i ++) degrees[i].resize(core);
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
	DfsMiner(Graph *g, unsigned c, unsigned size = 3, bool use_dag = true) {
		graph = g;
		core = c;
		max_size = size;
		is_dag = use_dag;
		total_num.reset();
		for (int i = 0; i < numThreads; i++) {
			egonets.getLocal(i)->allocate(core, size);
			emb_lists.getLocal(i)->allocate(core, size);
		}
		edge_list.init(*g, is_dag);
	}
	virtual ~DfsMiner() {}
	// construct the subgraph induced by vertex u's neighbors
	void build_egonet_from_vertex(const GNode &u, Egonet &egonet, EmbeddingList &emb_list, UintList &ids, UintList &old_id) {
		if (old_id.empty()) {
			ids.resize(graph->size());
			old_id.resize(core);
			for (unsigned i = 0; i < graph->size(); i ++) ids[i] = (unsigned)-1;
		}
		//UintList ids(graph->size(), (unsigned)-1);
		//ids.resize(graph->size());
		//for (unsigned i = 0; i < ids.size(); i ++) ids[i] = (unsigned)-1;
		unsigned level = max_size-1;
		if(debug) printf("\n\n=======================\ninit: u = %d, level = %d\n", u, level);
		for (unsigned i = 0; i < emb_list.size(level); i ++) emb_list.set_label(i, 0);
		unsigned new_size = 0;
		for (auto e : graph->edges(u)) {
			auto v = graph->getEdgeDst(e);
			ids[v] = new_size;
			old_id[new_size] = v;
			emb_list.set_label(new_size, level);
			emb_list.set_vertex(level, new_size, new_size);
			egonet.set_degree(level, new_size, 0);//new degrees
			if(debug) printf("init: v[%d] = %d\n", new_size, v);
			new_size ++;
		}
		if(debug) printf("init: num_neighbors = %d\n", new_size);
		emb_list.set_size(level, new_size); // number of neighbors of u. Since u is in level k, u's neighbors are in level k-1
		//reodering adjacency list and computing new degrees
		//unsigned i = 0;
		//for (auto e0 : graph->edges(u)) {
		//	auto v = graph->getEdgeDst(e0);
		for (unsigned i = 0; i < emb_list.size(level); i ++) {
			unsigned v = old_id[i];
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
			//i ++;
		}
		for (auto e : graph->edges(u)) {
			auto v = graph->getEdgeDst(e);
			ids[v] = (unsigned)-1;
		}
	}

	// construct the subgraph induced by vertex u's neighbors
	void build_egonet_from_edge(const Edge &edge, Egonet &egonet, EmbeddingList &emb_list, UintList &ids, UintList &old_id) {
		unsigned u = edge.src, v = edge.dst;
		if (old_id.empty()) {
			ids.resize(graph->size());
			old_id.resize(core);
			for (unsigned i = 0; i < graph->size(); i ++) ids[i] = (unsigned)-1;
		}
		unsigned level = max_size-1;
		if(debug) printf("\n\n=======================\ninit: u = %d, v = %d, level = %d\n", u, v, level);
		for (unsigned i = 0; i < emb_list.size(level); i ++) emb_list.set_label(i, 0);
		for (auto e : graph->edges(v)) {
			auto dst = graph->getEdgeDst(e);
			ids[dst] = (unsigned)-2;
		}
		unsigned new_size = 0;
		for (auto e : graph->edges(u)) {
			auto dst = graph->getEdgeDst(e);
			if (ids[dst] == (unsigned)-2) {
				if (max_size == 3) total_num += 1;
				else {
					ids[dst] = new_size;
					old_id[new_size] = dst;
					emb_list.set_label(new_size, max_size-2);
					emb_list.set_vertex(max_size-2, new_size, new_size);
					egonet.set_degree(max_size-2, new_size, 0);//new degrees
				}
				if(debug) printf("init: v[%d] = %d\n", new_size, dst);
				new_size ++;
			}
		}
		if (max_size > 3) {
			emb_list.set_size(max_size-2, new_size); // number of neighbors of u. Since u is in level k, u's neighbors are in level k-1
			if(debug) printf("init: num_neighbors = %d\n", new_size);
			for (unsigned i = 0; i < emb_list.size(max_size-2); i ++) {
				unsigned x = old_id[i];
				// intersection of two neighbor lists
				for (auto e : graph->edges(x)) {
					auto dst = graph->getEdgeDst(e); // dst is the neighbor's neighbor
					unsigned new_id = ids[dst];
					if (new_id < (unsigned)-2) { // if dst is also a neighbor of u
						unsigned degree = egonet.get_degree(max_size-2, i);
						egonet.set_adj(core * i + degree, new_id); // relabel
						egonet.set_degree(max_size-2, i, degree+1);
					}
				}
				if(debug) printf("vertex %d, number of common neighbors: %d\n", v, egonet.get_degree(level, i));
				//i ++;
			}
		}
		for (auto e : graph->edges(v)) {
			auto dst = graph->getEdgeDst(e);
			ids[dst] = (unsigned)-1;
		}
	}

	// each task extends from a vertex, level starts from k-1 and decreases until bottom level
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

	void vertex_process() {
		//galois::do_all(galois::iterate((size_t)0, graph->size()),
		//galois::for_each(galois::iterate(graph->begin(), graph->end()),
			//[&](const size_t &u, auto &ctx) {
		galois::do_all(galois::iterate(graph->begin(), graph->end()),
			[&](const size_t &u) {
				Egonet *egonet = egonets.getLocal();
				EmbeddingList *emb_list = emb_lists.getLocal();
				UintList *id_list = id_lists.getLocal();
				UintList *old_id_list = old_id_lists.getLocal();
				build_egonet_from_vertex(u, *egonet, *emb_list, *id_list, *old_id_list);
				//if (max_size > 3)
					dfs_extend(max_size-1, *egonet, *emb_list);
			},
			galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::no_conflicts(),
			galois::loopname("KclSolver")
		);
	}

	void edge_process() {
		std::cout << "num_edges in edge_list = " << edge_list.size() << "\n\n";
		//galois::do_all(galois::iterate((size_t)0, graph->size()),
		//galois::for_each(galois::iterate(edge_list.begin(), edge_list.end()),
			//[&](const Edge &edge, auto &ctx) {
		galois::do_all(galois::iterate(edge_list.begin(), edge_list.end()),
			[&](const Edge &edge) {
				Egonet *egonet = egonets.getLocal();
				EmbeddingList *emb_list = emb_lists.getLocal();
				UintList *id_list = id_lists.getLocal();
				UintList *old_id_list = old_id_lists.getLocal();
				build_egonet_from_edge(edge, *egonet, *emb_list, *id_list, *old_id_list);
				if (max_size > 3) dfs_extend(max_size-2, *egonet, *emb_list);
			},
			galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::no_conflicts(),
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
	bool is_dag;
	UlongAccu total_num;
	EmbeddingLists emb_lists;
	Egonets egonets;
	Lists id_lists;
	Lists old_id_lists;
	EdgeList edge_list;
};

#endif
