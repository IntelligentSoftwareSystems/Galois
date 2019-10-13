#define USE_DAG
#define USE_DFS
#define ALGO_EDGE
#define USE_ADHOC
#define USE_EGONET
#define USE_SIMPLE
#define USE_BASE_TYPES
#define CHUNK_SIZE 256
#include "pangolin.h"

// This is a implementation of the WWW'18 paper:
// Danisch et al., Listing k-cliques in Sparse Real-World Graphs, WWW 2018
const char* name = "Kcl";
const char* desc = "Counts the K-Cliques in a graph using DFS traversal (inputs do NOT need to be symmetrized)";
const char* url  = 0;

class AppMiner : public VertexMiner {
public:
	AppMiner(Graph *g, unsigned size, int np, bool use_dag, unsigned c) : VertexMiner(g, size, np, use_dag, c) {}
	~AppMiner() {}
	void print_output() {
		std::cout << "\n\ttotal_num_cliques = " << get_total_count() << "\n";
	}
	// each task extends from a vertex, level starts from k-1 and decreases until bottom level
	void dfs_extend(unsigned level, Egonet &egonet, EmbeddingList &emb_list) {
		if (level == max_size-2) {
			for(size_t emb_id = 0; emb_id < emb_list.size(level); emb_id ++) { //list all edges
				auto vid = emb_list.get_vertex(level, emb_id);
				auto begin = egonet.edge_begin(vid);
				auto end = begin + egonet.get_degree(level, vid);
				for (unsigned e = begin; e < end; e ++)
					total_num += 1;
			}
			return;
		}
		// compute the subgraphs induced on the neighbors of each node in current level,
		// and then recurse on such a subgraph
		for(size_t emb_id = 0; emb_id < emb_list.size(level); emb_id ++) {
			// for each vertex in current level
			// a new induced subgraph G[∆G(u)] is built
			auto vid = emb_list.get_vertex(level, emb_id);
			auto begin = egonet.edge_begin(vid);
			auto end = begin + egonet.get_degree(level, vid);
			emb_list.set_size(level+1, 0);
			// extend one vertex v which is a neighbor of u
			for (unsigned edge = begin; edge < end; edge ++) {
				// for each out-neighbor v of node u in G, set its label to level-1
				// if the label was equal to level. We thus have that if a label of a
				// node v is equal to level-1 it means that node v is in the new subgraph
				auto dst = egonet.getEdgeDst(edge);
				// relabeling vertices and forming U'.
				if (emb_list.get_label(dst) == level) {
					auto pos = emb_list.size(level+1);
					emb_list.set_vertex(level+1, pos, dst);
					emb_list.set_size(level+1, pos+1);
					emb_list.set_label(dst, level+1);
					egonet.set_degree(level+1, dst, 0);//new degrees
				}
			}
			// update egonet (perform intersection)
			for (size_t new_emb_id = 0; new_emb_id < emb_list.size(level+1); new_emb_id ++) {
				auto src = emb_list.get_vertex(level+1, new_emb_id);
				begin = egonet.edge_begin(src);
				end = begin + egonet.get_degree(level, src);
				for (unsigned e = begin; e < end; e ++) {
					auto dst = egonet.getEdgeDst(e);
					if (emb_list.get_label(dst) == level+1)
						egonet.inc_degree(level+1, src);
					else {
						egonet.set_adj(e--, egonet.getEdgeDst(--end));
						egonet.set_adj(end, dst);
					}
				}
			}
			dfs_extend(level+1, egonet, emb_list);
			for (unsigned emb_id = 0; emb_id < emb_list.size(level+1); emb_id ++) {//restoring labels
				auto src = emb_list.get_vertex(level+1, emb_id);
				emb_list.set_label(src, level);
			}
		}
	}

	void vertex_process_adhoc() {
		std::cout << "Starting Ad-Hoc k-clique Solver (vertex-parallel)\n";
		//galois::do_all(galois::iterate((size_t)0, graph->size()),
		//galois::do_all(galois::iterate(graph->begin(), graph->end()),
		//	[&](const size_t &u) {
		galois::for_each(galois::iterate(graph->begin(), graph->end()),
			[&](const auto &vid, auto &ctx) {
				Egonet *egonet = egonets.getLocal();
				EmbeddingList *emb_list = emb_lists.getLocal();
				init_egonet_from_vertex(vid, *egonet, *emb_list);
				if (max_size > 3) dfs_extend(1, *egonet, *emb_list);
			},
			galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::no_conflicts(),
			galois::loopname("KclDfsVertexAdHocSolver")
		);
	}
	void edge_process_adhoc() {
		std::cout << "Starting Ad-Hoc k-clique Solver (edge-parallel)\n";
		//galois::do_all(galois::iterate((size_t)0, graph->size()),
		//galois::for_each(galois::iterate(edge_list.begin(), edge_list.end()),
			//[&](const Edge &edge, auto &ctx) {
		galois::do_all(galois::iterate(edge_list.begin(), edge_list.end()),
			[&](const Edge &edge) {
				EmbeddingList *emb_list = emb_lists.getLocal();
				Egonet *egonet = egonets.getLocal();
				init_egonet_from_edge(edge, *egonet, *emb_list);
				if (max_size > 3) dfs_extend(2, *egonet, *emb_list);
			},
			galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::no_conflicts(),
			galois::loopname("KclDfsEdgeAdHocSolver")
		);
	}

	void init_egonet_from_vertex(const VertexId vid, Egonet &egonet, EmbeddingList &emb_list) {
		UintList *ids = id_lists.getLocal();
		UintList *old_ids = old_id_lists.getLocal();
		if (ids->empty()) {
			ids->resize(graph->size());
			old_ids->resize(core);
			std::fill(ids->begin(), ids->end(), (unsigned)-1);
		}
		unsigned level = 1;
		emb_list.init_vertex(vid);
		for (size_t i = 0; i < emb_list.size(level); i ++) emb_list.set_label(i, 0);
		unsigned new_id = 0;
		for (auto e : graph->edges(vid)) {
			auto dst = graph->getEdgeDst(e);
			(*ids)[dst] = new_id;
			(*old_ids)[new_id] = dst;
			emb_list.set_label(new_id, level);
			emb_list.set_vertex(level, new_id, new_id);
			egonet.set_degree(level, new_id, 0);//new degrees
			new_id ++;
		}
		size_t new_size = new_id;
		emb_list.set_size(level, new_size);
		//unsigned i = 0;
		//for (auto e0 : graph->edges(vid)) {
		//	auto src = graph->getEdgeDst(e0);
		for (unsigned i = 0; i < emb_list.size(level); i ++) {
			unsigned src = (*old_ids)[i];
			// intersection of two neighbor lists
			for (auto e : graph->edges(src)) {
				auto dst = graph->getEdgeDst(e); // dst is the neighbor's neighbor
				unsigned new_id = (*ids)[dst];
				if (new_id != (unsigned)-1) { // if dst is also a neighbor of u
					if (max_size == 3) total_num += 1; //listing 3-clique here!!!
					else {
						auto degree = egonet.get_degree(level, i);
						egonet.set_adj(core * i + degree, new_id); // relabel
						egonet.set_degree(level, i, degree+1);
					}
				}
			}
			//i ++;
		}
		for (auto e : graph->edges(vid)) {
			auto dst = graph->getEdgeDst(e);
			(*ids)[dst] = (unsigned)-1;
		}
	}
	// construct the subgraph induced by vertex u's neighbors
	void init_egonet_from_edge(const Edge &edge, Egonet &egonet, EmbeddingList &emb_list) {
		UintList *ids = id_lists.getLocal();
		UintList *old_ids = old_id_lists.getLocal();
		if (ids->empty()) {
			ids->resize(graph->size());
			old_ids->resize(core);
			std::fill(ids->begin(), ids->end(), (unsigned)-1);
		}
		for (auto e : graph->edges(edge.dst)) {
			auto dst = graph->getEdgeDst(e);
			(*ids)[dst] = (unsigned)-2;
		}
		for (size_t i = 0; i < emb_list.size(1); i ++) emb_list.set_label(i, 0);
		unsigned new_id = 0;
		for (auto e : graph->edges(edge.src)) {
			auto dst = graph->getEdgeDst(e);
			if ((*ids)[dst] == (unsigned)-2) {
				if (max_size == 3) total_num += 1;
				else {
					(*ids)[dst] = new_id;
					(*old_ids)[new_id] = dst;
					emb_list.set_label(new_id, 2);
					emb_list.set_vertex(2, new_id, new_id);
					egonet.set_degree(2, new_id, 0);//new degrees
				}
				new_id ++;
			}
		}
		size_t new_size = new_id;
		if (max_size > 3) {
			emb_list.set_size(2, new_size); // number of neighbors of u. Since u is in level k, u's neighbors are in level k-1
			for (unsigned i = 0; i < emb_list.size(2); i ++) {
				unsigned x = (*old_ids)[i];
				for (auto e : graph->edges(x)) {
					auto dst = graph->getEdgeDst(e); // dst is the neighbor's neighbor
					unsigned new_id = (*ids)[dst];
					if (new_id < (unsigned)-2) { // if dst is also a neighbor of u
						auto degree = egonet.get_degree(2, i);
						egonet.set_adj(core * i + degree, new_id); // relabel
						egonet.set_degree(2, i, degree+1);
					}
				}
			}
		}
		for (auto e : graph->edges(edge.dst)) {
			auto dst = graph->getEdgeDst(e);
			(*ids)[dst] = (unsigned)-1;
		}
	}
};

#include "DfsMining/engine.h"
