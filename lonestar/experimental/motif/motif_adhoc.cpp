#define USE_DFS
#define USE_MAP
#define USE_PID
#define ALGO_EDGE
#define USE_ADHOC
#define USE_EGONET
#define USE_SIMPLE
#define VERTEX_INDUCED
#define CHUNK_SIZE 256
#include "pangolin.h"

// This is a implementation of the ICDM'15 paper:
// Nesreen K. Ahmed et al., Efficient Graphlet Counting for Large Networks, ICDM 2015
const char* name = "Motif";
const char* desc = "Counts motifs in a graph using DFS traversal";
const char* url  = 0;
int num_patterns[3] = {2, 6, 21};

class AppMiner : public VertexMiner {
public:
	AppMiner(Graph *g, unsigned size, int np, bool use_dag, unsigned c) : VertexMiner(g, size, np, use_dag, c) {}
	~AppMiner() {}
	void edge_process_adhoc() {
		std::cout << "Starting Ad-Hoc Motif Solver\n";
		if (max_size > 4) {
			std::cout << "not supported yet\n";
			return;
		}
		galois::do_all(galois::iterate((size_t)0, edge_list.size()),
			[&](const size_t& pos) {
				EmbeddingList *emb_list = emb_lists.getLocal();
				emb_list->init(edge_list.get_edge(pos));
				dfs_extend_adhoc(1, 0, *emb_list);
			},
			galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::no_conflicts(),
			galois::loopname("DfsAdhocSolver")
		);
		motif_count();
	}
	// construct the subgraph induced by edge (u, v)'s neighbors
	void dfs_extend_adhoc(unsigned level, unsigned pos, EmbeddingList &emb_list) {
		unsigned n = level + 1;
		VertexEmbedding emb(n);
		emb_list.get_embedding<VertexEmbedding>(level, pos, emb);
		auto u = emb.get_vertex(0), v = emb.get_vertex(1);
		//std::cout << "Edge: " << u << " --> " << v << "\n";

		UintList *ids = id_lists.getLocal();
		if (ids->empty()) {
			ids->resize(graph->size());
			std::fill(ids->begin(), ids->end(), 0);
		}

		UintList *T_vu = Tri_vids.getLocal(); // to record the third vertex in each triangle
		UintList *W_u = Wed_vids.getLocal(); //  to record the third vertex in each wedge
		if (W_u->empty()) {
			T_vu->resize(core+1); // hold the vertices that form a triangle with u and v
			W_u->resize(core+1); // hold the vertices that form a wedge with u and v
			std::fill(T_vu->begin(), T_vu->end(), 0);
			std::fill(W_u->begin(), W_u->end(), 0);
		}

		Ulong wedge_count = 0, tri_count = 0;
		mark_neighbors(v, u, *ids);
		if (max_size == 3) {
			auto begin = graph->edge_begin(u);
			auto end = graph->edge_end(u);
			for (auto e = begin; e < end; e ++) {
				auto w = graph->getEdgeDst(e);
				if (w == v) continue;
				if ((*ids)[w] == 1) tri_count++;
			}
			//solve_3motif_equations(v, u, tri_count, wedge_count);
		} else if (max_size == 4) {
			triangles_and_wedges(v, u, *T_vu, tri_count, *W_u, wedge_count, *ids);
			//solve_4motif_equations(v, u, tri_count, wedge_count);
			Ulong clique4_count = 0, cycle4_count = 0;
			cycle(wedge_count, *W_u, cycle4_count, *ids);
			clique(tri_count, *T_vu, clique4_count, *ids);
			accumulators[5] += clique4_count;
			accumulators[2] += cycle4_count;
		} else {
		}
		solve_motif_equations(v, u, tri_count, wedge_count);
		reset_perfect_hash(v, *ids);
	}
	void print_output() { printout_motifs(); }
};

#include "DfsMining/engine.h"

