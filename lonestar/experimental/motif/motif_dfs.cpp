//#define USE_DAG
#define USE_DFS
#define USE_MAP
#define USE_PID
#define ALGO_EDGE
#define USE_EGONET
#define USE_SIMPLE
#define VERTEX_INDUCED
#define CHUNK_SIZE 256
#include "pangolin.h"

const char* name = "Motif";
const char* desc = "Counts motifs in a graph using DFS traversal";
const char* url  = 0;
int num_patterns[3] = {2, 6, 21};

class AppMiner : public VertexMiner {
public:
	AppMiner(Graph *g, unsigned size, int np, bool use_dag, unsigned c) : VertexMiner(g, size, np, use_dag, c) {}
	~AppMiner() {}
	// customized pattern classification method
	unsigned getPattern(unsigned n, unsigned i, VertexId dst, const VertexEmbedding &emb, unsigned previous_pid) { 
		if (n < 4) return find_motif_pattern_id_dfs(n, i, dst, emb, previous_pid);
		return 0;
	}
	void print_output() { printout_motifs(); }
	#ifdef USE_EGONET
	bool toExtend(unsigned n, const VertexEmbedding &emb, unsigned pos) {
		return pos == n-1;
	}
	bool toAdd(unsigned n, const VertexEmbedding &emb, VertexId dst, unsigned pos) {
		return dst != emb.get_vertex(0);
	}
	void reduction(unsigned level, const VertexEmbedding &emb, const EmbeddingList &emb_list, unsigned src, unsigned dst, unsigned emb_id) {
		UintList *ids = id_lists.getLocal();
		unsigned *trian_count = Tri_counts.getLocal();
		if (dst != emb.get_vertex(0)) {
			if (max_size == 3 && (*ids)[dst] == 1) {
				(*trian_count) += 1;
			}
			else if (max_size == 4) {
				auto previous_pid = emb_list.get_pid(level, emb_id);
				if (dst > src && previous_pid == 0 && (*ids)[dst] == 3) { // clique
					accumulators[5] += 1;
				} else if (previous_pid == 1 && (*ids)[dst] == 1) { // 4-cycle
					accumulators[2] += 1;
				}
			}
		}
	}
	void update(unsigned level, unsigned dst, unsigned start, EmbeddingList &emb_list) {
		UintList *ids = id_lists.getLocal();
		unsigned *trian_count = Tri_counts.getLocal();
		unsigned *wedge_count = Wed_counts.getLocal();
		unsigned pid = 0;
		if ((*ids)[dst] == 1) {
			(*ids)[dst] = 3;
			(*trian_count) += 1;
		} else {
			(*ids)[dst] = 2;
			(*wedge_count) += 1;
			pid = 1;
		}
		emb_list.set_pid(level+1, start, pid);
	}
	void init_egonet_from_edge(const Edge &edge, Egonet &egonet, EmbeddingList &emb_list) {
		emb_list.init(edge);
		UintList *ids = id_lists.getLocal();
		if (ids->empty()) {
			ids->resize(graph->size());
			std::fill(ids->begin(), ids->end(), 0);
		}
		mark_neighbors(edge.src, edge.dst, *ids);
		unsigned *trian_count = Tri_counts.getLocal();
		unsigned *wedge_count = Wed_counts.getLocal();
		*wedge_count = 0, *trian_count = 0;
		unsigned *v0 = src_ids.getLocal();
		unsigned *v1 = dst_ids.getLocal();
		*v0 = edge.src;
		*v1 = edge.dst;
	}
	void post_processing(unsigned level) {
		unsigned *v0 = src_ids.getLocal();
		unsigned *v1 = dst_ids.getLocal();
		UintList *ids = id_lists.getLocal();
		unsigned *trian_count = Tri_counts.getLocal();
		unsigned *wedge_count = Wed_counts.getLocal();
		if (level == max_size-2) {
			solve_motif_equations(*v0, *v1, *trian_count, *wedge_count);
			reset_perfect_hash(*v0, *ids);
		} else {
			reset_perfect_hash(*v0, *ids);
			reset_perfect_hash(*v1, *ids);
		}
	}
	#endif
};

#include "DfsMining/engine.h"

