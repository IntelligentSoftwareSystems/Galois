//#define USE_DAG
#define USE_DFS
#define USE_MAP
#define USE_PID
//#define USE_FORMULA
#define ALGO_EDGE
#define LARGE_SIZE // for large graphs such as soc-Livejournal1 and com-Orkut
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
	AppMiner(Graph *g) : VertexMiner(g) {}
	~AppMiner() {}
	void init(unsigned max_degree, bool use_dag) {
		assert(k > 2);
		set_max_size(k);
		set_max_degree(max_degree);
		set_num_patterns(num_patterns[k-3]);
		set_directed(use_dag);
		#ifdef ALGO_EDGE
		init_edgelist();
		#endif
		init_emb_list();
	}
	void print_output() { printout_motifs(); }
	// customized pattern classification method
	unsigned getPattern(unsigned level, VertexId vid, EmbeddingList &emb_list, unsigned previous_pid, BYTE src_idx) { 
		if (level < 3) return find_pattern_id_dfs(level, vid, emb_list, previous_pid, src_idx);
		return 0;
	}
	#ifdef USE_FORMULA // only find triangles, 4-cycles and 4-cliques
	bool toExtend(unsigned level, unsigned v_idx) { return v_idx == level; }
	bool toAdd(unsigned level, VertexId vid, const EmbeddingList &emb_list, unsigned src_idx) { 
		return vid != emb_list.get_vid(0, 0);
	}
	void reduction(unsigned level, EmbeddingList &emb_list, VertexId src, VertexId dst, unsigned previous_pid) {
		if (max_size == 3 && emb_list.get_label(dst) == 3) {
			emb_list.inc_tri_count();
		} else if (max_size == 4) {
			if (src < dst && previous_pid == 0 && emb_list.get_label(dst) == 3) { // clique
				accumulators[5] += 1;
			} else if (previous_pid == 1 && emb_list.get_label(dst) == 1) { // 4-cycle
				accumulators[2] += 1;
			}
		}
	}
	void update(unsigned level, unsigned vid, unsigned start, unsigned previous_pid, EmbeddingList &emb_list) {
		unsigned pid = 0;
		if (level == 1) { // triangles and wedges
			if (emb_list.get_label(vid) == 1) {
				emb_list.set_label(vid, 3);
				emb_list.inc_tri_count();
			} else {
				emb_list.set_label(vid, 2);
				emb_list.inc_wed_count();
				pid = 1;
			}
		}
		emb_list.set_pid(level+1, start, pid);
	}
	void post_processing(unsigned level, EmbeddingList &emb_list) {
		if (level == max_size-2) {
			solve_motif_equations(emb_list);
			emb_list.clear_labels(emb_list.get_vid(0, 0));
		} else {
			emb_list.clear_labels(emb_list.get_vid(0, 0));
			emb_list.clear_labels(emb_list.get_vid(1, 0));
		}
	}
	#endif
};

#include "DfsMining/engine.h"

