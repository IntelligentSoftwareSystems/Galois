#define USE_PID
#include "../lonestarmine.h"
#include "DfsMining/vertex_miner.h"

const char* name = "Motif";
const char* desc = "Counts motifs in a graph using DFS traversal";
const char* url  = 0;
int num_patterns[3] = {2, 6, 21};

#include "DfsMining/vertex_miner_api.h"
class MyAPI: public VertexMinerAPI<BaseEmbedding> {
typedef EmbeddingList<SimpleElement, BaseEmbedding> MyEmbeddingList;
public:
	// customized pattern classification method
	static inline unsigned getPattern(unsigned level, VertexId vid, 
		unsigned previous_pid, BYTE src_idx, BaseEmbedding *emb) { 
		//if (level < 3) return find_pattern_id_dfs(level, vid, emb_list, previous_pid, src_idx);
		return 0;
	}

	#ifdef USE_FORMULA // only find triangles, 4-cycles and 4-cliques
	bool toExtend(unsigned level, unsigned v_idx, BaseEmbedding *emb) { 
		return v_idx == level; 
	}
	bool toAdd(unsigned level, VertexId vid, unsigned src_idx, 
		BYTE ccode, BaseEmbedding *emb) { 
		return vid != emb_list.get_vid(0, 0);
	}
	void reduction(unsigned level, VertexId src, VertexId dst, 
		BYTE ccode, unsigned pcode) {
		if (max_size == 3 && ccode == 3) {
			//emb_list.inc_tri_count(); // TODO
		} else if (max_size == 4) {
			if (src < dst && pcode == 0 && ccode == 3) { // clique
				accumulators[5] += 1;
			} else if (pcode == 1 && ccode == 1) { // 4-cycle
				accumulators[2] += 1;
			}
		}
	}
	void update(unsigned level, unsigned vid, unsigned start, 
		unsigned previous_pid, MyEmbeddingList &emb_list) {
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
	void post_processing(unsigned level, MyEmbeddingList &emb_list) {
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

class AppMiner : public VertexMinerDFS<SimpleElement, BaseEmbedding, MyAPI, false, false> {
public:
	AppMiner(unsigned ms, int nt) : 
		VertexMinerDFS<SimpleElement, BaseEmbedding, MyAPI, false, false>(ms, nt) {
		assert(k > 2);
		set_num_patterns(num_patterns[k-3]);
	}
	~AppMiner() {}
	void print_output() { printout_motifs(); }
};

#include "DfsMining/engine.h"

