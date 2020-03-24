#include "../lonestarmine.h"
#include "DfsMining/vertex_miner.h"

const char* name = "Sgl";
const char* desc = "Counts a single arbitrary pattern in a graph using DFS traversal";
const char* url  = 0;

#include "DfsMining/vertex_miner_api.h"
class MyAPI: public VertexMinerAPI<BaseEmbedding> {
public:
	static inline bool toExtend(unsigned level, unsigned pos, BaseEmbedding *emb) {
		return pos == level;
	}
	static inline bool toAdd(unsigned level, unsigned max_level, VertexId vid, 
		unsigned src_idx, BYTE ccode, const std::vector<VertexId> *emb) { 
		return ccode == level; 
	}

//#define DIAMOND
	unsigned getPattern(unsigned level, VertexId vid, EmbeddingList &emb_list, unsigned previous_pid, BYTE src_idx) { 
		if (level < 3) return find_pattern_id_dfs(level, vid, emb_list, previous_pid, src_idx);
		return 0;
	}
	void reduction(unsigned level, EmbeddingList &emb_list, VertexId src_idx, VertexId vid, unsigned previous_pid) {
		unsigned qcode = emb_list.get_label(vid);
		if (previous_pid == 0 && qcode < 5 && qcode != 3) total_num += 1; 
		if (previous_pid == 1) {
			if (src_idx == 0) {
				if (qcode == 3 || qcode == 5) total_num += 1;
			} else {
				if (qcode == 3 || qcode == 6) total_num += 1;;
			}
		}
	}
	/*
	// diamond
	void reduction(unsigned level, EmbeddingList &emb_list, VertexId src_idx, VertexId vid, unsigned previous_pid) {
		if (previous_pid == 0) { // extending a triangle
			if (emb_list.get_label(vid) == 3 || emb_list.get_label(vid) == 5 || emb_list.get_label(vid) == 6) {
				total_num += 1; // diamond
			}
		} else if (emb_list.get_label(vid) == 7) total_num += 1; // diamond
	}
	//*/
	/*
	// 4-cycle
	void reduction(unsigned level, EmbeddingList &emb_list, VertexId src_idx, VertexId vid, unsigned previous_pid) {
		if (previous_pid == 1) { // extending a wedge
			if (src_idx == 0) {
				if (emb_list.get_label(vid) == 6) total_num += 1; // 4-cycle
			} else {
				if (emb_list.get_label(vid) == 5) total_num += 1; // 4-cycle
			}
		}
	}
	//*/
	/* 
	//3-star
	void reduction(unsigned level, EmbeddingList &emb_list, VertexId src_idx, VertexId vid, unsigned previous_pid) {
		if (previous_pid == 1) { // extending a wedge
			if (src_idx == 0) {
				if (emb_list.get_label(vid) == 1) total_num += 1; // 3-star
			} else {
				if (emb_list.get_label(vid) == 2) total_num += 1; // 3-star
			}
		}
	}
				if (emb_list.get_label(vid) == 3 || emb_list.get_label(vid) == 5) pid = 3; // tailed-triangle
				if (emb_list.get_label(vid) == 3 || emb_list.get_label(vid) == 6) pid = 3; // tailed-triangle

	//*/

