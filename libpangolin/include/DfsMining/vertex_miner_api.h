#pragma once
#include "gtypes.h"
#include "embedding_list_dfs.h"

// Sandslash APIs
//template <bool use_formula=false>
class VertexMinerAPI {
public:
	VertexMinerAPI() {}
	~VertexMinerAPI() {}
	// pos: the position of the vertex (to be extended) in the embedding
	static inline bool toExtend(unsigned level, unsigned pos, const std::vector<VertexId> *emb) {
		return true;
	}

	// vid: the candidate vertex to be added to the embedding
	// src_idx: the position of the source vertex (that vid is extended from) in the embedding
	// ccode: connectivity code of vid (shows connectivity between vid and each vertex in the embedding)
	static inline bool toAdd(unsigned level, unsigned max_level, VertexId vid, 
		unsigned src_idx, BYTE ccode, const std::vector<VertexId> *emb) {
		return !is_automorphism(level, vid, src_idx, ccode, *emb);
	}

	// src: the source vertex (that dst is extended from) in the embedding
	// dst: the candidate vertex to be added to the embedding
	static inline unsigned getPattern(unsigned level, unsigned max_level, VertexId src, 
		VertexId dst, BYTE ccode, unsigned pcode, BYTE src_idx, const std::vector<VertexId> *emb) {
		return 0;
	}

	// defines the reduction operator: do couting by default
	static inline void reduction(UlongAccu &acc) { acc += 1; }
	
	static inline void local_reduction(unsigned level, int pid, Ulong &counter) { counter ++; } 

	static inline void update(unsigned level, unsigned vid, 
		unsigned pos, BYTE ccode, unsigned previous_pid) { }
	//static inline void post_processing(unsigned level) { }

protected:
/*
	static inline bool is_automorphism(unsigned level, 
		VertexId dst, unsigned idx, BYTE ccode, const EmbeddingTy *emb) {
		// the new vertex id should be larger than the first vertex id
		if (dst <= emb->get_vertex(0)) return true;
		// the new vertex should not already exist in the embedding
		for (unsigned i = 1; i < level+1; ++i)
			if (dst == emb->get_vertex(i)) return true;
		// the new vertex should not already be extended by any previous vertex in the embedding
		for (unsigned i = 0; i < idx; ++i)
			if (is_connected(level, dst, i, ccode)) return true;
		// the new vertex id should be larger than any vertex id after its source vertex in the embedding
		for (unsigned i = idx+1; i < level+1; ++i)
			if (dst < emb->get_vertex(i)) return true;
		return false;
	}
//*/
	static inline bool is_automorphism(unsigned level, 
		VertexId dst, unsigned idx, BYTE ccode, const std::vector<VertexId> &emb) {
		// the new vertex id should be larger than the first vertex id
		if (dst <= emb[0]) return true;
		// the new vertex should not already exist in the embedding
		for (unsigned i = 1; i < level+1; ++i)
			if (dst == emb[i]) return true;
		// the new vertex should not already be extended by any previous vertex in the embedding
		for (unsigned i = 0; i < idx; ++i)
			if (is_connected(level, dst, i, ccode)) return true;
		// the new vertex id should be larger than any vertex id after its source vertex in the embedding
		for (unsigned i = idx+1; i < level+1; ++i)
			if (dst < emb[i]) return true;
		return false;
	}

	static inline bool is_connected(unsigned level, VertexId vid, unsigned element_id, BYTE ccode) {
		if (level == 1) return (ccode == element_id + 1) || (ccode == 3);
		else if (level == 2) {
			return (ccode & (1 << element_id));
		} else return false;
	}
	static inline unsigned get_pattern_id(unsigned level, VertexId vid, BYTE ccode, unsigned previous_pid = 0, unsigned src_idx = 0) {
		unsigned pid = 0;
		if (level == 1) { // count 3-motifs
			if (ccode == 3) {
				pid = 0; // triangle
			} else {
				pid = 1; //wedge 
			}
		} else if (level == 2) { // count 4-motifs
			if (previous_pid == 0) { // extending a triangle
				if (ccode == 7) {
					pid = 5; // clique
				} else if (ccode == 3 || ccode == 5 || ccode == 6) {
					pid = 4; // diamond
				} else pid = 3; // tailed-triangle
			} else {
				if (ccode == 7) {
					pid = 4; // diamond
				} else if (src_idx == 0) {
					if (ccode == 6) pid = 2; // 4-cycle
					else if (ccode == 3 || ccode == 5) pid = 3; // tailed-triangle
					else if (ccode == 1) pid = 1; // 3-star
					else pid = 0 ; // 4-chain
				} else {
					if (ccode == 5) pid = 2; // 4-cycle
					else if (ccode == 3 || ccode == 6) pid = 3; // tailed-triangle
					else if (ccode == 2) pid = 1; // 3-star
					else pid = 0; // 4-chain
				}
			}
		}
		return pid;
	}

};
