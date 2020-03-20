#pragma once
#include "gtypes.h"
#include "embedding_list_dfs.h"

// Sandslash APIs
template <typename EmbeddingTy, bool use_formula=false>
class VertexMinerAPI {
public:
	VertexMinerAPI() {}
	~VertexMinerAPI() {}
	static inline bool toExtend(unsigned level, unsigned pos, const EmbeddingTy *emb) {
		return true;
	}
	static inline bool toAdd(unsigned level, unsigned max_level, VertexId vid, 
		unsigned src_idx, BYTE ccode, const EmbeddingTy *emb) {
		return !is_automorphism(level, vid, src_idx, ccode, emb);
	}
	static inline unsigned getPattern(unsigned level, VertexId dst, 
		unsigned previous_pid, BYTE src_idx, const EmbeddingTy *emb) {
		return 0;
	}
	static inline void reduction(UlongAccu &acc) { acc += 1; }
	static inline void reduction(unsigned level, VertexId src,
		VertexId dst, BYTE ccode, unsigned pcode) { }
	//static inline void reduction(unsigned level, VertexId src,
	//	VertexId dst, BYTE ccode, unsigned pcode, EmbeddingTy &emb) { }
	static inline void update(unsigned level, unsigned vid, 
		unsigned pos, BYTE ccode, unsigned previous_pid) { }
	//static inline void update(unsigned level, unsigned vid, 
	//	unsigned pos, BYTE ccode, unsigned previous_pid, EmbeddingTy &emb) { }
	static inline void post_processing(unsigned level) { }

protected:
/*
	static inline bool is_vertex_automorphism(unsigned n, Graph &g, const EmbeddingTy& emb, unsigned idx, VertexId dst) {
		//unsigned n = emb.size();
		// the new vertex id should be larger than the first vertex id
		if (dst <= emb.get_vertex(0)) return true;
		// the new vertex should not already exist in the embedding
		for (unsigned i = 1; i < n; ++i)
			if (dst == emb.get_vertex(i)) return true;
		// the new vertex should not already be extended by any previous vertex in the embedding
		for (unsigned i = 0; i < idx; ++i)
			if (is_connected(g, emb.get_vertex(i), dst)) return true;
		// the new vertex id must be larger than any vertex id after its source vertex in the embedding
		for (unsigned i = idx+1; i < n; ++i)
			if (dst < emb.get_vertex(i)) return true;
		return false;
	}
	static inline bool is_all_connected_dag(Graph &g, unsigned dst, const EmbeddingTy &emb, unsigned end, unsigned start = 0) {
		assert(start >= 0 && end > 0);
		bool all_connected = true;
		for(unsigned i = start; i < end; ++i) {
			unsigned from = emb.get_vertex(i);
			if (!is_connected_dag(g, dst, from)) {
				all_connected = false;
				break;
			}
		}
		return all_connected;
	}
	static inline bool is_connected(Graph &g, unsigned a, unsigned b) {
		if (g.get_degree(a) == 0 || g.get_degree(b) == 0) return false;
		unsigned key = a;
		unsigned search = b;
		if (g.get_degree(a) < g.get_degree(b)) {
			key = b;
			search = a;
		} 
		auto begin = g.edge_begin(search);
		auto end = g.edge_end(search);
		return binary_search(g, key, begin, end);
	}
	static inline int is_connected_dag(Graph &g, unsigned key, unsigned search) {
		if (g.get_degree(search) == 0) return false;
		auto begin = g.edge_begin(search);
		auto end = g.edge_end(search);
		return binary_search(g, key, begin, end);
	}
	static inline bool binary_search(Graph &g, unsigned key, Graph::edge_iterator begin, Graph::edge_iterator end) {
		auto l = begin;
		auto r = end-1;
		while (r >= l) { 
			auto mid = l + (r - l) / 2; 
			unsigned value = g.getEdgeDst(mid);
			if (value == key) return true;
			if (value < key) l = mid + 1; 
			else r = mid - 1; 
		} 
		return false;
	}
*/
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
	static inline bool is_connected(unsigned level, VertexId vid, unsigned element_id, BYTE ccode) {
		if (level == 1) return (ccode == element_id + 1) || (ccode == 3);
		else if (level == 2) {
			return (ccode & (1 << element_id));
		} else return false;
	}
};
