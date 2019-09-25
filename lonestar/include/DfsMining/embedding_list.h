#ifndef EMBEDDING_LIST_H
#define EMBEDDING_LIST_H
#include "embedding.h"

class EmbeddingList {
public:
	EmbeddingList() {}
	EmbeddingList(int core, unsigned k) {
		allocate(core, k);
	}
	~EmbeddingList() {}
	void init(Edge e) {
		for (unsigned i = 2; i < k; i ++) sizes[i] = 0;
		sizes[1] = 1;
		vid_lists[1].resize(1);
		idx_lists[1].resize(1);
		vid_lists[1][0] = e.src;
		idx_lists[1][0] = e.dst;
	}
	void allocate(int core, unsigned k) {
		max_level = k;
		cur_level = k-1;
		last_level = 1;
		sizes.resize(k);
		label.resize(core);
		for (unsigned i = 0; i < k; i ++) sizes[i] = 0;
		vid_lists.resize(k);
		idx_lists.resize(k);
		for (unsigned i = BOTTOM; i < k; i ++) vid_lists[i].resize(core);
		for (unsigned i = BOTTOM; i < k; i ++) idx_lists[i].resize(core);
	}
	template <typename EmbeddingTy>
	inline void get_embedding(unsigned level, unsigned pos, EmbeddingTy &emb) {
		VertexId vid = get_vid(level, pos);
		IndexTy idx = get_idx(level, pos);
		ElementType ele(vid);
		emb.set_element(level, ele);
		// backward constructing the embedding
		for (unsigned l = 1; l < level; l ++) {
			VertexId u = get_vid(level-l, idx);
			ElementType ele(u);
			emb.set_element(level-l, ele);
			idx = get_idx(level-l, idx);
		}
		ElementType ele0(idx);
		emb.set_element(0, ele0);
	}
	size_t size() const { return sizes[cur_level]; }
	size_t size(unsigned level) { return sizes[level]; }
	unsigned get_vertex(unsigned level, unsigned i) { return vid_lists[level][i]; }
	unsigned get_vid(unsigned level, unsigned i) { return vid_lists[level][i]; }
	IndexTy get_idx(unsigned level, IndexTy id) const { return idx_lists[level][id]; }
	BYTE get_label(unsigned vid) { return label[vid]; }
	unsigned get_level() { return cur_level; }
	void set_size(unsigned level, unsigned size) { sizes[level] = size; }
	void set_vertex(unsigned level, unsigned i, unsigned value) { vid_lists[level][i] = value; }
	void set_vid(unsigned level, IndexTy id, VertexId vid) { vid_lists[level][id] = vid; }
	void set_idx(unsigned level, IndexTy id, IndexTy idx) { idx_lists[level][id] = idx; }
	void set_label(unsigned vid, BYTE value) { label[vid] = value; }
	void set_level(unsigned level) { cur_level = level; }

protected:
	VertexLists vid_lists;
	IndexLists idx_lists;
	UintList sizes; //sizes[level]: no. of embeddings (i.e. no. of vertices in the the current level)
	ByteList label; //label[i] is the label of each vertex i that shows its current level
	unsigned max_level;
	unsigned cur_level;
	unsigned last_level;
};
typedef galois::substrate::PerThreadStorage<EmbeddingList> EmbeddingLists;

#endif
