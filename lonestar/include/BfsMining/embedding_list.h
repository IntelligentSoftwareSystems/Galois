#ifndef EMBEDDING_LIST_H_
#define EMBEDDING_LIST_H_
#include "embedding.h"

// Embedding list: SoA structure
class EmbeddingList {
public:
	EmbeddingList() {}
	~EmbeddingList() {}
	void init(Graph& graph, unsigned max_size = 2, bool is_dag = false) {
		last_level = 1;
		max_level = max_size;
		vid_lists.resize(max_level);
		idx_lists.resize(max_level);
		size_t num_emb = graph.sizeEdges();
		if (!is_dag) num_emb = num_emb / 2;
		vid_lists[1].resize(num_emb);
		idx_lists[1].resize(num_emb);
		#ifdef ENABLE_LABEL
		his_lists.resize(max_level);
		his_lists[1].resize(num_emb);
		//for (size_t i = 0; i< num_emb; i ++)
		galois::do_all(galois::iterate((size_t)0, num_emb),
			[&](const size_t& pos) {
				his_lists[1][pos] = 0;
			},
			galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::no_conflicts(),
			galois::loopname("Init-his")
		);
		#endif
		if(is_dag) {
			galois::do_all(galois::iterate(graph.begin(), graph.end()),
				[&](const GNode& src) {
					for (auto e : graph.edges(src)) {
						GNode dst = graph.getEdgeDst(e);
						vid_lists[1][*e] = dst;
						idx_lists[1][*e] = src;
					}
				},
				galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::no_conflicts(),
				galois::loopname("Init-vid")
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
				galois::loopname("Init-vid-alloc")
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
						if (src < dst) { // TODO: this may be incorrect for FSM, which maybe causes the 4-FSM bug
							vid_lists[1][start] = dst;
							idx_lists[1][start] = src;
							start ++;
						}
					}
				},
				galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::no_conflicts(),
				galois::loopname("Init-vid-insert")
			);
		}
	}
	VertexId get_vid(unsigned level, IndexTy id) const { return vid_lists[level][id]; }
	IndexTy get_idx(unsigned level, IndexTy id) const { return idx_lists[level][id]; }
	BYTE get_his(unsigned level, IndexTy id) const { return his_lists[level][id]; }
	IndexTy get_pid(IndexTy id) const { return pid_list[id]; }
	void set_vid(unsigned level, IndexTy id, VertexId vid) { vid_lists[level][id] = vid; }
	void set_idx(unsigned level, IndexTy id, IndexTy idx) { idx_lists[level][id] = idx; }
	void set_his(unsigned level, IndexTy id, BYTE lab) { his_lists[level][id] = lab; }
	void set_pid(IndexTy id, IndexTy pid) { pid_list[id] = pid; }
	size_t size() const { return vid_lists[last_level].size(); }
	size_t size(unsigned level) const { return vid_lists[level].size(); }
	VertexList get_vid_list(unsigned level) { return vid_lists[level]; }
	UintList get_idx_list(unsigned level) { return idx_lists[level]; }
	ByteList get_his_list(unsigned level) { return his_lists[level]; }
	void remove_tail(unsigned idx) {
		vid_lists[last_level].erase(vid_lists[last_level].begin()+idx, vid_lists[last_level].end());
		#ifdef ENABLE_LABEL
		his_lists[last_level].erase(his_lists[last_level].begin()+idx, his_lists[last_level].end());
		#endif
	}
	void add_level(unsigned size) { // TODO: this size could be larger than 2^32, when running LiveJournal and even larger graphs
		last_level ++;
		assert(last_level < max_level);
		vid_lists[last_level].resize(size);
		idx_lists[last_level].resize(size);
		#ifdef ENABLE_LABEL
		his_lists[last_level].resize(size);
		#endif
		#ifdef USE_PID
		pid_list.resize(size);
		#endif
	}
	void printout_embeddings(int level, bool verbose = false) {
		std::cout << "Number of embeddings in level " << level << ": " << size() << std::endl;
		if(verbose) {
			for (size_t pos = 0; pos < size(); pos ++) {
				EmbeddingType emb(last_level+1);
				get_embedding(last_level, pos, emb);
				std::cout << emb << "\n";
			}
		}
	}
private:
	UintList pid_list;
	ByteLists his_lists;
	IndexLists idx_lists;
	VertexLists vid_lists;
	unsigned last_level;
	unsigned max_level;
	void get_embedding(unsigned level, unsigned pos, EmbeddingType &emb) {
		VertexId vid = get_vid(level, pos);
		IndexTy idx = get_idx(level, pos);
		BYTE his = 0;
		#ifdef ENABLE_LABEL
		his = get_his(level, pos);
		#endif
		ElementType ele(vid, 0, 0, his);
		emb.set_element(level, ele);
		for (unsigned l = 1; l < level; l ++) {
			vid = get_vid(level-l, idx);
			#ifdef ENABLE_LABEL
			his = get_his(level-l, idx);
			#endif
			ElementType ele(vid, 0, 0, his);
			emb.set_element(level-l, ele);
			idx = get_idx(level-l, idx);
		}
		ElementType ele0(idx, 0, 0, 0);
		emb.set_element(0, ele0);
	}
};

#ifdef USE_BASE_TYPES
typedef BaseEmbeddingQueue EmbeddingQueueType;
#endif
#ifdef VERTEX_INDUCED
typedef VertexEmbeddingQueue EmbeddingQueueType;
#endif
#ifdef EDGE_INDUCED
typedef EdgeEmbeddingQueue EmbeddingQueueType;
#endif

#endif // EMBEDDING_HPP_
