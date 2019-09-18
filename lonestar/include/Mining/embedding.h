#ifndef EMBEDDING_HPP_
#define EMBEDDING_HPP_
#include <map>
#include <set>
#include <queue>
#include <vector>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <cstring>
#include <iostream>
#include <string.h>
#include <algorithm>
#include <functional>
#include <unordered_map>
#include <unordered_set>

// bliss headers
#include "bliss/defs.hh"
#include "bliss/graph.hh"
#include "bliss/utils.hh"
#include "bliss/bignum.hh"
#include "bliss/uintseqhash.hh"

#include "element.h"
#include "galois/Bag.h"
#include "galois/Galois.h"

typedef unsigned IndexTy;
typedef unsigned long Ulong;
typedef galois::gstl::Vector<BYTE> ByteList;
typedef galois::gstl::Vector<unsigned> UintList;
typedef galois::gstl::Vector<Ulong> UlongList;
typedef galois::gstl::Vector<VertexId> VertexList;
typedef galois::gstl::Vector<UintList> IndexLists;
typedef galois::gstl::Vector<ByteList> ByteLists;
typedef galois::gstl::Vector<VertexList> VertexLists;
typedef galois::gstl::Set<VertexId> VertexSet;

template <typename ElementTy>
class Embedding {
//using iterator = typename std::vector<ElementTy>::iterator;
using iterator = typename galois::gstl::Vector<ElementTy>::iterator;
public:
	Embedding() {}
	Embedding(size_t n) { elements.resize(n); }
	Embedding(const Embedding &emb) { elements = emb.elements; }
	~Embedding() { elements.clear(); }
	VertexId get_vertex(unsigned i) const { return elements[i].get_vid(); }
	BYTE get_history(unsigned i) const { return elements[i].get_his(); }
	BYTE get_label(unsigned i) const { return elements[i].get_vlabel(); }
	BYTE get_key(unsigned i) const { return elements[i].get_key(); }
	bool empty() const { return elements.empty(); }
	iterator begin() { return elements.begin(); }
	iterator end() { return elements.end(); }
	iterator insert(iterator pos, const ElementTy& value ) { return elements.insert(pos, value); }
	void push_back(ElementTy ele) { elements.push_back(ele); }
	void pop_back() { elements.pop_back(); }
	ElementTy& back() { return elements.back(); }
	const ElementTy& back() const { return elements.back(); }
	size_t size() const { return elements.size(); }
	void resize (size_t n) { elements.resize(n); }
	ElementTy* data() { return elements.data(); }
	const ElementTy* data() const { return elements.data(); }
	ElementTy get_element(unsigned i) const { return elements[i]; }
	void set_element(unsigned i, ElementTy &ele) { elements[i] = ele; }
	void set_vertex(unsigned i, VertexId vid) { elements[i].set_vertex_id(vid); }
	//std::vector<ElementTy> get_elements() const { return elements; }
	galois::gstl::Vector<ElementTy> get_elements() const { return elements; }
	void clean() { elements.clear(); }
protected:
	//std::vector<ElementTy> elements;
	galois::gstl::Vector<ElementTy> elements;
};

template <typename ElementTy> class EdgeInducedEmbedding;
template <typename ElementTy> std::ostream& operator<<(std::ostream& strm, const EdgeInducedEmbedding<ElementTy>& emb);

template <typename ElementTy>
class EdgeInducedEmbedding : public Embedding<ElementTy> {
friend std::ostream & operator<< <>(std::ostream & strm, const EdgeInducedEmbedding<ElementTy>& emb);
public:
	EdgeInducedEmbedding() { qp_id = 0xFFFFFFFF; }
	EdgeInducedEmbedding(size_t n) : Embedding<ElementTy>(n) {}
	~EdgeInducedEmbedding() {}
	void set_qpid(unsigned i) { qp_id = i; } // set the quick pattern id
	unsigned get_qpid() const { return qp_id; } // get the quick pattern id
private:
	unsigned qp_id; // quick pattern id
};
typedef EdgeInducedEmbedding<ElementType> EdgeEmbedding;

class BaseEmbedding : public Embedding<SimpleElement> {
friend std::ostream & operator<<(std::ostream & strm, const BaseEmbedding& emb);
public:
	BaseEmbedding() {}
	BaseEmbedding(size_t n) : Embedding(n) {}
	~BaseEmbedding() {}
	inline unsigned get_hash() const {
		bliss::UintSeqHash h;
		for(unsigned i = 0; i < size(); ++i)
			h.update(elements[i].get_vid());
		return h.get_value();
	}
	BaseEmbedding& operator=(const BaseEmbedding& other) {
		if(this == &other) return *this;
		elements = other.get_elements();
		return *this;
	}
	friend bool operator==(const BaseEmbedding &e1, const BaseEmbedding &e2) {
		return e1.elements == e2.elements;
	}
};

class VertexInducedEmbedding: public BaseEmbedding {
friend std::ostream & operator<<(std::ostream & strm, const VertexInducedEmbedding& emb);
public:
	VertexInducedEmbedding() : BaseEmbedding() { hash_value = 0; }
	VertexInducedEmbedding(size_t n) : BaseEmbedding(n) { hash_value = 0; }
	VertexInducedEmbedding(const VertexInducedEmbedding &emb) : BaseEmbedding() {
		elements = emb.get_elements();
		hash_value = emb.get_pid();
	}
	~VertexInducedEmbedding() {}
	SimpleElement operator[](size_t i) const { return elements[i]; }
	VertexInducedEmbedding& operator=(const VertexInducedEmbedding& other) {
		if(this == &other) return *this;
		elements = other.get_elements();
		hash_value = other.get_pid();
		return *this;
	}
	inline unsigned get_pid() const { return hash_value; } // get the pattern id
	inline void set_pid(unsigned i) { hash_value = i; } // set the pattern id
protected:
	unsigned hash_value;
};
typedef VertexInducedEmbedding VertexEmbedding;

std::ostream & operator<<(std::ostream & strm, const BaseEmbedding& emb) {
	if (emb.empty()) {
		strm << "(empty)";
		return strm;
	}
	strm << "(";
	for(unsigned index = 0; index < emb.size() - 1; ++index)
		std::cout << emb.get_vertex(index) << ", ";
	std::cout << emb.get_vertex(emb.size()-1);
	strm << ")";
	return strm;
}

std::ostream & operator<<(std::ostream & strm, const VertexEmbedding& emb) {
	if (emb.empty()) {
		strm << "(empty)";
		return strm;
	}
	std::cout << "(";
	for(unsigned index = 0; index < emb.size() - 1; ++index)
		std::cout << emb.get_vertex(index) << ", ";
	std::cout << emb.get_vertex(emb.size()-1);
	std::cout << ") --> " << emb.get_pid();
	return strm;
}

template <typename ElementTy>
std::ostream & operator<<(std::ostream & strm, const EdgeInducedEmbedding<ElementTy>& emb) {
	if (emb.empty()) {
		strm << "(empty)";
		return strm;
	}
	strm << "(";
	for(unsigned index = 0; index < emb.size() - 1; ++index)
		std::cout << emb.get_element(index) << ", ";
	std::cout << emb.get_element(emb.size()-1);
	strm << ")";
	return strm;
}

namespace std {
	template<>
	struct hash<BaseEmbedding> {
		std::size_t operator()(const BaseEmbedding& emb) const {
			return std::hash<int>()(emb.get_hash());
		}
	};
}

namespace std {
	template<>
	struct hash<VertexEmbedding> {
		std::size_t operator()(const VertexEmbedding& emb) const {
			return std::hash<int>()(emb.get_hash());
		}
	};
}

#ifdef USE_BASE_TYPES
typedef BaseEmbedding EmbeddingType;
#endif
#ifdef VERTEX_INDUCED
typedef VertexEmbedding EmbeddingType;
#endif
#ifdef EDGE_INDUCED
typedef EdgeEmbedding EmbeddingType;
#endif

// print out the embeddings in the task queue
template <typename EmbeddingTy>
class EmbeddingQueue : public galois::InsertBag<EmbeddingTy> {
public:
	void printout_embeddings(int level, bool verbose = false) {
		int num_embeddings = std::distance(this->begin(), this->end());
		//unsigned embedding_size = (level+1)* sizeof(ElementType);
		std::cout << "Number of embeddings in level " << level << ": " << num_embeddings << std::endl;
		if(verbose) for (auto emb : *this) std::cout << emb << "\n";
	}
	void clean() { for (auto emb : *this) emb.clean(); this->clear(); }
};

typedef EmbeddingQueue<EdgeEmbedding> EdgeEmbeddingQueue;
typedef EmbeddingQueue<BaseEmbedding> BaseEmbeddingQueue;
typedef EmbeddingQueue<VertexEmbedding> VertexEmbeddingQueue;

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
