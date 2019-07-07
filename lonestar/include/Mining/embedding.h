#ifndef EMBEDDING_HPP_
#define EMBEDDING_HPP_
#include <map>
#include <set>
#include <queue>
#include <cassert>
#include <vector>
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
typedef galois::gstl::Vector<unsigned> UintList;
typedef galois::gstl::Vector<VertexId> VertexList;
typedef galois::gstl::Vector<UintList> IndexLists;
typedef galois::gstl::Vector<VertexList> VertexLists;
typedef galois::gstl::Set<VertexId> VertexSet;
//typedef std::unordered_set<VertexId> VertexSet;

template <typename ElementTy>
class Embedding {
//friend std::ostream & operator<<(std::ostream & strm, const Embedding<ElementTy>& emb);
using iterator = typename std::vector<ElementTy>::iterator;
public:
	Embedding() {}
	Embedding(size_t n) { elements.resize(n); }
	Embedding(const Embedding &emb) { elements = emb.elements; }
	~Embedding() { elements.clear(); }
	VertexId get_vertex(unsigned i) const { return elements[i].get_vid(); }
	BYTE get_history(unsigned i) const { return elements[i].get_his(); }
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
	void set_element(unsigned i, ElementTy ele) { elements[i] = ele; }
	std::vector<ElementTy> get_elements() const { return elements; }
	void clean() { elements.clear(); }
protected:
	std::vector<ElementTy> elements;
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
	//std::vector<bool> connected;
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
	//for (auto it = emb.get_elements().begin(); it != emb.get_elements().end() - 1; ++ it)
	//	strm << (*it) << ", ";
	//strm << emb.back();
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

// print out the embeddings in the task queue
template <typename EmbeddingTy>
class EmbeddingQueue : public galois::InsertBag<EmbeddingTy> {
public:
	void printout_embeddings(int level, bool verbose = false) {
		int num_embeddings = std::distance(this->begin(), this->end());
		unsigned embedding_size = (level+2)* sizeof(ElementType);
		std::cout << "Number of embeddings in level " << level << ": " << num_embeddings << " (embedding_size = " << embedding_size << " Bytes)" << std::endl;
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
	void init(Graph& graph, unsigned max_size = 2) {
		last_level = 1;
		max_level = max_size;
		unsigned eid = 0;
		vid_lists.resize(max_level);
		idx_lists.resize(max_level);
		for (auto src : graph) {
			for (auto e : graph.edges(src)) {
				GNode dst = graph.getEdgeDst(e);
				if (src < dst) {
					vid_lists[0].push_back(src);
					idx_lists[0].push_back(0);
					vid_lists[1].push_back(dst);
					idx_lists[1].push_back(eid++);
				}
			}
		}
		if (show) printout_embeddings(1);
	}
	VertexId get_vid(unsigned level, IndexTy id) const { return vid_lists[level][id]; }
	IndexTy get_idx(unsigned level, IndexTy id) const { return idx_lists[level][id]; }
	IndexTy get_pid(IndexTy id) const { return pid_list[id]; }
	void set_vid(unsigned level, IndexTy id, VertexId vid) { vid_lists[level][id] = vid; }
	void set_idx(unsigned level, IndexTy id, IndexTy idx) { idx_lists[level][id] = idx; }
	void set_pid(IndexTy id, IndexTy pid) { pid_list[id] = pid; }
	size_t size() const { return vid_lists[last_level].size(); }
	size_t size(unsigned level) const { return vid_lists[level].size(); }
	void add_level(unsigned size) {
		last_level ++;
		assert(last_level < max_level);
		vid_lists[last_level].resize(size);
		idx_lists[last_level].resize(size);
		#ifdef USE_PID
		pid_list.resize(size);
		#endif
	}
	void printout_embeddings(int level, bool verbose = false) {
		std::cout << "Number of embeddings in level " << level << ": " << size() << std::endl;
		if(verbose) std::cout << "\n";
	}
private:
	UintList pid_list;
	IndexLists idx_lists;
	VertexLists vid_lists;
	unsigned last_level;
	unsigned max_level;
};

#endif // EMBEDDING_HPP_
