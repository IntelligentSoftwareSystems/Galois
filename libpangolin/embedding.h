#ifndef EMBEDDING_HPP_
#define EMBEDDING_HPP_
/**
 * Code from on below link. Modified under Galois.
 *
 * https://github.com/rstream-system/RStream/
 *
 * Copyright (c) 2018, Kai Wang and the respective contributors
 * All rights reserved.
 * Reused/revised under 3-BSD
 */

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

// Basic Vertex-induced embedding
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

namespace std {
	template<>
	struct hash<BaseEmbedding> {
		std::size_t operator()(const BaseEmbedding& emb) const {
			return std::hash<int>()(emb.get_hash());
		}
	};
}

// Vertex-induced embedding with hash value
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

namespace std {
	template<>
	struct hash<VertexEmbedding> {
		std::size_t operator()(const VertexEmbedding& emb) const {
			return std::hash<int>()(emb.get_hash());
		}
	};
}

// Edge induced embedding
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

#ifdef USE_BASE_TYPES
typedef BaseEmbedding EmbeddingType;
#endif
#ifdef VERTEX_INDUCED
typedef VertexEmbedding EmbeddingType;
#endif
#ifdef EDGE_INDUCED
typedef EdgeEmbedding EmbeddingType;
#endif

// Embedding queue: AoS structure
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
