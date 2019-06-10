#ifndef TYPE_HPP_
#define TYPE_HPP_
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
#include "defs.hh"
#include "graph.hh"
#include "utils.hh"
#include "bignum.hh"
#include "uintseqhash.hh"

#include "element.h"
#include "galois/Bag.h"
#include "galois/Galois.h"

//#ifdef ENABLE_LABEL
//#define ElementType LabeledElement
//#else
//#ifdef USE_SIMPLE
//#define ElementType SimpleElement
//#else
//#define ElementType StructuralElement
//#endif
//#endif

typedef std::set<int> IntSet;
typedef std::vector<VertexId> VertexList;
typedef std::unordered_set<int> HashIntSet;
typedef std::vector<std::unordered_set<int> > HashIntSets;
typedef std::unordered_map<unsigned, unsigned> UintHashMap;
typedef std::map<unsigned, unsigned> UintMap;

template <typename ElementTy>
class Embedding {
//friend std::ostream & operator<<(std::ostream & strm, const Embedding<ElementTy>& emb);
public:
	Embedding() { }
	~Embedding() {}
	VertexId get_vertex(unsigned i) const { return elements[i].get_vid(); }
	BYTE get_history(unsigned i) const { return elements[i].get_his(); }
	bool empty() const { return elements.empty(); }
	void push_back(ElementTy ele) { elements.push_back(ele); }
	void pop_back() { elements.pop_back(); }
	ElementTy& back() { return elements.back(); }
	const ElementTy& back() const { return elements.back(); }
	size_t size() const { return elements.size(); }
	ElementTy* data() { return elements.data(); }
	const ElementTy* data() const { return elements.data(); }
	ElementTy get_element(unsigned i) const { return elements[i]; }
	std::vector<ElementTy> get_elements() const { return elements; }
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
	~EdgeInducedEmbedding() {}
	void set_qpid(unsigned i) { qp_id = i; }
	unsigned get_qpid() { return qp_id; }
private:
	unsigned qp_id;
};
typedef EdgeInducedEmbedding<ElementType> EdgeEmbedding;
typedef EdgeInducedEmbedding<StructuralElement> ESEmbedding;
typedef EdgeInducedEmbedding<LabeledElement> ELEmbedding;

class BaseEmbedding : public Embedding<SimpleElement> {
friend std::ostream & operator<<(std::ostream & strm, const BaseEmbedding& emb);
public:
	BaseEmbedding() {}
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
	VertexInducedEmbedding() : BaseEmbedding() { 
		qp_id = 0;
		//connected.push_back(true);
	}
	VertexInducedEmbedding(const VertexInducedEmbedding &emb) : BaseEmbedding() {
		elements = emb.get_elements();
		qp_id = emb.get_qpid();
	}
	~VertexInducedEmbedding() {}
	//unsigned get_num_edges() const { return num_edges; }
	//void set_num_edges(unsigned i) { num_edges = i; }
	void set_qpid(unsigned i) { qp_id = i; }
	unsigned get_qpid() const { return qp_id; }
	SimpleElement operator[](size_t i) const { return elements[i]; }
	VertexInducedEmbedding& operator=(const VertexInducedEmbedding& other) {
		if(this == &other) return *this;
		elements = other.get_elements();
		qp_id = other.get_qpid();
		return *this;
	}
	//void resize_connected() { for (size_t i=0; i<size(); i++) connected.push_back(false); }
	//void set_connected(unsigned i, unsigned j) { connected[i*(i-1)/2+j] = true; }
	//void unset_connected(unsigned i) { connected[i] = false; }
	//bool is_connected(unsigned i, unsigned j) { return connected[i*(i-1)/2+j]; }
	//size_t get_connected_size() { return connected.size(); }
private:
	unsigned qp_id;
	//unsigned num_edges;
	//std::vector<bool> connected;
};
typedef VertexInducedEmbedding VertexEmbedding;

std::ostream & operator<<(std::ostream & strm, const BaseEmbedding& emb) {
	if (emb.empty()) {
		strm << "(empty)";
		return strm;
	}
	strm << "(";
	for (auto it = emb.get_elements().begin(); it != emb.get_elements().end() - 1; ++ it)
		strm << (*it) << ", ";
	strm << emb.back();
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
	std::cout << ") --> " << emb.get_qpid();
	return strm;
}

template <typename ElementTy>
std::ostream & operator<<(std::ostream & strm, const EdgeInducedEmbedding<ElementTy>& emb) {
	if (emb.empty()) {
		strm << "(empty)";
		return strm;
	}
	strm << "(";
	for (auto it = emb.get_elements().begin(); it != emb.get_elements().end() - 1; ++ it)
		strm << (*it) << ", ";
	strm << emb.back();
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
};

typedef EmbeddingQueue<EdgeEmbedding> EdgeEmbeddingQueue;
typedef EmbeddingQueue<BaseEmbedding> BaseEmbeddingQueue;
typedef EmbeddingQueue<VertexEmbedding> VertexEmbeddingQueue;

#endif // TYPE_HPP_
