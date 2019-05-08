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

typedef int VertexId;
typedef float Weight;
typedef unsigned char BYTE;

struct Edge {
	VertexId src;
	VertexId target;
#ifdef USE_DOMAIN
	unsigned src_domain;
	unsigned target_domain;
	Edge(VertexId _src, VertexId _target, unsigned _src_domain, unsigned _target_domain) : src(_src), target(_target), src_domain(_src_domain), target_domain(_target_domain) {}
#endif
	Edge(VertexId _src, VertexId _target) : src(_src), target(_target) {}
	Edge() : src(0), target(0) {}
	~Edge() {}
	std::string toString() {
		return "(" + std::to_string(src) + ", " + std::to_string(target) + ")";
	}
	void swap() {
		if (src > target) {
			VertexId tmp = src;
			src = target;
			target = tmp;
#ifdef USE_DOMAIN
			unsigned domain = src_domain;
			src_domain = target_domain;
			target_domain = domain;
#endif
		}
	}
};

class EdgeComparator {
public:
	int operator()(const Edge& oneEdge, const Edge& otherEdge) {
		if(oneEdge.src == otherEdge.src) {
			return oneEdge.target > otherEdge.target;
		} else {
			return oneEdge.src > otherEdge.src;
		}
	}
};

struct LabeledEdge {
	VertexId src;
	VertexId target;
	BYTE src_label;
	BYTE target_label;
	LabeledEdge(VertexId _src, VertexId _target, BYTE _src_label, BYTE _target_label) : src(_src), target(_target), src_label(_src_label), target_label(_target_label) {}
	LabeledEdge() : src(0), target(0), src_label(0), target_label(0) {}
	std::string toString() {
		return "(" + std::to_string(src) + ", " + std::to_string(target) + ")";
	}
}__attribute__((__packed__));

/*
 *  Graph mining support. Join on all keys for each vertex tuple.
 *  Each element in the tuple contains 8 bytes, first 4 bytes is vertex id,
 *  second 4 bytes contains edge label(1byte) + vertex label(1byte) + history info(1byte).
 *  History info is used to record subgraph structure.
 *
 *
 *  [ ] [ ] [ ] [ ] || [ ] [ ] [ ] [ ]
 *    vertex id        idx  el  vl info
 *     4 bytes          1   1   1    1
 *
 * */
struct LabeledElement {
	VertexId vertex_id;
	BYTE key_index;
	BYTE edge_label;
	BYTE vertex_label;
	BYTE history_info;
	LabeledElement() { }
	LabeledElement(VertexId _vertex_id) :
		vertex_id(_vertex_id), key_index(0), edge_label(0), vertex_label(0), history_info(0) { }
	LabeledElement(VertexId _vertex_id, BYTE _edge_label, BYTE _vertex_label) :
		vertex_id(_vertex_id), key_index(0), edge_label(_edge_label), vertex_label(_vertex_label), history_info(0) { }
	LabeledElement(VertexId _vertex_id, BYTE _edge_label, BYTE _vertex_label, BYTE _history) :
				vertex_id(_vertex_id), key_index(0), edge_label(_edge_label), vertex_label(_vertex_label), history_info(_history) { }
	LabeledElement(VertexId _vertex_id, BYTE _key_index, BYTE _edge_label, BYTE _vertex_label, BYTE _history) :
		vertex_id(_vertex_id), key_index(_key_index), edge_label(_edge_label), vertex_label(_vertex_label), history_info(_history) { }
	~LabeledElement() { }
	inline void set_vertex_id(VertexId new_id) { vertex_id = new_id; }
	inline int cmp(const LabeledElement& other) const {
		//compare vertex id
		if(vertex_id < other.vertex_id) return -1;
		if(vertex_id > other.vertex_id) return 1;
		//compare history info
		if(history_info < other.history_info) return -1;
		if(history_info > other.history_info) return 1;
		//compare vertex label
		if(vertex_label < other.vertex_label) return -1;
		if(vertex_label > other.vertex_label) return 1;
		//compare edge label
		if(edge_label < other.edge_label) return -1;
		if(edge_label > other.edge_label) return 1;
		//compare index
		if(key_index < other.key_index) return -1;
		if(key_index > other.key_index) return 1;
		return 0;
	}
};

inline std::ostream & operator<<(std::ostream & strm, const LabeledElement& element) {
	strm << "[" << element.vertex_id << ", " << (int)element.key_index << ", " << (int)element.edge_label << ", "
			<< (int)element.vertex_label << ", " << (int)element.history_info << "]";
	return strm;
}

inline std::ostream & operator<<(std::ostream & strm, const std::vector<LabeledElement>& tuple) {
	if(tuple.empty()){
		strm << "(empty)";
		return strm;
	}
	strm << "(";
	for(auto it = tuple.begin(); it != tuple.end() - 1; ++it) {
		strm << (*it) << ", ";
	}
	strm << tuple.back();
	strm << ")";
	return strm;
}

struct Base_Element {
	VertexId id;
	//Base_Element() : id(0) {}
	Base_Element(VertexId vid) : id(vid) {}
	~Base_Element() {}
};

inline std::ostream & operator<<(std::ostream & strm, const Base_Element& element) {
	strm << "[" << element.id << "]";
	return strm;
}

inline std::ostream & operator<<(std::ostream & strm, const std::vector<Base_Element>& tuple) {
	if(tuple.empty()){
		strm << "(empty)";
		return strm;
	}
	strm << "(";
	for(auto it = tuple.begin(); it != tuple.end() - 1; ++it) {
		strm << (*it) << ", ";
	}
	strm << tuple.back();
	strm << ")";
	return strm;
}

struct StructuralElement {
	VertexId vertex_id;
	BYTE history_info;
	StructuralElement() { }
	StructuralElement(VertexId _vertex_id) : vertex_id(_vertex_id), history_info(0) { }
	StructuralElement(VertexId _vertex_id, BYTE _history) : vertex_id(_vertex_id), history_info(_history) { }
	StructuralElement(VertexId _vertex_id, BYTE _edge_label, BYTE _vertex_label, BYTE _history) :
		vertex_id(_vertex_id), history_info(_history) { }
	StructuralElement(VertexId _vertex_id, BYTE _key_index, BYTE _edge_label, BYTE _vertex_label, BYTE _history) :
		vertex_id(_vertex_id), history_info(_history) { }
	~StructuralElement() { }
	inline void set_vertex_id(VertexId new_id) { vertex_id = new_id; }
	inline int cmp(const StructuralElement& other) const {
		//compare vertex id
		if(vertex_id < other.vertex_id) return -1;
		if(vertex_id > other.vertex_id) return 1;
		//compare history info
		if(history_info < other.history_info) return -1;
		if(history_info > other.history_info) return 1;
		return 0;
	}
};

inline std::ostream & operator<<(std::ostream & strm, const StructuralElement& element) {
	strm << "[" << element.vertex_id << ", " << (int)element.history_info << "]";
	return strm;
}

inline std::ostream & operator<<(std::ostream & strm, const std::vector<StructuralElement>& embedding) {
	if (embedding.empty()) {
		strm << "(empty)";
		return strm;
	}
	strm << "(";
	for (auto it = embedding.begin(); it != embedding.end() - 1; ++ it)
		strm << (*it) << ", ";
	strm << embedding.back();
	strm << ")";
	return strm;
}

typedef unsigned SimpleElement;
#ifdef ENABLE_LABEL
#define ElementType LabeledElement
#else
#define ElementType StructuralElement
#endif

typedef std::set<int> IntSet;
typedef std::unordered_set<int> HashIntSet;
typedef std::vector<std::unordered_set<int> > HashIntSets;
typedef std::unordered_map<unsigned, unsigned> UintHashMap;
typedef std::map<unsigned, unsigned> UintMap;

//typedef std::vector<ElementType> Embedding;
class Embedding: public std::vector<ElementType> {
public:
	Embedding() { qp_id = 0xFFFFFFFF; }
	~Embedding() {}
	void set_qpid(unsigned i) { qp_id = i; }
	unsigned get_qpid() { return qp_id; }
private:
	unsigned qp_id;
};

//typedef std::vector<SimpleElement> BaseEmbedding;
class BaseEmbedding: public std::vector<SimpleElement> {
public:
	inline unsigned get_hash() const {
		bliss::UintSeqHash h;
		for(unsigned i = 0; i < size(); ++i)
			h.update(data()[i]);
		return h.get_value();
	}
};

namespace std {
	template<>
	struct hash<BaseEmbedding> {
		std::size_t operator()(const BaseEmbedding& emb) const {
			return std::hash<int>()(emb.get_hash());
		}
	};
}

#endif /* TYPE_HPP_ */
