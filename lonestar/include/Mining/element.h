// defines the embedding element classes
// LabeledElement: vertex_id, history_info, vertex_label, edge_label. Used for FSM.
// StructuralElement: vertex_id, history_info. Used for Motifs.
// SimpleElement: vertex_id. Used for KCL and TC.
#ifndef ELEMENT_HPP_
#define ELEMENT_HPP_

typedef unsigned VertexId;
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
protected:
	VertexId vertex_id;
	BYTE key_index;
	BYTE edge_label;
	BYTE vertex_label;
	BYTE history_info;
public:
	LabeledElement() { }
	LabeledElement(VertexId _vertex_id) :
		vertex_id(_vertex_id), key_index(0), edge_label(0), vertex_label(0), history_info(0) { }
	LabeledElement(VertexId _vertex_id, BYTE _history) :
		vertex_id(_vertex_id), key_index(0), edge_label(0), vertex_label(0), history_info(_history) { }
	LabeledElement(VertexId _vertex_id, BYTE _edge_label, BYTE _vertex_label) :
		vertex_id(_vertex_id), key_index(0), edge_label(_edge_label), vertex_label(_vertex_label), history_info(0) { }
	LabeledElement(VertexId _vertex_id, BYTE _edge_label, BYTE _vertex_label, BYTE _history) :
		vertex_id(_vertex_id), key_index(0), edge_label(_edge_label), vertex_label(_vertex_label), history_info(_history) { }
	LabeledElement(VertexId _vertex_id, BYTE _key_index, BYTE _edge_label, BYTE _vertex_label, BYTE _history) :
		vertex_id(_vertex_id), key_index(_key_index), edge_label(_edge_label), vertex_label(_vertex_label), history_info(_history) { }
	~LabeledElement() { }
	inline void set_vertex_id(VertexId new_id) { vertex_id = new_id; }
	inline void set_history_info(BYTE his) { history_info = his; }
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
	VertexId get_vid() const { return vertex_id; }
	BYTE get_key() const { return key_index; }
	BYTE get_elabel() const { return edge_label; }
	BYTE get_vlabel() const { return vertex_label; }
	BYTE get_his() const { return history_info; }
	friend std::ostream & operator<<(std::ostream & strm, const LabeledElement& element) {
		strm << "[" << element.get_vid() << ", " << (int)element.get_key() << ", " << (int)element.get_elabel() << ", "
			<< (int)element.get_vlabel() << ", " << (int)element.get_his() << "]";
		return strm;
	}
};

struct StructuralElement {
protected:
	VertexId vertex_id;
	BYTE history_info;
public:
	StructuralElement() { }
	StructuralElement(VertexId _vertex_id) : vertex_id(_vertex_id), history_info(0) { }
	StructuralElement(VertexId _vertex_id, BYTE _history) : vertex_id(_vertex_id), history_info(_history) { }
	StructuralElement(VertexId _vertex_id, BYTE _edge_label, BYTE _vertex_label, BYTE _history) :
		vertex_id(_vertex_id), history_info(_history) { }
	StructuralElement(VertexId _vertex_id, BYTE _key_index, BYTE _edge_label, BYTE _vertex_label, BYTE _history) :
		vertex_id(_vertex_id), history_info(_history) { }
	~StructuralElement() { }
	inline void set_vertex_id(VertexId new_id) { vertex_id = new_id; }
	inline void set_history_info(BYTE his) { history_info = his; }
	inline int cmp(const StructuralElement& other) const {
		//compare vertex id
		if(vertex_id < other.vertex_id) return -1;
		if(vertex_id > other.vertex_id) return 1;
		//compare history info
		if(history_info < other.history_info) return -1;
		if(history_info > other.history_info) return 1;
		return 0;
	}
	VertexId get_vid() const { return vertex_id; }
	BYTE get_his() const { return history_info; }
	friend std::ostream & operator<<(std::ostream & strm, const StructuralElement& element) {
		strm << "[" << element.get_vid() << ", " << (int)element.get_his() << "]";
		return strm;
	}
};

//typedef unsigned SimpleElement;
struct SimpleElement {
protected:
	VertexId vertex_id;
public:
	SimpleElement() : vertex_id(0) { }
	SimpleElement(VertexId _vertex_id) : vertex_id(_vertex_id) { }
	SimpleElement(VertexId _vertex_id, BYTE _edge_label, BYTE _vertex_label, BYTE _history) { }
	SimpleElement(VertexId _vertex_id, BYTE _key_index, BYTE _edge_label, BYTE _vertex_label, BYTE _history) { }
	~SimpleElement() { }
	inline void set_vertex_id(VertexId new_id) { vertex_id = new_id; }
	VertexId get_vid() const { return vertex_id; }
	BYTE get_his() const { return 0; }
	inline int cmp(const SimpleElement& other) const {
		if(vertex_id < other.get_vid()) return -1;
		if(vertex_id > other.get_vid()) return 1;
		return 0;
	}
	friend bool operator==(const SimpleElement &e1, const SimpleElement &e2) {
		return e1.get_vid() == e2.get_vid();
	}
	friend std::ostream & operator<<(std::ostream & strm, const SimpleElement& element) {
		strm << "[" << element.get_vid() << "]";
		return strm;
	}
};

#endif
