// defines the embedding element classes
// LabeledElement: vertex_id, history_info, vertex_label, edge_label. Used for FSM.
// StructuralElement: vertex_id, history_info. Used for Motifs.
// SimpleElement: vertex_id. Used for KCL and TC.
#ifndef ELEMENT_HPP_
#define ELEMENT_HPP_
/**
 * Code from on below link. Modified under Galois.
 *
 * https://github.com/rstream-system/RStream/
 *
 * Copyright (c) 2018, Kai Wang and the respective contributors
 * All rights reserved.
 * Reused/revised under 3-BSD
 */

#include "pangolin/gtypes.h"

// This is the data structure used in RStream.
// Each element contains 8 bytes, first 4 bytes is vertex id,
// second 4 bytes contains edge label(1byte) + vertex label(1byte) + history info(1byte).
// History info is used to record subgraph structure.
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
	~LabeledElement() = default;
	inline void set_vertex_id(VertexId new_id) { vertex_id = new_id; }
	inline void set_history_info(BYTE his) { history_info = his; }
	inline void set_vertex_label(BYTE lab) { vertex_label = lab; }
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
	bool has_history() { return true; }
	friend std::ostream & operator<<(std::ostream & strm, const LabeledElement& element) {
		strm << "[" << element.get_vid() << ", " //<< (int)element.get_key() << ", " << (int)element.get_elabel() << ", "
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
	~StructuralElement() = default;
	inline void set_vertex_id(VertexId new_id) { vertex_id = new_id; }
	inline void set_history_info(BYTE his) { history_info = his; }
	inline void set_vertex_label(BYTE lab) { }
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
	BYTE get_vlabel() const { return 0; }
	BYTE get_key() const { return 0; }
	bool has_history() { return true; }
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
	SimpleElement(VertexId _vertex_id, BYTE _edge_label, BYTE _vertex_label, BYTE _history) : vertex_id(_vertex_id) { }
	SimpleElement(VertexId _vertex_id, BYTE _key_index, BYTE _edge_label, BYTE _vertex_label, BYTE _history) : vertex_id(_vertex_id) { }
	~SimpleElement() = default;
	inline void set_vertex_id(VertexId new_id) { vertex_id = new_id; }
	inline void set_history_info(BYTE his) { }
	inline void set_vertex_label(BYTE lab) { }
	VertexId get_vid() const { return vertex_id; }
	BYTE get_his() const { return 0; }
	BYTE get_key() const { return 0; }
	BYTE get_vlabel() const { return 0; }
	bool has_history() { return false; }
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
