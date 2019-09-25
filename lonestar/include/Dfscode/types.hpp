#ifndef __TYPES_HPP__
#define __TYPES_HPP__

#include <set>
#include <map>
#include <vector>
#include <string>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <sstream>
#include <iterator>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include "../common_types.h"

typedef std::vector<VeridT> RMPath;
//typedef int edge_label_t;
//typedef int vertex_label_t;
//typedef std::set<edge_label_t> edge_label_set_t;
//typedef std::set<vertex_label_t> vertex_label_set_t;

// Labelled Edge
struct LabEdge {
	VeridT from;
	VeridT to;
	LabelT elabel;
	unsigned id;
	LabEdge() : from(0), to(0), elabel(0), id(0) {}
	LabEdge(VeridT src, VeridT dst, LabelT el, unsigned eid) :
		from(src), to(dst), elabel(el), id(eid) {}
	std::string to_string() const {
		std::stringstream ss;
		ss << "e(" << from << "," << to << "," << elabel << ")";
		return ss.str();
	}
};
typedef std::vector<LabEdge *> EdgeList;

// Used for construct canonical graph
class Vertex {
public:
	typedef std::vector<LabEdge>::iterator edge_iterator;
	typedef std::vector<LabEdge>::const_iterator const_edge_iterator;
	LabelT label;
	VeridT global_vid, vertex_part_id, orig_part_id;
	bool is_boundary_vertex;
	std::vector<LabEdge> edge; //neighbor list
	void push(VeridT from, VeridT to, LabelT elabel) {
		edge.resize(edge.size() + 1);
		edge[edge.size() - 1].from = from;
		edge[edge.size() - 1].to = to;
		edge[edge.size() - 1].elabel = elabel;
		return;
	}
	bool find(VeridT from, VeridT to, LabEdge &result) const {
		for(size_t i = 0; i < edge.size(); i++) {
			if(edge[i].from == from && edge[i].to == to) {
				result = edge[i];
				return true;
			}
		} // for i
		return false;
	} // find
/*
	static size_t get_serialized_size(const Vertex &vrtx) {
		return sizeof(int) + 4 * vrtx.edge.size() * sizeof(int) + sizeof(int)     + sizeof(int);
	}
	static size_t get_serialized_size(char *buffer, size_t buffer_size) {
		int s = *((int*)buffer);
		return s;
	}
	static size_t serialize(const Vertex &vrtx, char *buffer, size_t buffer_size) {
		if(buffer_size < get_serialized_size(vrtx)) throw std::runtime_error("Buffer too small.");
		int pos = 0;
		// size of this serialized vertex in bytes.
		*((int*)(buffer + pos)) = get_serialized_size(vrtx);
		pos += sizeof(int);
		// store the vertex label
		*((int*)(buffer + pos)) = vrtx.label;
		pos += sizeof(int);
		// store number of edges
		*((int*)(buffer + pos)) = vrtx.edge.size();
		pos += sizeof(int);
		for(size_t i = 0; i < vrtx.edge.size(); i++) {
			*((int*)(buffer + pos)) = vrtx.edge[i].from;
			pos += sizeof(int);
			*((int*)(buffer + pos)) = vrtx.edge[i].to;
			pos += sizeof(int);
			*((int*)(buffer + pos)) = vrtx.edge[i].elabel;
			pos += sizeof(int);
			*((int*)(buffer + pos)) = vrtx.edge[i].id;
			pos += sizeof(int);
		} // for i
		return pos;
	} // Vertex::serialize
	static size_t deserialize(Vertex &vrtx, char *buffer, size_t buffer_size) {
		// TODO: check minimum buffer size
		if(buffer_size < get_serialized_size(buffer, buffer_size)) throw std::runtime_error("Buffer too small.");
		int pos = 0;
		vrtx.edge.clear();
		// read buffer s
		pos += sizeof(int);
		// read the vertex label
		vrtx.label = *((int*)(buffer + pos));
		pos += sizeof(int);
		// read the number of edges
		int edge_count = *((int*)(buffer + pos));
		pos += sizeof(int);
		for(int i = 0; i < edge_count; i++) {
			LabEdge tmp_edge;
			tmp_edge.from = *((int*)(buffer + pos));
			pos += sizeof(int);
			tmp_edge.to = *((int*)(buffer + pos));
			pos += sizeof(int);
			tmp_edge.elabel = *((int*)(buffer + pos));
			pos += sizeof(int);
			tmp_edge.id = *((int*)(buffer + pos));
			pos += sizeof(int);
			vrtx.edge.push_back(tmp_edge);
		} // for i
		return pos;
	} // Vertex::deserialize
//*/
};

#endif
