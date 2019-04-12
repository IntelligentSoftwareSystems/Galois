#ifndef __TYPES_HPP__
#define __TYPES_HPP__

#include <string>
#include <vector>
#include <sstream>

typedef int integer_t;
typedef integer_t int_t;
typedef unsigned int unsigned_integer_t;
typedef unsigned_integer_t uint_t;
typedef uint_t symbol_t;
typedef double double_t;
typedef char char_t;
typedef char * charp_t;
typedef float float_t;
typedef long long_t;
typedef unsigned long unsigned_long_t;
typedef unsigned long ulong_t;
typedef bool bool_t;
typedef std::string string_t;
typedef void * void_ptr_t;
typedef std::vector<int> RMPath;

struct LabEdge {
	int from;
	int to;
	int elabel;
	unsigned id;
	LabEdge() : from(0), to(0), elabel(0), id(0) {}
	LabEdge(int src, int dst, int el, unsigned eid) :
		from(src), to(dst), elabel(el), id(eid) {}
	std::string to_string() const {
		std::stringstream ss;
		ss << "e(" << from << "," << to << "," << elabel << ")";
		return ss.str();
	}
};
typedef std::vector<LabEdge *> EdgeList;

class Vertex {
public:
	typedef std::vector<LabEdge>::iterator edge_iterator;
	typedef std::vector<LabEdge>::const_iterator const_edge_iterator;
	int label, global_vid, vertex_part_id, orig_part_id;
	bool is_boundary_vertex;
	std::vector<LabEdge> edge; //neighbor list
	void push(int from, int to, int elabel) {
		edge.resize(edge.size() + 1);
		edge[edge.size() - 1].from = from;
		edge[edge.size() - 1].to = to;
		edge[edge.size() - 1].elabel = elabel;
		return;
	}
	bool find(int from, int to, LabEdge &result) const {
		for(int i = 0; i < edge.size(); i++) {
			if(edge[i].from == from && edge[i].to == to) {
				result = edge[i];
				return true;
			}
		} // for i
		return false;
	} // find
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
		for(int i = 0; i < vrtx.edge.size(); i++) {
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
};

inline void split(const std::string& str, std::vector<std::string>& tokens, const std::string& delimiters = " ") {
  // Skip delimiters at beginning.
  std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
  // Find first "non-delimiter".
  std::string::size_type pos     = str.find_first_of(delimiters, lastPos);
  while (std::string::npos != pos || std::string::npos != lastPos) {
    // Found a token, add it to the vector.
    tokens.push_back(str.substr(lastPos, pos - lastPos));
    // Skip delimiters.  Note the "not_of"
    lastPos = str.find_first_not_of(delimiters, pos);
    // Find next "non-delimiter"
    pos = str.find_first_of(delimiters, lastPos);
  }
}

#endif
