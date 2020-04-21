#ifndef __DFS_CODE_HPP__
#define __DFS_CODE_HPP__

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

typedef unsigned VeridT;
typedef unsigned LabelT;
typedef std::vector<VeridT> RMPath;

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
};

inline void split(const std::string& str, std::vector<std::string>& tokens, const std::string& delimiters = " ") {
	// Skip delimiters at beginning.
	std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
	// Find first "non-delimiter".
	std::string::size_type pos = str.find_first_of(delimiters, lastPos);
	while (std::string::npos != pos || std::string::npos != lastPos) {
		// Found a token, add it to the vector.
		tokens.push_back(str.substr(lastPos, pos - lastPos));
		// Skip delimiters.  Note the "not_of"
		lastPos = str.find_first_not_of(delimiters, pos);
		// Find next "non-delimiter"
		pos = str.find_first_of(delimiters, lastPos);
	}
}

// Canonical graph used for canonical check.
// A pattern (DFSCode) is converted to a canonical graph
// to perform a canonical check (minimal DFSCode)
class CGraph : public std::vector<Vertex> {
private:
	unsigned edge_size_;
public:
	typedef std::vector<Vertex>::iterator vertex_iterator;
	std::map<int,int> global_local_id_map;
	int max_local_vid;
	bool has_ext_neighbor;
	CGraph() : edge_size_(0), directed(false) {}
	CGraph(bool _directed) { directed = _directed; }
	bool directed;
	unsigned edge_size() const { return edge_size_; }
	unsigned vertex_size() const { return (unsigned)size(); } // wrapper
	void buildEdge() {
		char buf[512];
		std::map <std::string, unsigned> tmp;
		unsigned id = 0;
		for(VeridT from = 0; from < (VeridT)size(); ++from) {
			for(Vertex::edge_iterator it = (*this)[from].edge.begin();
					it != (*this)[from].edge.end(); ++it) {
				if(directed || from <= it->to)
					std::sprintf(buf, "%d %d %d", from, it->to, it->elabel);
				else
					std::sprintf(buf, "%d %d %d", it->to, from, it->elabel);
				// Assign unique id's for the edges.
				if(tmp.find(buf) == tmp.end()) {
					it->id = id;
					tmp[buf] = id;
					++id;
				} else {
					it->id = tmp[buf];
				}
			}
		}
		edge_size_ = id;
	}
};

// A 5-tuple element in a DFSCode
class DFS {
public:
	VeridT from; // source vertex
	VeridT to;   // target vertex
	LabelT fromlabel; // source vertex label
	LabelT elabel;    // edge label
	LabelT tolabel;   // target vertex label
	friend bool operator==(const DFS &d1, const DFS &d2) {
		return (d1.from == d2.from && d1.to == d2.to && d1.fromlabel == d2.fromlabel
				&& d1.elabel == d2.elabel && d1.tolabel == d2.tolabel);
	}
	friend bool operator!=(const DFS &d1, const DFS &d2) {
		return (!(d1 == d2));
	}
	friend std::ostream &operator<<(std::ostream &out, const DFS &d) {
		out << d.to_string().c_str();
		return out;
	}
	friend bool operator<(const DFS &d1, const DFS &d2){
		if(d1.from < d2.from) return true;
		if(d1.from > d2.from) return false;
		if(d1.to < d2.to) return true;
		if(d1.to > d2.to) return false;
		if(d1.fromlabel < d2.fromlabel) return true;
		if(d1.fromlabel > d2.fromlabel) return false;
		if(d1.elabel < d2.elabel) return true;
		if(d1.elabel > d2.elabel) return false;
		if(d1.tolabel < d2.tolabel) return true;
		if(d1.tolabel > d2.tolabel) return false;
		return false;
	}

	DFS() : from(0), to(0), fromlabel(0), elabel(0), tolabel(0) {}
	DFS(VeridT from, VeridT to, LabelT fromlabel, LabelT elabel, LabelT tolabel) :
		from(from), to(to), fromlabel(fromlabel), elabel(elabel), tolabel(tolabel) {}
	DFS(char *buffer, int size);
	DFS(const DFS &d) : from(d.from), to(d.to), fromlabel(d.fromlabel), elabel(d.elabel), tolabel(d.tolabel) {}
	//std::string to_string(bool print_edge_type = true) const;
	std::string to_string(bool print_edge_type = true) const {
		std::stringstream ss;
		if(print_edge_type) {
			if(is_forward()) ss << "F";
			else ss << "B";
		}
		ss << "(" << from << " " << to << " " << fromlabel << " " << elabel << " " << tolabel << ")";
		return ss.str();
	}
	bool is_forward() const { return from < to; }
	bool is_backward() const { return from > to; }
};

// DFSCode (pattern) is a sequence of 5-tuples
struct DFSCode : public std::vector<DFS> {
private:
	RMPath rmpath; // right-most path
public:
	const RMPath &get_rmpath() const { return rmpath; }
	// RMPath is in the opposite order than the DFS code, i.e., the
	// indexes into DFSCode go from higher numbers to lower numbers.
	const RMPath &buildRMPath() {
		rmpath.clear();
		VeridT old_from = (VeridT)-1;
		for(int i = size() - 1; i >= 0; --i) {
			if((*this)[i].from < (*this)[i].to &&  // forward
					(rmpath.empty() || old_from == (*this)[i].to)) {
				rmpath.push_back(i);
				old_from = (*this)[i].from;
			}
		}
		return rmpath;
	}
	// Convert current DFS code into a canonical graph.
	bool toGraph(CGraph &g) const {
		g.clear();
		for(DFSCode::const_iterator it = begin(); it != end(); ++it) {
			g.resize(std::max(it->from, it->to) + 1);
			if(it->fromlabel != (LabelT)-1)
				g[it->from].label = it->fromlabel;
			if(it->tolabel != (LabelT)-1)
				g[it->to].label = it->tolabel;
			g[it->from].push(it->from, it->to, it->elabel);
			if(g.directed == false)
				g[it->to].push(it->to, it->from, it->elabel);
		}
		g.buildEdge();
		return (true);
	}
	// Return number of nodes in the graph.
	unsigned nodeCount(void) {
		unsigned nodecount = 0;
		for(DFSCode::iterator it = begin(); it != end(); ++it)
			nodecount = std::max(nodecount, (unsigned)(std::max(it->from, it->to) + 1));
		return (nodecount);
	}
	DFSCode &operator=(const DFSCode &other) {
		if(this == &other) return *this;
		std::vector<DFS>::operator=(other);
		rmpath = other.rmpath;
		return *this;
	}
	friend bool operator==(const DFSCode &d1, const DFSCode &d2) {
		if(d1.size() != d2.size()) return false;
		for(size_t i = 0; i < d1.size(); i++)
			if(d1[i] != d2[i]) return false;
		return true;
	}
	friend bool operator<(const DFSCode &d1, const DFSCode &d2) {
		if(d1.size() < d2.size()) return true;
		else if(d1.size() > d2.size()) return false;
		for(size_t i = 0; i < d1.size(); i++) {
			if(d1[i] < d2[i]) return true;
			else if(d2[i] < d1[i]) return false;
		}
		return false;         //equal
	}
	friend std::ostream &operator<<(std::ostream &out, const DFSCode &code);
	void push(VeridT from, VeridT to, LabelT fromlabel, LabelT elabel, LabelT tolabel) {
		resize(size() + 1);
		DFS &d = (*this)[size() - 1];
		d.from = from;
		d.to = to;
		d.fromlabel = fromlabel;
		d.elabel = elabel;
		d.tolabel = tolabel;
	}
	void pop() { resize(size() - 1); }
	std::string to_string(bool print_edge_type = true) const {
		if (empty()) return "";
		std::stringstream ss;
		size_t i = 0;
		ss << (*this)[i].to_string(print_edge_type);
		i ++;
		for (; i < size(); ++i) {
			ss << ";" << (*this)[i].to_string(print_edge_type);
		}
		return ss.str();
	}
};

std::ostream &operator<<(std::ostream &out, const DFSCode &code) {
	out << code.to_string();
	return out;
}

#endif
