#ifndef __DFSCODE_H__
#define __DFSCODE_H__
/**
 * Code from on below link. Modified under Galois.
 *
 * https://github.com/zakimjz/DistGraph
 *
 * Copyright (c) 2017 Nilothpal Talukder and Mohammed J. Zaki
 * All rights reserved.
 * Reused/revised under 3-BSD
 */

typedef unsigned VeridT;
typedef unsigned LabelT;
typedef std::vector<VeridT> RMPath;

struct LabEdge {
	VeridT src;
	VeridT dst;
	LabelT elabel;
	unsigned eid;
	LabEdge() : src(0), dst(0), elabel(0), eid(0) {}
	LabEdge(VeridT src_, VeridT dst_, LabelT elabel_, unsigned eid_) :
		src(src_), dst(dst_), elabel(elabel_), eid(eid_) {}
	std::string to_string() const {
		std::stringstream ss;
		ss << "e(" << src << "," << dst << "," << elabel << ")";
		return ss.str();
	}
};
typedef std::vector<LabEdge *> LabEdgeList;

// Used for construct canonical graph
class Vertex {
public:
	LabelT label;
	std::vector<Edge> edge; //neighbor list
	void push(VeridT src_, VeridT dst_, LabelT elabel_ = 0) {
		edge.resize(edge.size() + 1);
		edge[edge.size() - 1].src = src_;
		edge[edge.size() - 1].dst = dst_;
		//edge[edge.size() - 1].elabel = elabel_;
		return;
	}
	bool find(VeridT src_, VeridT dst_, Edge &result) const {
		for(size_t i = 0; i < edge.size(); i++) {
			if(edge[i].src == src_ && edge[i].dst == dst_) {
				result = edge[i];
				return true;
			}
		}
		return false;
	}
	std::string to_string() const {
		std::stringstream ss;
		ss << "vlabel: " << label << ", ";
		for (auto e : edge) ss << e.to_string() << " ";
		return ss.str();
	}
};

// Canonical graph used for canonical check.
// A pattern (DFSCode) is converted to a canonical graph
// to perform a canonical check (minimal DFSCode)
class CGraph : public std::vector<Vertex> {
private:
	unsigned num_edges;
	bool directed;
public:
	typedef std::vector<Vertex>::iterator vertex_iterator;
	CGraph() : num_edges(0), directed(false) {}
	CGraph(bool _directed) { directed = _directed; }
	unsigned edge_size() const { return num_edges; }
	unsigned vertex_size() const { return (unsigned)size(); } // wrapper
	bool is_directed() const { return directed; }
	void buildEdge() {
		std::pair<unsigned,unsigned> new_edge;
		std::map <std::pair<unsigned,unsigned>, unsigned> edge_map;
		unsigned id = 0;
		for(size_t src = 0; src < size(); ++ src) {
			for(auto it = (*this)[src].edge.begin(); it != (*this)[src].edge.end(); ++it) {
				auto dst = it->dst;
				if(directed || src <= dst) new_edge = std::make_pair(src, dst);
				else new_edge = std::make_pair(dst, src);
				// Assign unique id's for the edges.
				if(edge_map.find(new_edge) == edge_map.end()) {
					//it->id = id;
					edge_map[new_edge] = id;
					++id;
				} else {
					//it->id = edge_map[new_edge];
				}
			}
		}
		num_edges = id;
		//if (debug) std::cout << "num_edges = " << num_edges << "\n";
	}
	std::string to_string() const {
		std::stringstream ss;
		for(size_t vid = 0; vid < size(); ++ vid)
			ss << vid << ": " << (*this)[vid].to_string() << "\n";
		return ss.str();
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
		return (d1.from == d2.from && d1.to == d2.to 
				&& d1.fromlabel == d2.fromlabel
				&& d1.elabel == d2.elabel 
				&& d1.tolabel == d2.tolabel);
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
		for(auto it = begin(); it != end(); ++it) {
			g.resize(std::max(it->from, it->to) + 1);
			if(it->fromlabel != (LabelT)-1)
				g[it->from].label = it->fromlabel;
			if(it->tolabel != (LabelT)-1)
				g[it->to].label = it->tolabel;
			g[it->from].push(it->from, it->to, it->elabel);
			if(g.is_directed() == false)
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

// An embedding consists of an edge (pointer) 
// and an embedding pointer to its parent embedding
struct BaseEdgeEmbedding {
	unsigned num_vertices;
	Edge *edge;
	BaseEdgeEmbedding *prev;
	BaseEdgeEmbedding() : num_vertices(0), edge(0), prev(0) {};
	std::string to_string() const {
		std::stringstream ss;
		ss << "[" << edge->to_string() << "]";
		return ss.str();
	}
	std::string to_string_all() {
		std::vector<Edge> ev;
		ev.push_back(*edge);
		for (BaseEdgeEmbedding *p = prev; p; p = p->prev) {
			ev.push_back(*(p->edge));
		}
		std::reverse(ev.begin(), ev.end());
		std::stringstream ss;
		for (size_t i = 0; i < ev.size(); i++) {
			ss << ev[i].to_string() << "; ";
		}
		return ss.str();
	}
};

struct LabEdgeEmbedding {
	unsigned num_vertices;
	LabEdge *edge;
	LabEdgeEmbedding *prev;
	LabEdgeEmbedding() : num_vertices(0), edge(0), prev(0) {};
	std::string to_string() const {
		std::stringstream ss;
		ss << "[" << edge->to_string() << "]";
		return ss.str();
	}
	std::string to_string_all() {
		std::vector<LabEdge> ev;
		ev.push_back(*edge);
		for (LabEdgeEmbedding *p = prev; p; p = p->prev) {
			ev.push_back(*(p->edge));
		}
		std::reverse(ev.begin(), ev.end());
		std::stringstream ss;
		for (size_t i = 0; i < ev.size(); i++) {
			ss << ev[i].to_string() << "; ";
		}
		return ss.str();
	}
};

// Embedding list
class BaseEdgeEmbeddingList : public std::vector<BaseEdgeEmbedding> {
public:
	void push(int n, Edge *edge, BaseEdgeEmbedding *prev) {
		BaseEdgeEmbedding d;
		d.num_vertices = n;
		d.edge = edge;
		d.prev = prev;
		push_back(d);
	}
	std::string to_string() const {
		std::stringstream ss;
		for(size_t i = 0; i < size(); i++)
			ss << (*this)[i].to_string() << "; ";
		return ss.str();
	}
};

class LabEdgeEmbeddingList : public std::vector<LabEdgeEmbedding> {
public:
	void push(int n, LabEdge *edge, LabEdgeEmbedding *prev) {
		LabEdgeEmbedding d;
		d.num_vertices = n;
		d.edge = edge;
		d.prev = prev;
		push_back(d);
	}
	std::string to_string() const {
		std::stringstream ss;
		for(size_t i = 0; i < size(); i++)
			ss << (*this)[i].to_string() << "; ";
		return ss.str();
	}
};

typedef std::map<int, std::map <int, BaseEdgeEmbeddingList> >                 EmbeddingLists2D;
typedef std::map<int, BaseEdgeEmbeddingList>                                  EmbeddingLists1D;
typedef std::map<int, std::map <int, std::map <int, LabEdgeEmbeddingList> > > LabEmbeddingLists3D;
typedef std::map<int, std::map <int, LabEdgeEmbeddingList> >                  LabEmbeddingLists2D;
typedef std::map<int, LabEdgeEmbeddingList>                                   LabEmbeddingLists1D;

// Stores information of edges/nodes that were already visited in the
// current DFS branch of the search.
// TODO: change type 'Edge' to 'LabEdge' to enable edge label
class History : public std::vector<Edge*> {
private:
	std::set<int> edge;
	std::set<int> vertex;
public:
	bool hasEdge(unsigned id) { return (bool)edge.count(id); }
	bool hasEdge(Edge e) {
		for(auto it = this->begin(); it != this->end(); ++it) {
			if((*it)->src == e.src && (*it)->dst == e.dst)
				return true;
			else if((*it)->src == e.dst && (*it)->dst == e.src)
				return true;
		}
		return false;
	}
	bool hasEdge(unsigned src, unsigned dst) {
		for(auto it = this->begin(); it != this->end(); ++it) {
			if((*it)->src == src && (*it)->dst == dst) return true;
			else if((*it)->src == dst && (*it)->dst == src) return true;
		}
		return false;
	}
	bool hasVertex(unsigned id) { return (bool)vertex.count(id); }
	History() {}
	History(BaseEdgeEmbedding *p) { build(p); }
	void build(BaseEdgeEmbedding *e) {
		if(e) {
			push_back(e->edge);
			vertex.insert(e->edge->src);
			vertex.insert(e->edge->dst);
			for(BaseEdgeEmbedding *p = e->prev; p; p = p->prev) {
				push_back(p->edge);       // this line eats 8% of overall instructions(!)
				vertex.insert(p->edge->src);
				vertex.insert(p->edge->dst);
			}
			std::reverse(begin(), end());
		}
	}
	std::string to_string() const {
		std::stringstream ss;
		for(size_t i = 0; i < size(); i++) {
			ss << at(i)->to_string() << "; ";
		}
		return ss.str();
	}
};

class LabHistory : public std::vector<LabEdge*> {
private:
	std::set<int> edge;
	std::set<int> vertex;
public:
	bool hasEdge(unsigned id) { return (bool)edge.count(id); }
	bool hasLabEdge(LabEdge e) {
		for(auto it = this->begin(); it != this->end(); ++it) {
			//if((*it)->src == e.src && (*it)->dst == e.dst && (*it)->elabel == e.elabel)
			if((*it)->src == e.src && (*it)->dst == e.dst)
				return true;
			//else if((*it)->src == e.dst && (*it)->dst == e.src && (*it)->elabel == e.elabel)
			else if((*it)->src == e.dst && (*it)->dst == e.src)
				return true;
		}
		return false;
	}
	bool hasVertex(unsigned id) { return (bool)vertex.count(id); }
	LabHistory() {}
	LabHistory(LabEdgeEmbedding *p) { build(p); }
	void build(LabEdgeEmbedding *e) {
		if(e) {
			push_back(e->edge);
			edge.insert(e->edge->eid);
			vertex.insert(e->edge->src);
			vertex.insert(e->edge->dst);
			for(LabEdgeEmbedding *p = e->prev; p; p = p->prev) {
				push_back(p->edge);       // this line eats 8% of overall instructions(!)
				edge.insert(p->edge->eid);
				vertex.insert(p->edge->src);
				vertex.insert(p->edge->dst);
			}
			std::reverse(begin(), end());
		}
	}
	std::string to_string() const {
		std::stringstream ss;
		for(size_t i = 0; i < size(); i++) {
			ss << at(i)->to_string() << "; ";
		}
		return ss.str();
	}
};

#endif
