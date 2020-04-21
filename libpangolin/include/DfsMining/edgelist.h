#pragma once
#include "edge.h" // had a bug when using edge_type.h
#include "gtypes.h"

class EdgeList : public std::vector<SEdge> {
//using iterator = typename galois::gstl::Vector<Edge>::iterator;
//using iterator = typename std::vector<Edge>::iterator;
//using const_iterator = typename std::vector<Edge>::const_iterator;
public:
	EdgeList() {}
	//EdgeList(Graph& graph, bool directed) {
	//	init(graph, directed);
	//}
	~EdgeList() {}
	void init(Graph &graph, bool directed, bool symmetrize = false);
	//bool empty() const { return all_edges.empty(); }
	//const_iterator begin() const { return all_edges.begin(); }
	//iterator begin() { return all_edges.begin(); }
	//const_iterator end() const { return all_edges.end(); }
	//iterator end() { return all_edges.end(); }
	//const size_t size() const { return all_edges.size(); }
	//size_t get_num_vertices() const { return num_vertices; }
	//void resize (size_t n) { all_edges.resize(n); }
	//void push_back(Edge e) { all_edges.push_back(e); }
	SEdge& get_edge(size_t i) { return (*this)[i]; }
	SEdge* get_edge_ptr(size_t i) { return &((*this)[i]); }
	unsigned get_core();
	void ord_core();
	void relabel();
	unsigned generate_graph(Graph &graph);
private:
	size_t num_vertices;
	size_t num_edges;
	void add_edge(size_t pos, VertexId src, VertexId dst) {
	//	all_edges[pos] = Edge(src, dst);
		(*this)[pos] = SEdge(src,dst);
	}
	std::vector<int> rank;
};

