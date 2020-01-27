#ifndef __LGRAPH_HPP__
#define __LGRAPH_HPP__

//defines the Learning Graph (LGraph) data structure
#include <set>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <algorithm>
#include "types.h"

struct Edge {
	IndexT src;
	IndexT dst;
	ValueT elabel;
	Edge() : src(0), dst(0), elabel(0) {}
	Edge(IndexT from, IndexT to, ValueT el) :
		src(from), dst(to), elabel(el) {}
	std::string to_string() const {
		std::stringstream ss;
		ss << "e(" << src << "," << dst << "," << elabel << ")";
		return ss.str();
	}
};
typedef std::vector<Edge> EdgeList;

class LGraph {
public:
	LGraph() : symmetrize_(false), directed_(false) {}
	void clean() {
		delete[] rowptr_;
		delete[] colidx_;
		delete[] weight_;
		degrees.clear();
		el.clear();
		//labels_.clear();
		//vertices.clear();
	}
	bool directed() const { return directed_; }
	size_t num_vertices() const { return num_vertices_; }
	size_t num_edges() const { return num_edges_; }
	IndexT * out_rowptr() const { return rowptr_; }
	IndexT * out_colidx() const { return colidx_; }
	unsigned out_degree(IndexT n) const { return rowptr_[n+1] - rowptr_[n]; }
	IndexT get_offset(IndexT n) { return rowptr_[n]; }
	IndexT get_dest(IndexT n) { return colidx_[n]; }
	ValueT get_weight(IndexT n) { return weight_[n]; }
	unsigned get_max_degree() { return max_degree; }
	//ValueT * labels() { return labels_.data(); }
	//ValueT get_label(IndexT n) { return labels_[n]; }
	void read_edgelist(const char *filename, bool symmetrize = false) {
		std::ifstream in;
		std::string line;
		in.open(filename, std::ios::in);
		IndexT max_vid = 0;
		while (std::getline(in, line)) {
			std::istringstream edge_stream(line);
			IndexT u, v;
			edge_stream >> u;
			edge_stream >> v;
			el.push_back(Edge(u, v, 1));
			if (symmetrize) el.push_back(Edge(v, u, 1));
			if (u > max_vid) max_vid = u;
			if (v > max_vid) max_vid = v;
		}
		in.close();
		directed_ = true;
		num_vertices_ = max_vid+1;
		num_edges_ = el.size();
		std::cout << "num_vertices_ " << num_vertices_ << " num_edges_ " << num_edges_ << "\n";
		MakeGraphFromEL();
	}

private:
	EdgeList el;
	bool symmetrize_; // whether to symmetrize a directed graph
	bool directed_;
	size_t num_vertices_;
	size_t num_edges_;
	IndexT *rowptr_;
	IndexT *colidx_;
	ValueT *weight_;
	unsigned max_degree;
	std::vector<IndexT> degrees;
	std::vector<ValueT> labels_;
	std::vector<std::vector<Edge> > vertices;

	static bool compare_id(Edge a, Edge b) { return (a.dst < b.dst); }

	void MakeGraphFromEL() {
		SquishGraph();
		MakeCSR(false);
	}

	void SquishGraph(bool remove_selfloops = true, bool remove_redundents = true) {
		std::vector<Edge> neighbors;
		for (size_t i = 0; i < num_vertices_; i++)
			vertices.push_back(neighbors);
		for (size_t i = 0; i < num_edges_; i ++)
			vertices[el[i].src].push_back(el[i]);
		el.clear();
		printf("Sorting the neighbor lists...");
		for (size_t i = 0; i < num_vertices_; i ++)
			std::sort(vertices[i].begin(), vertices[i].end(), compare_id);
		printf(" Done\n");
		//remove self loops
		int num_selfloops = 0;
		if(remove_selfloops) {
			printf("Removing self loops...");
			for(size_t i = 0; i < num_vertices_; i ++) {
				for(unsigned j = 0; j < vertices[i].size(); j ++) {
					if(i == vertices[i][j].dst) {
						vertices[i].erase(vertices[i].begin()+j);
						num_selfloops ++;
						j --;
					}
				}
			}
			printf(" %d selfloops are removed\n", num_selfloops);
			num_edges_ -= num_selfloops;
		}
		// remove redundent
		int num_redundents = 0;
		if(remove_redundents) {
			printf("Removing redundent edges...");
			for (size_t i = 0; i < num_vertices_; i ++) {
				for (unsigned j = 1; j < vertices[i].size(); j ++) {
					if (vertices[i][j].dst == vertices[i][j-1].dst) {
						vertices[i].erase(vertices[i].begin()+j);
						num_redundents ++;
						j --;
					}
				}
			}
			printf(" %d redundent edges are removed\n", num_redundents);
			num_edges_ -= num_redundents;
		}
	}

	void MakeCSR(bool transpose) {
		degrees.resize(num_vertices_);
		std::fill(degrees.begin(), degrees.end(), 0);
		for (size_t i = 0; i < num_vertices_; i ++)
			degrees[i] = vertices[i].size();
		max_degree = *(std::max_element(degrees.begin(), degrees.end()));

		std::vector<IndexT> offsets(degrees.size() + 1);
		IndexT total = 0;
		for (size_t n = 0; n < degrees.size(); n++) {
			offsets[n] = total;
			total += degrees[n];
		}
		offsets[degrees.size()] = total;

		assert(num_edges_ == offsets[num_vertices_]);
		weight_ = new ValueT[num_edges_];
		colidx_ = new IndexT[num_edges_];
		rowptr_ = new IndexT[num_vertices_+1]; 
		for (size_t i = 0; i < num_vertices_+1; i ++) rowptr_[i] = offsets[i];
		for (size_t i = 0; i < num_vertices_; i ++) {
			for (auto it = vertices[i].begin(); it < vertices[i].end(); it ++) {
				Edge e = *it;
				assert(i == e.src);
				if (symmetrize_ || (!symmetrize_ && !transpose)) {
					weight_[offsets[e.src]] = e.elabel;
					colidx_[offsets[e.src]++] = e.dst;
				}
				if (symmetrize_ || (!symmetrize_ && transpose)) {
					weight_[offsets[e.dst]] = e.elabel;
					colidx_[offsets[e.dst]++] = e.src;
				}
			}
		}
	}
};

#endif
