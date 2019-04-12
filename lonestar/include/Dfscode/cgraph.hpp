#ifndef __GRAPH_HPP__
#define __GRAPH_HPP__
#include <set>
#include <map>
#include <string>
#include <cstring>
#include <cassert>
#include <sstream>
#include <iterator>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include "types.hpp"

class CGraph; // canonical graph
typedef int edge_label_t;
typedef int vertex_label_t;
typedef std::set<edge_label_t> edge_label_set_t;
typedef std::set<vertex_label_t> vertex_label_set_t;
typedef std::vector<CGraph> graph_database_t;

class CGraph : public std::vector<Vertex> {
private:
	unsigned int edge_size_;
public:
	typedef std::vector<Vertex>::iterator vertex_iterator;
	std::map<int,int> global_local_id_map;
	int max_local_vid;
	bool has_ext_neighbor;
	CGraph() : edge_size_(0), directed(false) {}
	CGraph(bool _directed) { directed = _directed; }
	bool directed;
	unsigned int edge_size() const { return edge_size_; }
	unsigned int vertex_size() const { return (unsigned int)size(); } // wrapper
	void buildEdge() {
		char buf[512];
		std::map <std::string, unsigned int> tmp;
		unsigned int id = 0;
		for(int from = 0; from < (int)size(); ++from) {
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
	void read_csr(int m, int num_edges, int *row_offsets, int *column_indices, int *labels, int *weights) {
		for (int v = 0; v < m; ++ v) {
			this->resize(v + 1);
			(*this)[v].global_vid = v;
			if (labels != NULL)
				(*this)[v].label = labels[v];
			else 
				(*this)[v].label = rand() % 10 + 1;
			int row_begin = row_offsets[v];
			int row_end = row_offsets[v+1];
			for (int offset = row_begin; offset < row_end; offset ++) {
				int to = column_indices[offset];
				int elabel = 1;
				if (weights != NULL) elabel = weights[offset];
				//else elabel = rand() % 10 + 1;
				(*this)[v].push(v, to, elabel);
			}
		}
		this->max_local_vid = this->size() - 1;
		buildEdge();
	}
	std::istream & read_txt(std::istream &is) {
		char line[1024];
		std::vector<std::string> result;
		clear();
		while(true) {
			unsigned int pos = is.tellg();
			if(!is.getline(line, 1024)) { break; }
			result.clear();
			split(line, result);
			if(result.empty()) {
				// do nothing
			} else if(result[0] == "t") {
				if(!empty()) {   // use as delimiter
					is.seekg(pos, std::ios_base::beg);
					break;
				} else {
					// y = atoi (result[3].c_str());
				}
			} else if(result[0] == "v" && result.size() >= 3) {
				unsigned int id    = atoi(result[1].c_str());
				this->resize(id + 1);
				(*this)[id].label = atoi(result[2].c_str());
				(*this)[id].global_vid = id;
			} else if(result[0] == "e" && result.size() >= 4) {
				int from   = atoi(result[1].c_str());
				int to     = atoi(result[2].c_str());
				int elabel = atoi(result[3].c_str());
				if((int)size() <= from || (int)size() <= to) {
					std::cerr << "Format Error: define vertex lists before edges, from: " 
						<< from << "; to: " << to << "; vertex count: " << size() << std::endl;
					exit(1);
				}
				(*this)[from].push(from, to, elabel);
				if(directed == false)
					(*this)[to].push(to, from, elabel);
			}
		}
		this->max_local_vid = this->size() - 1;
		buildEdge();
		return is;
	}
	std::istream &read_adj(std::istream &is) {
		char line[1024];
		std::vector<std::string> result;
		clear();
		int num_vertices, num_edges, vertex_id;
		unsigned int pos = is.tellg();
		is.getline(line, 1024);
		result.clear();
		split(line, result);
		if(result.empty()) {
			std::cerr << "Empty first line" << std::endl;
		} else {
			num_vertices = atoi(result[0].c_str());
			num_edges = atoi(result[1].c_str());
		}
		vertex_id = 0;
		while(true) {
			pos = is.tellg();
			if(!is.getline(line, 1024)) break;
			result.clear();
			split(line, result);
			if(result.empty()) {
			} else {
				this->resize(vertex_id + 1);
				(*this)[vertex_id].label = atoi(result[0].c_str()) - 1;
				(*this)[vertex_id].global_vid = vertex_id;
				for(int i = 1; i < result.size(); i++) {
					int to = atoi(result[i++].c_str()) - 1;
					int elabel = atoi(result[i].c_str()) - 1;
					(*this)[vertex_id].push(vertex_id, to, elabel);
				}
			}
			vertex_id++;
		}
		this->max_local_vid = this->size() - 1;
		buildEdge();
		return is;
	}
/*
	void read_csr(int num_vertices, int num_edges, int *rowptr, int *colidx, int *labels, int *weights); // cxh
	std::istream &read_txt(std::istream &);  // read
	std::istream &read_adj(std::istream &);  // read
	std::ostream &write_txt(std::ostream &);  // write
	std::istream &read_partition_info(std::istream &, int part_id);  // read
	std::istream &read_partition_info_non_locals(std::istream &);  // read
	std::istream &read_adj_par(std::istream &);  // read
	std::istream &read_local_adjp(std::istream &is);
	std::ofstream &write_local_adjp(std::ofstream &os); // write
	void check(void);
*/
	int get_vertex_label(int vid) const { return at(vid).label; }
	int get_local_vid(int global_vid){
		if (global_local_id_map.count(global_vid) > 0)
			return global_local_id_map[global_vid];
		else return -1;
	}
	bool is_pseudo_local(int local_id){
		return (local_id > max_local_vid && (*this)[local_id].vertex_part_id != (*this)[local_id].orig_part_id);
	}
	bool is_external(int local_id){
		return (local_id > max_local_vid && (*this)[local_id].vertex_part_id == (*this)[local_id].orig_part_id);
	}
/*
	void delete_edge(int from, int to);
	void delete_vertices(std::vector<int> local_ids);
	std::string to_string() const;
	static size_t get_serialized_size(const CGraph &grph);
	static size_t get_serialized_size(char *buffer, size_t buffer_size);
	static size_t serialize(const CGraph &grph, char *buffer, size_t buffer_size);
	static size_t deserialize(CGraph &grph, char *buffer, size_t buffer_size);
	static size_t get_serialized_size(const graph_database_t &grph_db);
	static size_t get_serialized_size_db(char *buffer, size_t buffer_size);
	static size_t serialize(const graph_database_t &grph_db, char *buffer, size_t buffer_size);
	static size_t deserialize(graph_database_t &grph_db, char *buffer, size_t buffer_size);
	size_t get_serialized_size_for_partition(int vid);
	size_t serialize_neighbors_for_partition(int global_vid, int *buffer, int buffer_size);
	size_t get_serialized_size_for_partition(int vid, int requester_partition_id);
	size_t serialize_neighbors_for_partition(int requester_partition_id, int global_vid, int* &buffer, int buffer_size);
	size_t serialize_neighbors_for_partition(int requester_partition_id, int global_vid, int* &buffer, int buffer_size, std::set<int> &exclusions);
*/
};

#endif
