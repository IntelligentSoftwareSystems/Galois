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

// canonical graph
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
	std::istream & read_txt(std::istream &is) {
		char line[1024];
		std::vector<std::string> result;
		clear();
		while(true) {
			unsigned pos = is.tellg();
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
				unsigned id    = atoi(result[1].c_str());
				this->resize(id + 1);
				(*this)[id].label = atoi(result[2].c_str());
				(*this)[id].global_vid = id;
			} else if(result[0] == "e" && result.size() >= 4) {
				VeridT from   = atoi(result[1].c_str());
				VeridT to     = atoi(result[2].c_str());
				LabelT elabel = atoi(result[3].c_str());
				if((VeridT)size() <= from || (VeridT)size() <= to) {
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
		//int num_vertices, num_edges;
		//unsigned pos = is.tellg();
		is.getline(line, 1024);
		result.clear();
		split(line, result);
		if(result.empty()) {
			std::cerr << "Empty first line" << std::endl;
		} else {
			//num_vertices = atoi(result[0].c_str());
			//num_edges = atoi(result[1].c_str());
		}
		VeridT vertex_id = 0;
		while(true) {
			//pos = is.tellg();
			if(!is.getline(line, 1024)) break;
			result.clear();
			split(line, result);
			if(result.empty()) {
			} else {
				this->resize(vertex_id + 1);
				(*this)[vertex_id].label = atoi(result[0].c_str()) - 1;
				(*this)[vertex_id].global_vid = vertex_id;
				for(size_t i = 1; i < result.size(); i++) {
					VeridT to = atoi(result[i++].c_str()) - 1;
					LabelT elabel = atoi(result[i].c_str()) - 1;
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
	std::istream &read_partition_info(std::istream &, int part_id);  // read
	std::istream &read_partition_info_non_locals(std::istream &);  // read
	std::istream &read_adj_par(std::istream &);  // read
	std::istream &read_local_adjp(std::istream &is);
	std::ofstream &write_local_adjp(std::ofstream &os); // write
	void check(void);
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
	void delete_edge(int from, int to);
	void delete_vertices(std::vector<int> local_ids);
	std::string to_string() const;
	static size_t get_serialized_size(const CGraph &grph);
	static size_t get_serialized_size(char *buffer, size_t buffer_size);
	static size_t serialize(const CGraph &grph, char *buffer, size_t buffer_size);
	static size_t deserialize(CGraph &grph, char *buffer, size_t buffer_size);
	static size_t get_serialized_size_db(char *buffer, size_t buffer_size);
	size_t get_serialized_size_for_partition(int vid);
	size_t serialize_neighbors_for_partition(int global_vid, int *buffer, int buffer_size);
	size_t get_serialized_size_for_partition(int vid, int requester_partition_id);
	size_t serialize_neighbors_for_partition(int requester_partition_id, int global_vid, int* &buffer, int buffer_size);
	size_t serialize_neighbors_for_partition(int requester_partition_id, int global_vid, int* &buffer, int buffer_size, std::set<int> &exclusions);
*/
};

#endif
