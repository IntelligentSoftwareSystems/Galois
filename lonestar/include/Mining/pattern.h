/*
 * pattern.hpp
 *
 *  Created on: Jul 3, 2017
 *      Author: icuzzq
 */

#ifndef CORE_PATTERN_HPP_
#define CORE_PATTERN_HPP_

#include "quick_pattern.h"
#include "canonical_graph.h"
#include "mining_tuple.h"

class Pattern {

	/**
	 * The hook function that prints the found automorphisms.
	 * \a param must be a file descriptor (FILE *).
	 */
	static void report_aut(void* param, const unsigned int n,
			const unsigned int* aut) {
//		assert(param);
//		fprintf((FILE*) param, "Generator: ");
//		bliss::print_permutation((FILE*) param, n, aut, 1);
//		fprintf((FILE*) param, "\n");
	}

public:

	static bliss::AbstractGraph* turn_canonical_graph_bliss(Quick_Pattern & sub_graph, const bool is_directed) {
//		std::cout << "quick_pattern: \t" << sub_graph << std::endl;
		bliss::AbstractGraph* ag = 0;
		//read graph from tuple
		ag = readGraph(sub_graph, is_directed);
		//std::cout << "done read." << std::endl;
		//turn to canonical form
		bliss::AbstractGraph* cf = turnCanonical(ag);
		//std::cout << "done turn." << std::endl;
		delete ag;
		ag = 0;
		return cf;
	}

	static Canonical_Graph* turn_canonical_graph(Quick_Pattern & sub_graph, const bool is_directed){
		bliss::AbstractGraph* cf_bliss = turn_canonical_graph_bliss(sub_graph, is_directed);
//		std::cout << "done bliss." << std::endl;
		Canonical_Graph* cf = new Canonical_Graph(cf_bliss, is_directed);
		delete cf_bliss;
		return cf;
	}


	static bliss::AbstractGraph* turn_canonical_graph_bliss(std::vector<Element_In_Tuple> & sub_graph, const bool is_directed) {
		bliss::AbstractGraph* ag = 0;

		//read graph from tuple
		ag = readGraph(sub_graph, is_directed);
//		std::cout << "done read." << std::endl;

		//turn to canonical form
		bliss::AbstractGraph* cf = turnCanonical(ag);
//		std::cout << "done turn." << std::endl;

		delete ag;
		ag = 0;
		return cf;
	}

	static Canonical_Graph* turn_canonical_graph(std::vector<Element_In_Tuple> & sub_graph, const bool is_directed){
		bliss::AbstractGraph* cf_bliss = turn_canonical_graph_bliss(sub_graph, is_directed);
//		std::cout << "done bliss." << std::endl;
		Canonical_Graph* cf = new Canonical_Graph(cf_bliss, is_directed);
		delete cf_bliss;
		return cf;
	}

	static bool is_automorphism(std::vector<Element_In_Tuple> & sub_graph, const bool vertex_existed) {
		assert(sub_graph.size() >= 2);
		Element_In_Tuple last_tuple = sub_graph.back();

		//check with the first element
		if(last_tuple.vertex_id < sub_graph.front().vertex_id){
			return true;
		}

		//check loop edge
		if(last_tuple.vertex_id == sub_graph[sub_graph[last_tuple.history_info].history_info].vertex_id){
			return true;
		}

		std::pair<VertexId, VertexId> added_edge;
		getEdge(sub_graph, sub_graph.size() - 1, added_edge);

		//check to see if there already exists the vertex added; if so, just allow to add edge which is (smaller id -> bigger id)
		if(vertex_existed && added_edge.first > added_edge.second){
			return true;
		}


		for(unsigned int index = last_tuple.history_info + 1; index < sub_graph.size() - 1; ++index){
			std::pair<VertexId, VertexId> edge;
			getEdge(sub_graph, index, edge);
			int cmp = compare(added_edge, edge);
//			assert(cmp != 0);
			if(cmp <= 0){
				return true;
			}
		}

		return false;
	}

	static bool is_automorphism(MTuple & sub_graph, const bool vertex_existed) {
		assert(sub_graph.get_size() >= 2);
		Element_In_Tuple last_tuple = sub_graph.at(sub_graph.get_size() - 1);

		//check with the first element
		if(last_tuple.vertex_id < sub_graph.at(0).vertex_id){
			return true;
		}

		//check loop edge
		if(last_tuple.vertex_id == sub_graph.at(sub_graph.at(last_tuple.history_info).history_info).vertex_id){
			return true;
		}

		std::pair<VertexId, VertexId> added_edge;
		getEdge(sub_graph, sub_graph.get_size() - 1, added_edge);

		//check to see if there already exists the vertex added; if so, just allow to add edge which is (smaller id -> bigger id)
		if(vertex_existed && added_edge.first > added_edge.second){
			return true;
		}


		for(unsigned int index = last_tuple.history_info + 1; index < sub_graph.get_size() - 1; ++index){
			std::pair<VertexId, VertexId> edge;
			getEdge(sub_graph, index, edge);
			int cmp = compare(added_edge, edge);
//			assert(cmp != 0);
			if(cmp <= 0){
				return true;
			}
		}

		return false;
	}

//	static bool is_automorphism(MTuple_vector & sub_graph, const bool vertex_existed) {
//		return is_automorphism(sub_graph.get_elements(), vertex_existed);
//	}

	static bool is_automorphism_init(std::vector<Element_In_Tuple> & sub_graph) {
		assert(sub_graph.size() == 2);
		//check with the first element
		return sub_graph.back().vertex_id <= sub_graph.front().vertex_id;
	}

	static bool is_automorphism_init_clique(std::vector<Base_Element> & sub_graph) {
		assert(sub_graph.size() == 2);
		//check with the first element
		return sub_graph.back().id <= sub_graph.front().id;
	}


//	static void turn_quick_pattern_sideffect(std::vector<Element_In_Tuple> & sub_graph) {
//		std::unordered_map<VertexId, VertexId> map;
//		VertexId new_id = 1;
//
//		for(unsigned int i = 0; i < sub_graph.size(); i++) {
//			VertexId old_id = sub_graph.at(i).vertex_id;
//
//			auto iterator = map.find(old_id);
//			if(iterator == map.end()) {
//				sub_graph.at(i).set_vertex_id(new_id);
//				map[old_id] = new_id++;
//
//			} else {
//				sub_graph.at(i).set_vertex_id(iterator->second);
//			}
//		}
//
//	}
//
//	static void turn_quick_pattern_pure(std::vector<Element_In_Tuple> & sub_graph, std::vector<Element_In_Tuple> & graph_quick_pattern) {
//			std::unordered_map<VertexId, VertexId> map;
//			VertexId new_id = 1;
//
//			for(unsigned int i = 0; i < sub_graph.size(); i++) {
//				Element_In_Tuple old_tuple = sub_graph[i];
//				Element_In_Tuple * new_tuple = new Element_In_Tuple(old_tuple.vertex_id, (BYTE)0, old_tuple.edge_label, old_tuple.vertex_label, old_tuple.history_info);
//
//				VertexId old_id = new_tuple->vertex_id;
//				auto iterator = map.find(old_id);
//				if(iterator == map.end()) {
//					new_tuple->set_vertex_id(new_id);
//					map[old_id] = new_id++;
//
//				} else {
//					new_tuple->set_vertex_id(iterator->second);
//				}
//
//				graph_quick_pattern.push_back(*new_tuple);
//				delete new_tuple;
//			}
//
//		}

//	static void turn_quick_pattern_pure(std::vector<Element_In_Tuple> & sub_graph, Quick_Pattern & graph_quick_pattern, bool label_flag) {
//		std::unordered_map<VertexId, VertexId> map;
//		VertexId new_id = 1;
//
//		for(unsigned int i = 0; i < sub_graph.size(); i++) {
//			Element_In_Tuple old_tuple = sub_graph[i];
//
//			Element_In_Tuple * new_tuple;
//			if(label_flag)
//				new_tuple = new Element_In_Tuple(old_tuple.vertex_id, old_tuple.key_index, (BYTE)0, old_tuple.vertex_label, old_tuple.history_info);
//			else
//				new_tuple = new Element_In_Tuple(old_tuple.vertex_id, old_tuple.key_index, (BYTE)0, (BYTE)0, old_tuple.history_info);
//
//			VertexId old_id = new_tuple->vertex_id;
//			auto iterator = map.find(old_id);
//			if(iterator == map.end()) {
//				new_tuple->set_vertex_id(new_id);
//				map[old_id] = new_id++;
//
//			} else {
//				new_tuple->set_vertex_id(iterator->second);
//			}
//
//			graph_quick_pattern.push(*new_tuple);
//			delete new_tuple;
//		}
//	}

	static void turn_quick_pattern_pure(MTuple & sub_graph, Quick_Pattern & graph_quick_pattern, bool label_flag) {
		copy_elements(sub_graph, graph_quick_pattern);

		std::unordered_map<VertexId, VertexId> map;
		VertexId new_id = 1;

		for(unsigned int i = 0; i < graph_quick_pattern.get_size(); i++) {
			Element_In_Tuple& element = graph_quick_pattern.at(i);

//			Element_In_Tuple * new_tuple;
//			if(label_flag)
//				new_tuple = new Element_In_Tuple(old_tuple.vertex_id, old_tuple.key_index, (BYTE)0, old_tuple.vertex_label, old_tuple.history_info);
//			else
//				new_tuple = new Element_In_Tuple(old_tuple.vertex_id, old_tuple.key_index, (BYTE)0, (BYTE)0, old_tuple.history_info);

			if(!label_flag){
				element.vertex_label = (BYTE)0;
			}

			VertexId old_id = element.vertex_id;
			auto iterator = map.find(old_id);
			if(iterator == map.end()) {
				element.set_vertex_id(new_id);
				map[old_id] = new_id++;

			} else {
				element.set_vertex_id(iterator->second);
			}

//			graph_quick_pattern.push(*new_tuple);
//			delete new_tuple;
		}
	}




private:

	static void copy_elements(MTuple & sub_graph, Quick_Pattern & graph_quick_pattern){
		std::memcpy(graph_quick_pattern.get_elements(), sub_graph.get_elements(), graph_quick_pattern.get_size() * sizeof(Element_In_Tuple));
	}

	//generate a pair which consists of <keyId, addedId>
	static void getEdge(std::vector<Element_In_Tuple> & sub_graph, unsigned int index, std::pair<VertexId, VertexId>& edge){
		Element_In_Tuple tuple = sub_graph[index];
		edge.first = sub_graph[tuple.history_info].vertex_id;
		edge.second = tuple.vertex_id;
		assert(edge.first != edge.second);
	}

	static void getEdge(MTuple & sub_graph, unsigned int index, std::pair<VertexId, VertexId>& edge){
		Element_In_Tuple tuple = sub_graph.at(index);
		edge.first = sub_graph.at(tuple.history_info).vertex_id;
		edge.second = tuple.vertex_id;
		assert(edge.first != edge.second);
	}

	static void swap(std::pair<VertexId, VertexId>& pair){
		if (pair.first > pair.second) {
			VertexId tmp = pair.first;
			pair.first = pair.second;
			pair.second = tmp;
		}
	}

	static int compare(std::pair<VertexId, VertexId>& oneEdge, std::pair<VertexId, VertexId>& otherEdge){
		swap(oneEdge);
		swap(otherEdge);

		if(oneEdge.first == otherEdge.first){
			return oneEdge.second - otherEdge.second;
		}
		else{
			return oneEdge.first - otherEdge.first;
		}
	}

	static bliss::AbstractGraph* readGraph(Quick_Pattern & sub_graph, bool opt_directed){
		bliss::AbstractGraph* g = 0;

		//get the number of vertices
		std::unordered_map<VertexId, BYTE> vertices;
		for(unsigned int index = 0; index < sub_graph.get_size(); ++index){
			Element_In_Tuple tuple = sub_graph.at(index);
			vertices[tuple.vertex_id] = tuple.vertex_label;
		}
//		//for debugging only
//		for(auto it = vertices.begin(); it != vertices.end(); ++it){
//			std::cout << it->first << ": " << (int)it->second << std::endl;
//		}

		//construct graph
		const unsigned int number_vertices = vertices.size();
		assert(!opt_directed);
		if(opt_directed){
			g = new bliss::Digraph(vertices.size());
		}
		else{
			g = new bliss::Graph(vertices.size());
		}

		//set vertices
		for(unsigned int i = 0; i < number_vertices; ++i){
//			std::cout << i << ": " << (unsigned int)vertices[i + 1] << std::endl;
			g->change_color(i, (unsigned int)vertices[i + 1]);
		}

		//read edges
		assert(sub_graph.get_size() > 1);
		for(unsigned int index = 1; index < sub_graph.get_size(); ++index){
			Element_In_Tuple tuple = sub_graph.at(index);
			VertexId from = sub_graph.at(tuple.history_info).vertex_id;
			VertexId to = tuple.vertex_id;
			g->add_edge(from - 1, to - 1);
		}

		return g;
	}

	static bliss::AbstractGraph* readGraph(std::vector<Element_In_Tuple> & sub_graph, bool opt_directed){
		bliss::AbstractGraph* g = 0;

		//get the number of vertices
		std::unordered_map<VertexId, BYTE> vertices;
		for(unsigned int index = 0; index < sub_graph.size(); ++index){
			Element_In_Tuple tuple = sub_graph[index];
			vertices[tuple.vertex_id] = tuple.vertex_label;
		}
//		//for debugging only
//		for(auto it = vertices.begin(); it != vertices.end(); ++it){
//			std::cout << it->first << ": " << (int)it->second << std::endl;
//		}

		//construct graph
		const unsigned int number_vertices = vertices.size();
		assert(!opt_directed);
		if(opt_directed){
			g = new bliss::Digraph(vertices.size());
		}
		else{
			g = new bliss::Graph(vertices.size());
		}

		//set vertices
		for(unsigned int i = 0; i < number_vertices; ++i){
//			std::cout << i << ": " << (unsigned int)vertices[i + 1] << std::endl;
			g->change_color(i, (unsigned int)vertices[i + 1]);
		}

		//read edges
		assert(sub_graph.size() > 1);
		for(unsigned int index = 1; index < sub_graph.size(); ++index){
			Element_In_Tuple tuple = sub_graph[index];
			VertexId from = sub_graph[tuple.history_info].vertex_id;
			VertexId to = tuple.vertex_id;
			g->add_edge(from - 1, to - 1);
		}

		return g;
	}

	static bliss::AbstractGraph* turnCanonical(bliss::AbstractGraph* ag){
		//canonical labeling
		bliss::Stats stats;
		const unsigned int* cl = ag->canonical_form(stats, &report_aut, stdout);

		//permute to canonical form
		bliss::AbstractGraph* cf = ag->permute(cl);

//		delete[] cl;
		return cf;
	}

//	static std::vector<Element_In_Tuple> writeTuple(bliss::AbstractGraph* cf){
//		std::vector<Element_In_Tuple> out;
//		//TODO
//
//		return out;
//	}

	static void writeOutGraph(bliss::AbstractGraph* graph, std::string& outfile){
		FILE* const fp = fopen(outfile.c_str(), "w");
		if (!fp)
			printf("Cannot open '%s' for outputting the canonical form, aborting", outfile.c_str());
		graph->write_dimacs(fp);
		fclose(fp);
	}

};

#endif /* CORE_PATTERN_HPP_ */
