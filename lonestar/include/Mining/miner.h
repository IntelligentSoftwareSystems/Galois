#ifndef MINER_HPP_
#define MINER_HPP_
#include "pattern.h"
#include "mining_tuple.h"
#include "quick_pattern.h"
#include "platform_atomics.h"
#include "galois/Bag.h"
#include "galois/Galois.h"
#include "galois/substrate/PerThreadStorage.h"

typedef std::unordered_map<Quick_Pattern, int> QpMap;
typedef std::unordered_map<Canonical_Graph, int> CgMap;
typedef galois::substrate::PerThreadStorage<QpMap> LocalQpMap;
typedef galois::substrate::PerThreadStorage<CgMap> LocalCgMap;
typedef std::vector<Element_In_Tuple> Embedding;
typedef galois::InsertBag<Embedding> EmbeddingQueue;

class Miner {
public:
	Miner(bool label_f, int tuple_size, int n_vertices, int n_edges, std::vector<LabeledEdge> edge_list) : label_flag(label_f), sizeof_tuple(tuple_size) {
		edge_hashmap.resize(n_vertices);
		build_edge_hashmap(n_edges, 0, edge_list);
	}
	virtual ~Miner() {};
	// given an embedding, extend it with one more edge, and if it is not automorphism, insert the new embedding into the task queue
	void join_each(unsigned max_size, Embedding in_tuple, EmbeddingQueue &out_tuples) {
		std::unordered_set<VertexId> vertices_set;
		MTuple_join mtuple(sizeof_tuple);
		mtuple.init(in_tuple, vertices_set);
		std::unordered_set<VertexId> set;
		// for each vertex in the embedding
		for(unsigned i = 0; i < mtuple.get_size(); ++i) {
			VertexId id = mtuple.at(i).vertex_id;
			if(set.find(id) == set.end()) {
				set.insert(id);
				// try edge extension
				for(Element_In_Tuple& element : edge_hashmap[id]) {
					// create a new extended embedding
					Element_In_Tuple new_element(element.vertex_id, (BYTE)0, element.edge_label, element.vertex_label, (BYTE)i);
					bool vertex_existed = gen_an_out_tuple(mtuple, new_element, (BYTE)i, vertices_set);
					// number of vertices must be smaller than k.
					// check if this is automorphism
					if(mtuple.get_num_vertices() <= max_size && !Pattern::is_automorphism(mtuple, vertex_existed)) {
						unsigned num_elements = mtuple.get_size();
						Embedding out_tuple(num_elements+1);
						for (unsigned j = 0; j < num_elements; j ++)
							out_tuple[j] = mtuple.at(j);
						// insert the new extended embedding into the queue
						out_tuples.push_back(out_tuple);
					}
					mtuple.pop();
				}
			}
		}
	}
	void aggregate(std::vector<Embedding> in_tuples) {
		int num_tuples = in_tuples.size();
		// for each embedding, turn it into a quick pattern
		for(int i = 0; i < num_tuples; i ++) {
			Quick_Pattern quick_pattern(sizeof_tuple);
			MTuple in_tuple(sizeof_tuple);
			in_tuple.init(in_tuples[i]);
			Pattern::turn_quick_pattern_pure(in_tuple, quick_pattern, label_flag);
			// update counting for this quick pattern
			if (quick_patterns_aggregation.find(quick_pattern) != quick_patterns_aggregation.end()) {
				// if this quick pattern already exists, increase its count
				quick_patterns_aggregation[quick_pattern] = quick_patterns_aggregation[quick_pattern] + 1;
				quick_pattern.clean();
				// otherwise add this quick pattern into the map, and set the count as one
			} else { quick_patterns_aggregation[quick_pattern] = 1; }
		}
		aggregate_canonical();
	}
	void clear_map() {
		quick_patterns_aggregation.clear();
		canonical_graphs_agg.clear();
	}
	void quick_aggregate_each(Embedding tuple, QpMap& quick_patterns_agg) {
		Quick_Pattern quick_pattern(sizeof_tuple);
		MTuple in_tuple(sizeof_tuple);
		in_tuple.init(tuple);
		Pattern::turn_quick_pattern_pure(in_tuple, quick_pattern, label_flag);
		if (quick_patterns_agg.find(quick_pattern) != quick_patterns_agg.end()) {
			quick_patterns_agg[quick_pattern] += 1;
			//fetch_and_add(quick_patterns_agg[quick_pattern], 1);
			//__sync_fetch_and_add(&quick_patterns_agg[quick_pattern], 1);
		} else {
			quick_patterns_agg[quick_pattern] = 1;
			//__sync_lock_test_and_set(&quick_patterns_agg[quick_pattern], 1);
		}
	}
	void canonical_aggregate(QpMap quick_patterns_agg) {
		for (auto it = quick_patterns_agg.begin(); it != quick_patterns_agg.end(); ++it) {
			Quick_Pattern sub_graph = it->first;
			int s = it->second;
			// turn it into a canonical graph (i.e. real pattern)
			Canonical_Graph* cg = Pattern::turn_canonical_graph(sub_graph, false);
			sub_graph.clean();
			// if this pattern already exists, increase its count
			if (canonical_graphs_agg.find(*cg) != canonical_graphs_agg.end()) {
				canonical_graphs_agg[*cg] = canonical_graphs_agg[*cg] + s;
			} else {
				// otherwise add this pattern into the map, and set the count as 's'
				canonical_graphs_agg[*cg] = s;
			}
			delete cg;
		}
	}
	void canonical_aggregate_each(Quick_Pattern sub_graph, int num, CgMap& canonical_graphs_agg) {
		Canonical_Graph* cg = Pattern::turn_canonical_graph(sub_graph, false);
		sub_graph.clean();
		if (canonical_graphs_agg.find(*cg) != canonical_graphs_agg.end()) {
			canonical_graphs_agg[*cg] = canonical_graphs_agg[*cg] + num;
		} else canonical_graphs_agg[*cg] = num;
		delete cg;
	}
	// filtering for FSM
	void filter_each(Embedding tuple, EmbeddingQueue& out_queue, int threshold, CgMap cg_map) {
		MTuple mtuple(sizeof_tuple);
		mtuple.init(tuple);
		// check if the pattern of this embedding is frequent
		if (!filter_aggregate(mtuple, cg_map, threshold)) {
			// insert this embedding into the task queue
			out_queue.push_back(tuple);
		}
	}
	bool filter_aggregate(MTuple & in_tuple, CgMap& map, int threshold){
		// find the quick pattern of this embedding
		Quick_Pattern quick_pattern(in_tuple.get_size() * sizeof(Element_In_Tuple));
		Pattern::turn_quick_pattern_pure(in_tuple, quick_pattern, label_flag);
		// find the pattern (canonical graph) of this embedding
		Canonical_Graph* cf = Pattern::turn_canonical_graph(quick_pattern, false);
		quick_pattern.clean();
		// compare the count of this pattern with the threshold
		// TODO: this is not the correct support counting for FSM
		assert(map.find(*cf) != map.end());
		bool r = (map[*cf] < threshold);
		delete cf;
		return r;
	}
	void update_tuple_size() { sizeof_tuple += sizeof(Element_In_Tuple); }
	inline int get_sizeof_tuple() { return sizeof_tuple; }
	void printout_tuple(int level, Embedding tuple) {
		MTuple in_tuple(sizeof_tuple);
		in_tuple.init(tuple);
		std::cout << in_tuple << std::endl;
	}
	void printout_agg() {
		for (auto it = canonical_graphs_agg.begin(); it != canonical_graphs_agg.end(); ++it)
			std::cout << "{" << it->first << " --> " << it->second << std::endl;
	}
	void printout_agg(CgMap cg_map) {
		for (auto it = cg_map.begin(); it != cg_map.end(); ++it)
			std::cout << "{" << it->first << " --> " << it->second << std::endl;
	}

private:
	bool label_flag;
	int sizeof_tuple;
	std::vector<Embedding> edge_hashmap;
	QpMap quick_patterns_aggregation;
	CgMap canonical_graphs_agg;
	void build_edge_hashmap(int n_edges, int start_vertex, std::vector<LabeledEdge> edge_list) {
		for(int pos = 0; pos < n_edges; pos ++) {
			LabeledEdge e = edge_list[pos];
			edge_hashmap[e.src - start_vertex].push_back(Element_In_Tuple(e.target, (BYTE)0, e.target_label));
		}
	}
	bool gen_an_out_tuple(MTuple_join & in_tuple, Element_In_Tuple & element, BYTE history, std::unordered_set<VertexId>& vertices_set) {
		bool vertex_existed = true;
		auto num_vertices = vertices_set.size();
		if(vertices_set.find(element.vertex_id) == vertices_set.end()){
			num_vertices += 1;
			vertex_existed = false;
		}
		in_tuple.push(&element);
		in_tuple.set_num_vertices(num_vertices);
		return vertex_existed;
	}
	void aggregate_canonical() {
		// for each quick patter
		for (auto it = quick_patterns_aggregation.begin(); it != quick_patterns_aggregation.end(); ++it) {
			Quick_Pattern sub_graph = it->first;
			int s = it->second;
			// turn it into a canonical graph (i.e. real pattern)
			Canonical_Graph* cg = Pattern::turn_canonical_graph(sub_graph, false);
			sub_graph.clean();
			// if this pattern already exists, increase its count
			if (canonical_graphs_agg.find(*cg) != canonical_graphs_agg.end()) {
				canonical_graphs_agg[*cg] = canonical_graphs_agg[*cg] + s;
			} else {
				// otherwise add this pattern into the map, and set the count as 's'
				canonical_graphs_agg[*cg] = s;
			}
			delete cg;
		}
	}
};

#endif /* MINER_HPP_ */
