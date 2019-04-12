#ifndef MINER_HPP_
#define MINER_HPP_
#include "pattern.h"
//#include "mining_tuple.h"
#include "quick_pattern.h"
#include "galois/Bag.h"
#include "galois/Galois.h"
#include "galois/substrate/PerThreadStorage.h"

typedef unsigned SimpleElement;
typedef std::vector<Element_In_Tuple> Embedding;
//typedef std::vector<SimpleElement> BaseEmbedding;
class BaseEmbedding: public std::vector<SimpleElement> {
public:
	inline unsigned get_hash() const {
		bliss::UintSeqHash h;
		for(unsigned i = 0; i < size(); ++i)
			h.update(data()[i]);
		return h.get_value();
	}
};
typedef std::unordered_map<Quick_Pattern, unsigned> QpMap;
typedef std::unordered_map<Canonical_Graph, unsigned> CgMap;
typedef std::unordered_map<BaseEmbedding, unsigned> SimpleMap;
typedef galois::substrate::PerThreadStorage<QpMap> LocalQpMap;
typedef galois::substrate::PerThreadStorage<CgMap> LocalCgMap;
typedef galois::substrate::PerThreadStorage<SimpleMap> LocalSimpleMap;
typedef galois::InsertBag<Embedding> EmbeddingQueue;
typedef galois::InsertBag<BaseEmbedding> BaseEmbeddingQueue;

namespace std {
	template<>
	struct hash<BaseEmbedding> {
		std::size_t operator()(const BaseEmbedding& emb) const {
			return std::hash<int>()(emb.get_hash());
		}
	};
}

class Miner {
public:
	Miner(bool label_f, int tuple_size, Graph *g, std::vector<LabeledEdge> edge_list) : label_flag(label_f), sizeof_tuple(tuple_size) {
		graph = g;
		//edge_hashmap.resize(g->size());
		//build_edge_hashmap(g->sizeEdges(), 0, edge_list);
	}
	virtual ~Miner() {};
	// given an embedding, extend it with one more edge, and if it is not automorphism, insert the new embedding into the task queue
	void extend_edge(unsigned max_size, Embedding emb, EmbeddingQueue &queue) {
		unsigned size = emb.size();
		std::unordered_set<VertexId> vertices_set;
		vertices_set.reserve(size);
		for(unsigned i = 0; i < size; i ++) vertices_set.insert(emb[i].vertex_id);
		std::unordered_set<VertexId> set;
		// for each vertex in the embedding
		for(unsigned i = 0; i < size; ++i) {
			VertexId id = emb[i].vertex_id;
			assert(id >= 0 && id < graph->size());
			if(set.find(id) == set.end()) {
				set.insert(id);
				// try edge extension
				for(auto e : graph->edges(id)) {
					GNode dst = graph->getEdgeDst(e);
					auto dst_label = 0, edge_label = 0;
					if (label_flag) dst_label = graph->getData(dst);
					//edge_label = graph->getEdgeData(e);
					auto num_vertices = vertices_set.size();
					bool vertex_existed = true;
					if(vertices_set.find(dst) == vertices_set.end()) {
						num_vertices ++;
						vertex_existed = false;
					}
					// number of vertices must be smaller than k.
					// check if this is automorphism
					if(num_vertices <= max_size && !is_automorphism(emb, i, id, dst, vertex_existed)) {
						Element_In_Tuple new_element(dst, (BYTE)num_vertices, edge_label, dst_label, (BYTE)i);
						// insert the new extended embedding into the queue
						emb.push_back(new_element);
						queue.push_back(emb);
						emb.pop_back();
					}
				}
			}
		}
	}
	void extend_clique(BaseEmbedding emb, BaseEmbeddingQueue &queue) {
		int n = emb.size();
		for(int i = 0; i < n; ++i) {
			int id = emb[i];
			for(auto e : graph->edges(id)) {
				GNode dst = graph->getEdgeDst(e);
				BaseEmbedding out_emb = emb;
				if(dst > emb[n-1]) {
					out_emb.push_back(dst);
					queue.push_back(out_emb);
				}
			}
		}
	}
	void aggregate(EmbeddingQueue queue) {
		quick_patterns_agg.clear();
		for (auto emb : queue) {
			Quick_Pattern qp(sizeof_tuple);
			turn_quick_pattern_pure(emb, qp, label_flag);
			// update counting for this quick pattern
			if (quick_patterns_agg.find(qp) != quick_patterns_agg.end()) {
				// if this quick pattern already exists, increase its count
				quick_patterns_agg[qp] += 1;
				qp.clean();
				// otherwise add this quick pattern into the map, and set the count as one
			} else { quick_patterns_agg[qp] = 1; }
		}
		canonical_graphs_agg.clear();
		// for each quick patter
		for (auto it = quick_patterns_agg.begin(); it != quick_patterns_agg.end(); ++it) {
			Quick_Pattern qp = it->first;
			unsigned s = it->second;
			// turn it into a canonical graph (i.e. real pattern)
			Canonical_Graph* cg = Pattern::turn_canonical_graph(qp, false);
			qp.clean();
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
	void canonical_aggregate(QpMap qp_map) {
		canonical_graphs_agg.clear();
		for (auto it = qp_map.begin(); it != qp_map.end(); ++it) {
			Quick_Pattern qp = it->first;
			unsigned s = it->second;
			Canonical_Graph* cg = Pattern::turn_canonical_graph(qp, false);
			qp.clean();
			if (canonical_graphs_agg.find(*cg) != canonical_graphs_agg.end()) {
				canonical_graphs_agg[*cg] = canonical_graphs_agg[*cg] + s;
			} else {
				canonical_graphs_agg[*cg] = s;
			}
			delete cg;
		}
	}
	void aggregate_clique(BaseEmbeddingQueue &in_queue, BaseEmbeddingQueue &out_queue) {
		SimpleMap simple_agg;
		for (auto emb : in_queue) {
			auto it = simple_agg.find(emb);
			if(it != simple_agg.end()) {
				if(it->second == it->first.size() - 2) {
					out_queue.push_back(emb);
					simple_agg.erase(it);
				}
				else simple_agg[emb] += 1;
			}
			else simple_agg[emb] = 1;
		}
	}
	void aggregate_each_clique(BaseEmbedding &emb, SimpleMap& sm) {
		auto it = sm.find(emb);
		if(it != sm.end()) {
			sm[emb] += 1;
		}
		else sm[emb] = 1;
	}
	void quick_aggregate_each(const Embedding& emb, QpMap& qp_map) {
		Quick_Pattern qp(sizeof_tuple);
		turn_quick_pattern_pure(emb, qp, label_flag);
		if (qp_map.find(qp) != qp_map.end()) {
			qp_map[qp] += 1;
			qp.clean();
		} else {
			qp_map[qp] = 1;
		}
	}
	void canonical_aggregate_each(Quick_Pattern qp, int num, CgMap& canonical_graphs_agg) {
		Canonical_Graph* cg = Pattern::turn_canonical_graph(qp, false);
		qp.clean();
		if (canonical_graphs_agg.find(*cg) != canonical_graphs_agg.end()) {
			canonical_graphs_agg[*cg] = canonical_graphs_agg[*cg] + num;
		} else canonical_graphs_agg[*cg] = num;
		delete cg;
	}
	void filter_all(EmbeddingQueue &in_queue, EmbeddingQueue &out_queue, CgMap &cg_map) {
		for (auto emb : in_queue) {
			Quick_Pattern qp(sizeof_tuple);
			turn_quick_pattern_pure(emb, qp, label_flag);
			Canonical_Graph* cf = Pattern::turn_canonical_graph(qp, false);
			qp.clean();
			assert(cg_map.find(*cf) != cg_map.end());
			if(cg_map[*cf] >= threshold) {
				out_queue.push_back(emb);
			}
			delete cf;
		}
	}

	// filtering for FSM
	// check if the pattern of this embedding is frequent
	void filter_each(Embedding &emb, EmbeddingQueue &out_queue, CgMap &cg_map) {
		// find the quick pattern of this embedding
		Quick_Pattern qp(sizeof_tuple);
		turn_quick_pattern_pure(emb, qp, label_flag);
		// find the pattern (canonical graph) of this embedding
		Canonical_Graph* cf = Pattern::turn_canonical_graph(qp, false);
		qp.clean();
		// compare the count of this pattern with the threshold
		// TODO: this is not the correct support counting for FSM
		assert(cg_map.find(*cf) != cg_map.end());
		if (cg_map[*cf] >= threshold)
			// insert this embedding into the task queue
			out_queue.push_back(emb);
		delete cf;
	}
	void set_threshold(unsigned minsup) { threshold = minsup; }
	void update_embedding_size() { sizeof_tuple += sizeof(Element_In_Tuple); }
	void update_base_embedding_size() { sizeof_tuple += sizeof(SimpleElement); }
	inline int get_sizeof_embedding() { return sizeof_tuple; }
	void printout_embedding(int level, Embedding emb) {
		if(emb.size() == 0) {
			std::cout << "(empty)";
			return;
		}
		std::cout << "(";
		for(unsigned index = 0; index < emb.size() - 1; ++index)
			std::cout << emb[index] << ", ";
		std::cout << emb[emb.size() - 1];
		std::cout << ")\n";
	}
	void printout_embedding(int level, BaseEmbedding emb) {
		if(emb.size() == 0){
			std::cout << "(empty)";
			return;
		}
		std::cout << "(";
		for(unsigned index = 0; index < emb.size() - 1; ++index)
			std::cout << emb[index] << ", ";
		std::cout << emb[emb.size() - 1];
		std::cout << ")\n";
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
	unsigned threshold;
	Graph *graph;
	QpMap quick_patterns_agg;
	CgMap canonical_graphs_agg;
#if 0
	std::vector<Embedding> edge_hashmap;
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
	void gen_an_out_tuple(MTuple_join_simple & in_tuple, Base_Element & element) {
		in_tuple.push(&element);
	}
	void turn_quick_pattern_pure(MTuple & sub_graph, Quick_Pattern & qp, bool label_flag) {
		std::memcpy(qp.get_elements(), sub_graph.get_elements(), qp.get_size() * sizeof(Element_In_Tuple));
		std::unordered_map<VertexId, VertexId> map;
		VertexId new_id = 1;
		for(unsigned i = 0; i < qp.get_size(); i++) {
			Element_In_Tuple& element = qp.at(i);
			if(!label_flag) element.vertex_label = (BYTE)0;
			VertexId old_id = element.vertex_id;
			auto iterator = map.find(old_id);
			if(iterator == map.end()) {
				element.set_vertex_id(new_id);
				map[old_id] = new_id++;
			} else element.set_vertex_id(iterator->second);
		}
	}
#endif

	void turn_quick_pattern_pure(const Embedding & sub_graph, Quick_Pattern & qp, bool label_flag) {
		std::memcpy(qp.get_elements(), sub_graph.data(), qp.get_size() * sizeof(Element_In_Tuple));
		std::unordered_map<VertexId, VertexId> map;
		VertexId new_id = 1;
		for(unsigned i = 0; i < qp.get_size(); i++) {
			Element_In_Tuple& element = qp.at(i);
			if(!label_flag) element.vertex_label = (BYTE)0;
			VertexId old_id = element.vertex_id;
			auto iterator = map.find(old_id);
			if(iterator == map.end()) {
				element.set_vertex_id(new_id);
				map[old_id] = new_id++;
			} else element.set_vertex_id(iterator->second);
		}
	}


	inline bool is_automorphism(Embedding & sub_graph, BYTE history, VertexId src, VertexId dst, const bool vertex_existed) {
		//check with the first element
		if(dst < sub_graph.front().vertex_id) return true;
		//check loop edge
		if(dst == sub_graph[sub_graph[history].history_info].vertex_id) return true;
		//check to see if there already exists the vertex added; if so, just allow to add edge which is (smaller id -> bigger id)
		if(vertex_existed && src > dst) return true;
		std::pair<VertexId, VertexId> added_edge(src, dst);
		for(unsigned index = history + 1; index < sub_graph.size(); ++index) {
			std::pair<VertexId, VertexId> edge;
			getEdge(sub_graph, index, edge);
			int cmp = compare(added_edge, edge);
			if(cmp <= 0) return true;
		}
		return false;
	}
	inline void getEdge(Embedding & sub_graph, unsigned index, std::pair<VertexId, VertexId>& edge) {
		Element_In_Tuple tuple = sub_graph[index];
		edge.first = sub_graph[tuple.history_info].vertex_id;
		edge.second = tuple.vertex_id;
		assert(edge.first != edge.second);
	}
	inline void swap(std::pair<VertexId, VertexId>& pair) {
		if (pair.first > pair.second) {
			VertexId tmp = pair.first;
			pair.first = pair.second;
			pair.second = tmp;
		}
	}
	inline int compare(std::pair<VertexId, VertexId>& oneEdge, std::pair<VertexId, VertexId>& otherEdge) {
		swap(oneEdge);
		swap(otherEdge);
		if(oneEdge.first == otherEdge.first) return oneEdge.second - otherEdge.second;
		else return oneEdge.first - otherEdge.first;
	}
};

// print out the embeddings in the task queue
#ifdef USE_SIMPLE
void printout_embeddings(int level, Miner& miner, BaseEmbeddingQueue& queue) {
#else
void printout_embeddings(int level, Miner& miner, EmbeddingQueue& queue) {
#endif
	int num_embeddings = std::distance(queue.begin(), queue.end());
	unsigned embedding_size = miner.get_sizeof_embedding();
	std::cout << "Number of embeddings in level " << level << ": " << num_embeddings << " (embedding_size=" << embedding_size << ")" << std::endl;
	for (auto embedding : queue)
		miner.printout_embedding(level, embedding);
}

#endif /* MINER_HPP_ */
