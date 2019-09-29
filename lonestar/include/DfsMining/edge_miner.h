#ifndef EDGE_MINER_H_
#define EDGE_MINER_H_
#include "miner.h"
#include "egonet.h"
#include "dfscode.h"
#include "edgelist.h"
#include "embedding_list.h"
#include "domain_support.h"

class EdgeMiner : public Miner {
public:
	EdgeMiner(Graph *g, unsigned size) {
		graph = g;
		threshold = minsup;
		max_size = size;
		for (int i = 0; i < numThreads; i++)
			emb_lists.getLocal(i)->allocate(0, size);
		edge_list.init(*g);
	}
	virtual ~EdgeMiner() {}
	virtual bool toExtend(unsigned n, const EdgeEmbedding &emb, unsigned pos) {
		return emb.get_key(pos) == 0;
	}
	virtual bool toAdd(unsigned n, const EdgeEmbedding &emb, VertexId src, VertexId dst, unsigned pos, BYTE &existed, VertexSet vertex_set) {
		return !is_edge_automorphism(n, emb, pos, src, dst, existed, vertex_set);
	}
	virtual unsigned getPattern(unsigned n, unsigned i, VertexId dst, const EdgeEmbedding &emb, unsigned pos) {
		return 0;
	}
	virtual void reduction(unsigned pid) {
		//total_num += 1;
	}
	virtual void print_output() { }

	void init_dfs() {
		int single_edge_patterns = 0;
		int num_embeddings = 0;
		galois::do_all(galois::iterate(graph->begin(), graph->end()),
			[&](const GNode& src) {
				InitMap *lmap = init_pattern_maps.getLocal();
				auto& src_label = graph->getData(src);
				for (auto e : graph->edges(src)) {
					GNode dst = graph->getEdgeDst(e);
					auto& dst_label = graph->getData(dst);
					if (src_label <= dst_label) {
						InitPattern key = get_init_pattern(src_label, dst_label);
						if (lmap->find(key) == lmap->end()) {
							(*lmap)[key] = new DomainSupport(2);
							(*lmap)[key]->set_threshold(threshold);
						}
						(*lmap)[key]->add_vertex(0, src);
						(*lmap)[key]->add_vertex(1, dst);
					}
				}
			},
			galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::no_conflicts(),
			galois::loopname("InitReduction")
		);
		init_map = *(init_pattern_maps.getLocal(0));
		for (auto i = 1; i < numThreads; i++) {
			for (auto element : *init_pattern_maps.getLocal(i)) {
				DomainSupport *support = element.second;
				if (init_map.find(element.first) == init_map.end()) {
					init_map[element.first] = support;
				} else {
					for (unsigned i = 0; i < 2; i ++) {
						if (!init_map[element.first]->has_domain_reached_support(i)) {
							if (support->has_domain_reached_support(i))
								init_map[element.first]->set_domain_frequent(i);
							else init_map[element.first]->add_vertices(i, support->domain_sets[i]);
						}
					}
				}
			}
		}
		std::cout << "Number of single-edge patterns: " << init_map.size() << "\n";
	}
	void process() {
		init_dfs();
		galois::do_all(galois::iterate((size_t)0, edge_list.size()),
			[&](const size_t& pos) {
				EmbeddingList *emb_list = emb_lists.getLocal();
				Edge edge = edge_list.get_edge(pos);
				auto& src_label = graph->getData(edge.src);
				auto& dst_label = graph->getData(edge.dst);
				InitPattern key = get_init_pattern(src_label, dst_label);
				if (init_map[key]->get_support()) {
					emb_list->init(edge);
					dfs_extend_naive(1, 0, *emb_list);
				}
			},
			galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::no_conflicts(),
			galois::loopname("InitFilter")
		);
	}
	void dfs_extend_naive(unsigned level, unsigned pos, EmbeddingList &emb_list) {
		if (level == max_size) return;
		unsigned n = level + 1;
		//BYTE existed = 0;
		//VertexSet vert_set;
		//if (n > 3) for (unsigned i = 0; i < n; i ++) vert_set.insert(emb.get_vertex(i));
		for (size_t emb_id = 0; emb_id < emb_list.size(); ++ emb_id) {
			EdgeEmbedding emb(n);
			emb_list.get_embedding<EdgeEmbedding>(level, emb_id, emb);
			for (unsigned element_id = 0; element_id < n; ++ element_id) {
				VertexId src = emb.get_vertex(element_id);
				//if (!toExtend(n, emb, element_id)) continue;
				for (auto e : graph->edges(src)) {
					GNode dst = graph->getEdgeDst(e);
					//if (toAdd(n, emb, src, dst, element_id, existed, vert_set)) {
						auto dst_label = 0, edge_label = 0;
						unsigned start = emb_list.size(level+1);
						dst_label = graph->getData(dst);
						dfs_extend_naive(level+1, start, emb_list);
					//}
				}
			}
		}
		//reduction(getPattern(n, element_id, dst, emb, previous_pid));
	}

protected:
	unsigned threshold;
	InitMap init_map;
	InitMaps init_pattern_maps; // initialization map, only used for once, no need to clear
	EmbeddingLists emb_lists;
	EdgeList edge_list;
	inline InitPattern get_init_pattern(BYTE src_label, BYTE dst_label) {
		if (src_label <= dst_label) return std::make_pair(src_label, dst_label);
		else return std::make_pair(dst_label, src_label);
	}
	bool is_quick_automorphism(unsigned size, const EdgeEmbedding& emb, BYTE history, VertexId src, VertexId dst, BYTE& existed) {
		if (dst <= emb.get_vertex(0)) return true;
		if (dst == emb.get_vertex(1)) return true;
		if (history == 0 && dst < emb.get_vertex(1)) return true;
		if (size == 2) {
		} else if (size == 3) {
			if (history == 0 && emb.get_history(2) == 0 && dst <= emb.get_vertex(2)) return true;
			if (history == 0 && emb.get_history(2) == 1 && dst == emb.get_vertex(2)) return true;
			if (history == 1 && emb.get_history(2) == 1 && dst <= emb.get_vertex(2)) return true;
			if (dst == emb.get_vertex(2)) existed = 1;
		} else {
			std::cout << "Error: should go to detailed check\n";
		}
		return false;
	}
	bool is_edge_automorphism(unsigned size, const EdgeEmbedding& emb, BYTE history, VertexId src, VertexId dst, BYTE& existed, const VertexSet& vertex_set) {
		if (size < 3) return is_quick_automorphism(size, emb, history, src, dst, existed);
		if (dst <= emb.get_vertex(0)) return true;
		if (history == 0 && dst <= emb.get_vertex(1)) return true;
		if (dst == emb.get_vertex(emb.get_history(history))) return true;
		if (vertex_set.find(dst) != vertex_set.end()) existed = 1;
		if (existed && src > dst) return true;
		std::pair<VertexId, VertexId> added_edge(src, dst);
		for (unsigned index = history + 1; index < emb.size(); ++index) {
			std::pair<VertexId, VertexId> edge;
			edge.first = emb.get_vertex(emb.get_history(index));
			edge.second = emb.get_vertex(index);
			int cmp = compare(added_edge, edge);
			if(cmp <= 0) return true;
		}
		return false;
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

#endif
