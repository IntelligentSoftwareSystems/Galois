#ifndef EDGE_MINER_H_
#define EDGE_MINER_H_
#include "miner.h"
#include "egonet.h"
#include "dfscode.h"
#include "edgelist.h"
#include "embedding_list.h"
#include "domain_support.h"

//typedef galois::InsertBag<DFS> InitPatternQueue;
typedef galois::gstl::Vector<DFS> InitPatternQueue;
typedef std::vector<Edge> InitEmbeddingList;
typedef std::map<InitPattern, InitEmbeddingList> InitEmbeddingLists;

struct Status {
	//int thread_id;
	//unsigned task_split_level;
	//unsigned embeddings_regeneration_level;
	unsigned current_dfs_level;
	int frequent_patterns_count;
	//bool is_running;
	DFSCode DFS_CODE;
	//DFSCode DFS_CODE_IS_MIN;
	//CGraph GRAPH_IS_MIN;
	std::vector<std::deque<DFS> > dfs_task_queue;
	//std::deque<DFSCode> dfscodes_to_process;
};
typedef galois::substrate::PerThreadStorage<Status> MtStatus; // Multi-threaded Status

class EdgeMiner : public Miner {
public:
	EdgeMiner(Graph *g, unsigned size) {
		graph = g;
		threshold = minsup;
		max_size = size;
		//for (int i = 0; i < numThreads; i++)
		//	emb_lists.getLocal(i)->allocate(0, size);
		edge_list.init(*g);
		total_num.reset();
	}
	virtual ~EdgeMiner() {}
	virtual bool toExtend(unsigned n, const EdgeEmbedding &emb, unsigned pos) {
		//return emb.get_key(pos) == 0;
		return true;
	}
	virtual bool toAdd(unsigned n, const EdgeEmbedding &emb, VertexId src, VertexId dst, unsigned pos, BYTE &existed, VertexSet vertex_set) {
		//return !is_edge_automorphism(n, emb, pos, src, dst, existed, vertex_set);
		return true;
	}
	virtual unsigned getPattern(unsigned n, unsigned i, VertexId dst, const EdgeEmbedding &emb, unsigned pos) {
		return 0;
	}
	virtual void reduction(unsigned pid) {
		//total_num += 1;
	}
	virtual void print_output() { }

	void init_dfs() {
		//int single_edge_patterns = 0;
		//int num_embeddings = 0;
		galois::do_all(galois::iterate((size_t)0, edge_list.size()),
			[&](const size_t& pos) {
				InitMap *lmap = init_pattern_maps.getLocal();
				Edge edge = edge_list.get_edge(pos);
				auto& src_label = graph->getData(edge.src);
				auto& dst_label = graph->getData(edge.dst);
				if (src_label <= dst_label) {
					InitPattern key = get_init_pattern(src_label, dst_label);
					if (lmap->find(key) == lmap->end()) {
						(*lmap)[key] = new DomainSupport(2);
						(*lmap)[key]->set_threshold(threshold);
					}
					(*lmap)[key]->add_vertex(0, edge.src);
					(*lmap)[key]->add_vertex(1, edge.dst);
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

		// classify all the single-edge embeddings
		for (auto edge : edge_list) {
			auto& src_label = graph->getData(edge.src);
			auto& dst_label = graph->getData(edge.dst);
			if (src_label <= dst_label) {
				InitPattern key = get_init_pattern(src_label, dst_label);
				if (init_map[key]->get_support())
					init_emb_lists[src_label][dst_label].push(2, &edge, 0); // TODO: do not consider edge label for now
			}
		}
		// check each pattern-support pair, if the pattern is frequent, put it into the worklist
		for (auto ps_pair : init_map) {
			// if the pattern is frequent, add it to the pattern queue
			if (ps_pair.second->get_support())  {
				auto src_label = ps_pair.first.first;
				auto dst_label = ps_pair.first.second;
				DFS dfs(0, 1, src_label, 0, dst_label);
				init_freq_pattern_queue.push_back(dfs);
			}
		}
	}
	void process() {
		init_dfs();
		galois::do_all(galois::iterate(init_freq_pattern_queue),
			[&](const DFS& dfs) {
				Status *local_status = mt_status.getLocal();
				local_status->DFS_CODE.push(0, 1, dfs.fromlabel, dfs.elabel, dfs.tolabel);
				//InitPattern key = get_init_pattern(dfs.fromlabel, dfs.tolabel);
				dfs_extend_naive(1, init_emb_lists[dfs.fromlabel][dfs.tolabel], *local_status);
				local_status->DFS_CODE.pop();
			},
			galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::no_conflicts(),
			galois::loopname("InitFilter")
		);
	}
	void dfs_extend_naive(unsigned level, LabEdgeEmbeddingList &emb_list, Status &status) {
		if (is_min(status) == false) return; // check if this pattern is canonical: minimal DFSCode
		total_num += 1;
		if (level == max_size) return;
		unsigned n = level + 1;
		const RMPath &rmpath = status.DFS_CODE.buildRMPath(); // build the right-most path of this pattern
		auto minlabel = status.DFS_CODE[0].fromlabel; 
		auto maxtoc = status.DFS_CODE[rmpath[0]].to; // right-most vertex
	
		//EmbeddingLists3D emb_lists_3d;
		EmbeddingLists2D emb_lists_fwd;
		EmbeddingLists1D emb_lists_bck;
		for (size_t emb_id = 0; emb_id < emb_list.size(); ++ emb_id) {
			//EdgeEmbedding emb(n);
			//emb_list.get_embedding<EdgeEmbedding>(level, emb_id, emb);
			//for (unsigned element_id = 0; element_id < n; ++ element_id) {
			//	VertexId src = emb.get_vertex(element_id);
			//	if (!toExtend(n, emb, element_id)) continue;
			//	for (auto e : graph->edges(src)) {
			//		GNode dst = graph->getEdgeDst(e);
			//		if (toAdd(n, emb, src, dst, element_id, existed, vert_set)) {
			//			auto dst_label = 0, edge_label = 0;
			//			dst_label = graph->getData(dst);
			//		}
			//	}
			//}
			LabEdgeEmbedding *cur = &emb_list[n];
			unsigned emb_size = cur->num_vertices;
			History history(cur);
			auto e2 = history[rmpath[0]];
			// backward extension
			for (size_t i = rmpath.size() - 1; i >= 1; --i) {
				auto e1 = history[rmpath[i]];
				if(e1 == e2) continue;
				for (auto e : graph->edges(e2->src)) {
					//auto dst = graph->getEdgeDst(e);
					//auto elabel = graph->getEdgeData(e);
					if (history.hasEdge(*e)) continue;
					if (graph->getData(e1->dst) <= graph->getData(e2->src)) {
						auto edge = edge_list.get_edge(*e);
						emb_lists_bck[status.DFS_CODE[rmpath[i]].from].push(emb_size, &edge, cur);
						break;
					}
				}
			}
			// pure forward extension
			for (auto e : graph->edges(e2->dst)) {
				auto dst = graph->getEdgeDst(e);
				auto& dst_label = graph->getData(dst);
				if(minlabel > dst_label || history.hasVertex(dst)) continue;
				auto edge = edge_list.get_edge(*e);
				emb_lists_fwd[maxtoc][graph->getData(edge.dst)].push(emb_size+1, &edge, cur);
			}
			// backtracked forward extension
			for (size_t i = 0; i < rmpath.size(); ++i) {
				//if (get_forward_rmpath(*graph, edge_list, history[rmpath[i]], minlabel, history, edges)) {
				auto e1 = history[rmpath[i]];
				auto tolabel = graph->getData(e1->dst);
				for (auto e : graph->edges(e1->src)) {
					auto dst = graph->getEdgeDst(e);
					//auto elabel = graph->getEdgeData(e);
					auto& dst_label = graph->getData(dst);
					if (e1->dst == dst || minlabel > dst_label || history.hasVertex(dst)) continue;
					if (tolabel <= dst_label) {
						auto edge = edge_list.get_edge(*e);
						emb_lists_fwd[status.DFS_CODE[rmpath[i]].from][graph->getData(edge.dst)].push(emb_size+1, &edge, cur);
					}
				}
			}
		}
		std::deque<DFS> tmp;
		if(status.dfs_task_queue.size() <= level) {
			status.dfs_task_queue.push_back(tmp);
		}
		while(status.dfs_task_queue[level].size() > 0) {
			DFS dfs = status.dfs_task_queue[level].front(); // take a pattern from the task queue
			status.dfs_task_queue[level].pop_front();
			status.current_dfs_level = level; // ready to go to the next level
			status.DFS_CODE.push(dfs.from, dfs.to, dfs.fromlabel, dfs.elabel, dfs.tolabel); // update the pattern
			dfs_extend_naive(level+1, emb_list, status);
		}
	}
	Ulong get_total_count() { return total_num.reduce(); }

protected:
	UlongAccu total_num;
	unsigned threshold;
	InitMap init_map;
	InitMaps init_pattern_maps; // initial pattern map, only used once, no need to clear
	InitPatternQueue init_freq_pattern_queue;
	EdgeList edge_list;
	EmbeddingLists2D init_emb_lists;
	MtStatus mt_status;
	bool is_min(Status &status) {
		if(status.DFS_CODE.size() == 1) return true;
		return false;
	}
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
