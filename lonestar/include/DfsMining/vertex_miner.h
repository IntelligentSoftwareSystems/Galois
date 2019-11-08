#ifndef MINER_H_
#define MINER_H_
#include "miner.h"
#include "edgelist.h"
#include "embedding_list.h"

class VertexMiner : public Miner {
public:
	VertexMiner(Graph *g) {
		graph = g;
		degree_counting();
		// connected k=3 motifs
		total_3_tris = 0;
		total_3_path = 0;
		// connected k=4 motifs
		total_4_clique = 0;
		total_4_diamond = 0;
		total_4_tailed_tris = 0;
		total_4_cycle = 0;
		total_3_star = 0;
		total_4_path = 0;
	}
	void init_edgelist(bool symmetrize = false) {
		edge_list.init(*graph, is_directed, symmetrize);
		core = edge_list.get_core();
	}
	virtual ~VertexMiner() {}
	void set_max_size(unsigned size = 3) { max_size = size; }
	void set_max_degree(unsigned d = 1) { max_degree = d; }
	void set_directed(bool directed = true) { is_directed = directed; }
	void set_num_patterns(int np = 1) {
		npatterns = np;
		if (npatterns == 1) total_num.reset();
		else {
			accumulators.resize(npatterns);
			for (int i = 0; i < npatterns; i++) accumulators[i].reset();
			//std::cout << max_size << "-motif has " << npatterns << " patterns in total\n";
		}
	}
	// Pangolin APIs
	// toExtend
	virtual bool toExtend(unsigned n, const BaseEmbedding &emb, unsigned pos) { return true; }
	virtual bool toExtend(unsigned level, unsigned pos) { return true; }

	// toAdd (only add non-automorphisms)
	virtual bool toAdd(unsigned n, const BaseEmbedding &emb, VertexId dst, unsigned pos) { return true; }
	virtual bool toAdd(unsigned level, unsigned label) { return label == level; }
	virtual bool toAdd(unsigned level, VertexId vid, const EmbeddingList &emb_list) {
		return !is_vertexInduced_automorphism(level, vid, emb_list);
	}
	virtual unsigned getPattern(unsigned n, unsigned i, VertexId dst, const VertexEmbedding &emb, unsigned pos) {
		return 0;
	}
	virtual unsigned getPattern(unsigned level, VertexId dst, EmbeddingList &emb_list, unsigned previous_pid) {
		return 0;
	}
	virtual void reduction(unsigned pid) {
		#ifdef USE_MAP
		accumulators[pid] += 1;
		#else
		total_num += 1;
		#endif
	}
	virtual void reduction(unsigned pid, EmbeddingList &emb_list) {
	}
	virtual void reduction(unsigned level, EmbeddingList &emb_list, VertexId src, VertexId dst, unsigned previous_pid) {
	}
	virtual void update(unsigned level, unsigned vid, unsigned pos, unsigned previous_pid, EmbeddingList &emb_list) { }
	virtual void print_output() { }
	virtual void post_processing(unsigned level) { }
	virtual void edge_process_opt() { }

	void vertex_process() {
		//galois::do_all(galois::iterate((size_t)0, (size_t)numThreads), [&](const size_t& tid) {
		for (int i = 0; i < numThreads; i++) {
			emb_lists.getLocal(i)->allocate(graph, max_size, max_degree);
		}
		std::cout << "DFS vertex processing without advanced optimization\n";
		galois::do_all(galois::iterate(graph->begin(), graph->end()), [&](const auto& vid) {
			//std::cout << "debug: vid = " << vid << "\n";
			EmbeddingList *emb_list = emb_lists.getLocal();
			emb_list->init_vertex(vid);
			#ifdef USE_MAP
			//dfs_extend_multi(1, 0, *emb_list);
			ego_extend_multi(1, *emb_list);
			#else
			//dfs_extend_single(1, 0, *emb_list);
			ego_extend_single(1, *emb_list);
			emb_list->clear_labels(vid);
			#endif
		}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("DfsEdgeNaiveSolver"));
		#ifdef USE_MAP
		motif_count();
		#endif
	}

	void edge_process() {
		//galois::do_all(galois::iterate((size_t)0, (size_t)numThreads), [&](const size_t& tid) {
		for (int i = 0; i < numThreads; i++) {
			emb_lists.getLocal(i)->allocate(graph, max_size, max_degree);
		}
		std::cout << "DFS edge processing without advanced optimization\n";
		std::cout << "Number of single-edge embeddings: " << edge_list.size() << "\n";
		galois::do_all(galois::iterate((size_t)0, edge_list.size()), [&](const size_t& pos) {
			EmbeddingList *emb_list = emb_lists.getLocal();
			emb_list->init_edge(edge_list.get_edge(pos));
			#ifdef USE_MAP
			//dfs_extend_multi(1, 0, *emb_list);
			ego_extend_opt(1, *emb_list);
			solve_motif_equations(*emb_list);
			if (max_size == 4) emb_list->clear_labels(edge_list.get_edge(pos).dst);
			#else
			//dfs_extend_single(1, 0, *emb_list);
			ego_extend_single(1, *emb_list);
			#endif
			emb_list->clear_labels(edge_list.get_edge(pos).src);
		}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("DfsEdgeNaiveSolver"));
		#ifdef USE_MAP
		motif_count();
		#endif
	}

	void vertex_process_opt() {
		for (int i = 0; i < numThreads; i++) {
			emb_lists.getLocal(i)->allocate(graph, max_size, max_degree);
		}
		std::cout << "DFS vertex processing using advanced optimization\n";
		//galois::do_all(galois::iterate(*graph), [&](const auto& vid) {
		galois::for_each(galois::iterate(graph->begin(), graph->end()), [&](const auto &vid, auto &ctx) {
			EmbeddingList *emb_list = emb_lists.getLocal();
			emb_list->init_vertex(vid);
			#ifdef USE_MAP
			#else
			dfs_extend(1, *emb_list);
			#endif
		}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("DfsVertexSolver"));
		#ifdef USE_MAP
		motif_count();
		#endif
	}

	// naive DFS extension for k-cliques
	void dfs_extend_single(unsigned level, unsigned pos, EmbeddingList &emb_list) {
		unsigned n = level+1;
		BaseEmbedding emb(n);
		emb_list.get_embedding<BaseEmbedding>(level, pos, emb);
		if (level == max_size-2) {
			auto vid = emb_list.get_vertex(level, pos);
			auto begin = graph->edge_begin(vid);
			auto end = graph->edge_end(vid);
			for (auto e = begin; e != end; e ++) {
				auto dst = graph->getEdgeDst(e);
				if (toAdd(n, emb, dst, n-1))
					reduction(0);
			}
			return;
		}
		auto vid = emb_list.get_vertex(level, pos);
		auto begin = graph->edge_begin(vid);
		auto end = graph->edge_end(vid);
		emb_list.set_size(level+1, 0);
		for (auto e = begin; e < end; e ++) {
			auto dst = graph->getEdgeDst(e);
			if (toAdd(n, emb, dst, n-1)) {
				auto start = emb_list.size(level+1);
				emb_list.set_vid(level+1, start, dst);
				emb_list.set_idx(level+1, start, pos);
				emb_list.set_size(level+1, start+1);
				dfs_extend_single(level+1, start, emb_list);
			}
		}
	}
	// DFS extension for k-cliques
	void ego_extend_single(unsigned level, EmbeddingList &emb_list) {
		if (level == max_size-2) {
			for (size_t emb_id = 0; emb_id < emb_list.size(level); emb_id ++) {
				auto vid = emb_list.get_vertex(level, emb_id);
				auto begin = graph->edge_begin(vid);
				auto end = graph->edge_end(vid);
				for (auto e = begin; e < end; e ++) {
					auto dst = graph->getEdgeDst(e);
					//if (toAdd(n, emb, dst, n-1))
					if (toAdd(level, emb_list.get_label(dst)))
						reduction(0);
				}
			}
			return;
		}
		for (size_t emb_id = 0; emb_id < emb_list.size(level); emb_id ++) {
			auto vid = emb_list.get_vertex(level, emb_id);
			auto begin = graph->edge_begin(vid);
			auto end = graph->edge_end(vid);
			emb_list.set_size(level+1, 0);
			for (auto e = begin; e < end; e ++) {
				auto dst = graph->getEdgeDst(e);
				//if (toAdd(n, emb, dst, n-1)) {
				if (toAdd(level, emb_list.get_label(dst))) {
					auto start = emb_list.size(level+1);
					emb_list.set_vid(level+1, start, dst);
					emb_list.set_label(dst, level+1);
					emb_list.set_size(level+1, start+1);
				}
			}
			ego_extend_single(level+1, emb_list);
			emb_list.reset_labels(level);
		}
	}
	// DFS extension for k-cliques using graph shrinking
	void dfs_extend(unsigned level, EmbeddingList &emb_list) {
		if (level == max_size-2) {
			for (size_t emb_id = 0; emb_id < emb_list.size(level); emb_id ++) {
				auto vid = emb_list.get_vertex(level, emb_id);
				auto begin = emb_list.edge_begin(level, vid);
				auto end = emb_list.edge_end(level, vid);
				for (auto e = begin; e < end; e ++) {
					//auto dst = graph->getEdgeDst(e);
					//if (toAdd(level, emb_list.get_label(dst)))
						reduction(0);
				}
			}
			return;
		}
		for (size_t emb_id = 0; emb_id < emb_list.size(level); emb_id ++) {
			auto vid = emb_list.get_vertex(level, emb_id);
			auto begin = emb_list.edge_begin(level, vid);
			auto end = emb_list.edge_end(level, vid);
			emb_list.set_size(level+1, 0);
			for (auto e = begin; e < end; e ++) {
				auto dst = emb_list.getEdgeDst(e);
				if (toAdd(level, emb_list.get_label(dst))) {
					auto start = emb_list.size(level+1);
					emb_list.set_vid(level+1, start, dst);
					emb_list.set_label(dst, level+1);
					emb_list.set_size(level+1, start+1);
					emb_list.init_egonet_degree(level+1, dst);
				}
			}
			emb_list.update_egonet(level);
			dfs_extend(level+1, emb_list);
			emb_list.reset_labels(level);
		}
	}

	void dfs_extend_multi(unsigned level, unsigned pos, EmbeddingList &emb_list, unsigned previous_pid = 0) {
		unsigned n = level + 1;
		VertexEmbedding emb(n);
		emb_list.get_embedding<VertexEmbedding>(level, pos, emb);
		if (level == max_size-2) {
			// extending every vertex in the embedding
			for (unsigned element_id = 0; element_id < n; ++ element_id) {
				if (!toExtend(n, emb, element_id)) continue;
				auto src = emb.get_vertex(element_id);
				auto begin = graph->edge_begin(src);
				auto end = graph->edge_end(src);
				for (auto e = begin; e != end; e ++) {
					auto dst = graph->getEdgeDst(e);
					if (toAdd(n, emb, dst, element_id))
						reduction(getPattern(n, element_id, dst, emb, previous_pid));
				}
			}
			return;
		}
		// extending every vertex in the embedding
		for (unsigned element_id = 0; element_id < n; ++ element_id) {
			if (!toExtend(n, emb, element_id)) continue;
			auto src = emb.get_vertex(element_id);
			auto begin = graph->edge_begin(src);
			auto end = graph->edge_end(src);
			emb_list.set_size(level+1, 0);
			for (auto e = begin; e < end; e ++) {
				auto dst = graph->getEdgeDst(e);
				if (toAdd(n, emb, dst, element_id)) {
					auto start = emb_list.size(level+1);
					emb_list.set_vid(level+1, start, dst);
					emb_list.set_idx(level+1, start, pos);
					emb_list.set_size(level+1, start+1);
					unsigned pid = find_motif_pattern_id_dfs(n, element_id, dst, emb, start);
					dfs_extend_multi(level+1, start, emb_list, pid);
				}
			}
		}
	}

	void ego_extend_multi(unsigned level, EmbeddingList &emb_list) {
		if (level == max_size-2) {
			for (size_t emb_id = 0; emb_id < emb_list.size(level); emb_id++) {
				auto vid = emb_list.get_vid(level, emb_id);
				//for (unsigned element_id = 0; element_id < level+1; ++ element_id) {
				unsigned previous_pid = 0;
				if (level > 1) previous_pid = emb_list.get_pid(level, emb_id);
				//if (!toExtend(level, element_id)) continue; //if (element_id != level)
				//auto src = emb_list.get_vertex(level, element_id);
				auto begin = graph->edge_begin(vid);
				auto end = graph->edge_end(vid);
				for (auto e = begin; e != end; e ++) {
					auto dst = graph->getEdgeDst(e);
					if (toAdd(level, dst, emb_list))
						//reduction(getPattern(level, dst, emb_list, previous_pid), emb_list);
						reduction(level, emb_list, vid, dst, previous_pid);
				}
			}
			return;
		}
		for (size_t emb_id = 0; emb_id < emb_list.size(level); emb_id ++) {
			emb_list.set_size(level+1, 0);
			auto vid = emb_list.get_vid(level, emb_id);
			//for (unsigned element_id = 0; element_id < level+1; ++ element_id) {
				//if (!toExtend(level, element_id)) continue;
				//auto src = emb_list.get_vertex(level, element_id);
				unsigned previous_pid = 0;
				if (level > 1) {
					previous_pid = emb_list.get_pid(level, emb_id);
					emb_list.push_history(vid);
				}
				auto begin = graph->edge_begin(vid);
				auto end = graph->edge_end(vid);
				for (auto edge = begin; edge < end; edge ++) {
					auto dst = graph->getEdgeDst(edge);
					if (toAdd(level, dst, emb_list)) {
						auto start = emb_list.size(level+1);
						assert(start < max_degree);
						emb_list.set_vid(level+1, start, dst);
						emb_list.set_size(level+1, start+1);
						update(level, dst, start, previous_pid, emb_list);
					}
				}
			//}
			ego_extend_multi(level+1, emb_list);
			if (level > 1) emb_list.pop_history();
		}
	}
	void ego_extend_opt(unsigned level, EmbeddingList &emb_list) {
		if (level == max_size-2) {
			for (size_t emb_id = 0; emb_id < emb_list.size(level); emb_id ++) {
				auto src = emb_list.get_vid(level, emb_id);
				unsigned previous_pid = 0;
				if (level > 1) previous_pid = emb_list.get_pid(level, emb_id);
				auto begin = graph->edge_begin(src);
				auto end = graph->edge_end(src);
				for (auto e = begin; e < end; e ++) {
					auto dst = graph->getEdgeDst(e);
					if (toAdd(level, dst, emb_list))
						reduction(level, emb_list, src, dst, previous_pid);
				}
			}
			return;
		}
		for (size_t emb_id = 0; emb_id < emb_list.size(level); emb_id ++) {
			emb_list.set_size(level+1, 0);
			auto src = emb_list.get_vid(level, emb_id);
			unsigned previous_pid = 0;
			if (level > 1) {
				previous_pid = emb_list.get_pid(level, emb_id);
				emb_list.push_history(src);
			}
			for (auto e : graph->edges(src)) {
				auto dst = graph->getEdgeDst(e);
				if (toAdd(level, dst, emb_list)) {
					auto start = emb_list.size(level+1);
					emb_list.set_vid(level+1, start, dst);
					emb_list.set_size(level+1, start+1);
					update(level, dst, start, previous_pid, emb_list);
				}
			}
			ego_extend_opt(level+1, emb_list);
			if (level > 1) emb_list.pop_history();
		}
	}

	void solve_motif_equations(EmbeddingList &emb_list) {
		VertexId u = emb_list.get_vid(0, 0);
		VertexId v = emb_list.get_vid(1, 0);
		Ulong tri_count = emb_list.get_tri_count();
		//Ulong wed_count = emb_list.get_wed_count();
		assert(max_size < 5);
		unsigned deg_v = std::distance(graph->edge_begin(v), graph->edge_end(v));
		unsigned deg_u = std::distance(graph->edge_begin(u), graph->edge_end(u));
		if (max_size == 3) {
			accumulators[0] += tri_count;
			accumulators[1] += deg_v - tri_count - 1 + deg_u - tri_count - 1;
		} else {
			Ulong star3_count = (deg_v - tri_count - 1) + (deg_u - tri_count - 1);
			accumulators[4] += (tri_count * (tri_count - 1) / 2); // diamond
			accumulators[0] += tri_count * star3_count; // tailed_triangles
			accumulators[3] += (deg_v - tri_count - 1) * (deg_u - tri_count - 1); // 4-path
			accumulators[1] += (deg_v - tri_count - 1) * (deg_v - tri_count - 2) / 2; // 3-star
			accumulators[1] += (deg_u - tri_count - 1) * (deg_u - tri_count - 2) / 2;
		}
	}
	void solve_3motif_equations(EmbeddingList &emb_list) {
		VertexId u = emb_list.get_vid(0, 0);
		VertexId v = emb_list.get_vid(1, 0);
		Ulong tri_count = emb_list.get_tri_count();
		unsigned deg_v = std::distance(graph->edge_begin(v), graph->edge_end(v));
		unsigned deg_u = std::distance(graph->edge_begin(u), graph->edge_end(u));
		accumulators[0] += tri_count;
		accumulators[1] += deg_v - tri_count - 1 + deg_u - tri_count - 1;
	}
	void solve_4motif_equations(EmbeddingList &emb_list) {
		VertexId u = emb_list.get_vid(0, 0);
		VertexId v = emb_list.get_vid(1, 0);
		Ulong tri_count = emb_list.get_tri_count();
		unsigned deg_v = std::distance(graph->edge_begin(v), graph->edge_end(v));
		unsigned deg_u = std::distance(graph->edge_begin(u), graph->edge_end(u));
		Ulong star3_count = deg_v - tri_count - 1;
		star3_count = star3_count + deg_u - tri_count - 1;
		accumulators[4] += (tri_count * (tri_count - 1) / 2); // diamond
		accumulators[0] += tri_count * star3_count; // tailed_triangles
		accumulators[3] += (deg_v - tri_count - 1) * (deg_u - tri_count - 1); // 4-path
		accumulators[1] += (deg_v - tri_count - 1) * (deg_v - tri_count - 2) / 2; // 3-star
		accumulators[1] += (deg_u - tri_count - 1) * (deg_u - tri_count - 2) / 2;
	}

	void motif_count() {
		//std::cout << "[cxh debug] accumulators[0] = " << accumulators[0].reduce() << "\n";
		//std::cout << "[cxh debug] accumulators[3] = " << accumulators[3].reduce() << "\n";
		//std::cout << "[cxh debug] accumulators[1] = " << accumulators[1].reduce() << "\n";
		#if defined(USE_OPT) || defined(USE_FORMULA)
		if (accumulators.size() == 2) {
			if (is_directed) {
				total_3_tris = accumulators[0].reduce();
				total_3_path = accumulators[1].reduce();
			} else {
				total_3_tris = accumulators[0].reduce()/3;
				total_3_path = accumulators[1].reduce()/2;
			}
		} else {
			if (is_directed) {
				total_4_clique = accumulators[5].reduce();
				total_4_diamond = accumulators[4].reduce() - total_4_clique;
				total_4_cycle = accumulators[2].reduce();
				total_4_path = accumulators[0].reduce() - total_4_cycle;
				total_4_tailed_tris = accumulators[3].reduce() - 2*total_4_diamond;
				total_3_star = accumulators[1].reduce() - total_4_tailed_tris;
			} else {
				total_4_clique = accumulators[5].reduce() / 6;
				total_4_diamond = accumulators[4].reduce() - (6*total_4_clique);
				total_4_cycle = accumulators[2].reduce() / 4;
				total_4_path = accumulators[3].reduce() - (4*total_4_cycle);
				total_4_tailed_tris = (accumulators[0].reduce() - (4*total_4_diamond)) / 2;
				total_3_star = (accumulators[1].reduce() - total_4_tailed_tris) / 3;
			}
		}
		#else
		if (accumulators.size() == 2) {
			total_3_tris = accumulators[0].reduce();
			total_3_path = accumulators[1].reduce();
		} else {
			total_4_clique = accumulators[5].reduce();
			total_4_diamond = accumulators[4].reduce();
			total_4_cycle = accumulators[2].reduce();
			total_4_path = accumulators[0].reduce();
			total_4_tailed_tris = accumulators[3].reduce();
			total_3_star = accumulators[1].reduce();
		} 
		#endif
	}

	void printout_motifs() {
		std::cout << std::endl;
		if (accumulators.size() == 2) {
			std::cout << "\ttriangles\t" << total_3_tris << std::endl;
			std::cout << "\t3-paths\t\t" << total_3_path << std::endl;
		} else if (accumulators.size() == 6) {
			std::cout << "\t4-paths --> " << total_4_path << std::endl;
			std::cout << "\t3-stars --> " << total_3_star << std::endl;
			std::cout << "\t4-cycles --> " << total_4_cycle << std::endl;
			std::cout << "\ttailed-triangles --> " << total_4_tailed_tris << std::endl;
			std::cout << "\tdiamonds --> " << total_4_diamond << std::endl;
			std::cout << "\t4-cliques --> " << total_4_clique << std::endl;
		} else {
			std::cout << "Currently not supported!\n";
		}
		//std::cout << std::endl;
	}
	Ulong get_total_count() { return total_num.reduce(); }

protected:
	int npatterns;
	bool is_directed;
	unsigned max_degree;
	unsigned core;
	UlongAccu total_num;
	std::vector<UlongAccu> accumulators;
	EmbeddingLists emb_lists;
	EdgeList edge_list;
	Ulong total_3_tris;
	Ulong total_3_path;
	Ulong total_4_clique;
	Ulong total_4_diamond;
	Ulong total_4_tailed_tris;
	Ulong total_4_cycle;
	Ulong total_3_star;
	Ulong total_4_path;

	inline bool is_vertexInduced_automorphism(unsigned level, VertexId dst, const EmbeddingList& emb_list, unsigned idx = 0) {
		// the new vertex id should be larger than the first vertex id
		if (dst <= emb_list.get_vertex(level, 0)) return true;
		// the new vertex should not already exist in the embedding
		for (unsigned i = 1; i < level+1; ++i)
			if (dst == emb_list.get_vertex(level, i)) return true;
		// the new vertex should not already be extended by any previous vertex in the embedding
		for (unsigned i = 0; i < idx; ++i)
			if (is_connected(emb_list.get_vertex(level, i), dst)) return true;
		// the new vertex id should be larger than any vertex id after its source vertex in the embedding
		for (unsigned i = idx+1; i < level+1; ++i)
			if (dst < emb_list.get_vertex(level, i)) return true;
		return false;
	}
	inline unsigned find_motif_pattern_id_dfs(unsigned level, VertexId vid, EmbeddingList& emb_list, unsigned previous_pid = 0) {
		unsigned pid = 0;
		//VertexId vid = emb_list.get_vid(level, pos);
		if (level == 1) { // count 3-motifs
			if (emb_list.get_label(vid) == 1) {
				pid = 0; // triangle
			} else {
				pid = 1; //wedge 
			}
		} else if (level == 2) { // count 4-motifs
			if (previous_pid == 0) { // extending a triangle
				if (emb_list.get_label(vid) == 3) {
					pid = 5; // clique
				} else if (emb_list.get_label(vid) == 0) {
					pid = 3; // tailed-triangle
				} else pid = 4;
			} else {
				if (emb_list.get_label(vid) == 3) {
					pid = 4; // diamond
				} else if (emb_list.get_label(vid) == 0) {
					pid = 0; // 3-path
				} else if (emb_list.get_label(vid) == 1) {
					pid = 2; // 4-cycle
				} else pid = 3; // tailed-triangle
			}
		}
		return pid;
	}
	inline unsigned find_motif_pattern_id_dfs(unsigned n, unsigned idx, VertexId dst, const VertexEmbedding& emb, unsigned previous_pid) {
		unsigned pid = 0;
		if (n == 2) { // count 3-motifs
			pid = 1; // 3-chain
			if (idx == 0) {
				if (is_connected(emb.get_vertex(1), dst)) pid = 0; // triangle
			}
		} else if (n == 3) { // count 4-motifs
			unsigned num_edges = 1;
			if (previous_pid == 0) { // extending a triangle
				for (unsigned j = idx+1; j < n; j ++)
					if (is_connected(emb.get_vertex(j), dst)) num_edges ++;
				pid = num_edges + 2; // p3: tailed-triangle; p4: diamond; p5: 4-clique
			} else { // extending a 3-chain
				assert(previous_pid == 1);
				std::vector<bool> connected(3, false);
				connected[idx] = true;
				for (unsigned j = idx+1; j < n; j ++) {
					if (is_connected(emb.get_vertex(j), dst)) {
						num_edges ++;
						connected[j] = true;
					}
				}
				if (num_edges == 1) {
					pid = 0; // p0: 3-path
					unsigned center = 1;
					center = is_connected(emb.get_vertex(1), emb.get_vertex(2)) ? 1 : 0;
					if (idx == center) pid = 1; // p1: 3-star
				} else if (num_edges == 2) {
					pid = 2; // p2: 4-cycle
					unsigned center = 1;
					center = is_connected(emb.get_vertex(1), emb.get_vertex(2)) ? 1 : 0;
					if (connected[center]) pid = 3; // p3: tailed-triangle
				} else {
					pid = 4; // p4: diamond
				}
			}
		} else { // count 5-motif and beyond
			find_motif_pattern_id_eigen(n, idx, dst, emb);
		}
		return pid;
	}
};

#endif
