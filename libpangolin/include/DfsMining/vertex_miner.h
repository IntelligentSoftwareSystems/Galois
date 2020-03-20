#pragma once
#include "miner.h"
#include "ptypes.h"
#include "edgelist.h"
#include "embedding_list_dfs.h"

template <typename ElementTy, typename EmbeddingTy, typename API, 
	bool enable_dag=false, bool is_single=true, 
	bool use_ccode=true, bool shrink=false, bool use_formula=false>
class VertexMinerDFS : public Miner<ElementTy,EmbeddingTy,enable_dag> {
typedef EmbeddingList<ElementTy,EmbeddingTy,is_single,use_ccode,shrink,use_formula> EmbeddingListTy;
typedef galois::substrate::PerThreadStorage<EmbeddingListTy> EmbeddingLists;
public:
	VertexMinerDFS(unsigned max_sz, int nt) : 
		Miner<ElementTy,EmbeddingTy,enable_dag>(max_sz, nt), npatterns(1) {
		init_counter();
		if (is_single) {
			accumulators.resize(1);
			accumulators[0].reset();
		}
		is_directed = enable_dag; // DAG
	}
	virtual ~VertexMinerDFS() {}
	void print_output() { }
	void set_num_patterns(int np) { // for multi-pattern problems
		npatterns = np;
		accumulators.resize(np);
		for (int i = 0; i < np; i++) 
			accumulators[i].reset();
	}
	void init_counter() {
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
		removed_edges.reset();
	}

	void init_edgelist(bool symmetrize = false) {
		edge_list.init(this->graph, enable_dag, symmetrize);
		if (shrink) core = edge_list.get_core();
	}

	void init_emb_list() {
		for (int i = 0; i < this->num_threads; i++) {
			if (shrink) {
				emb_lists.getLocal(i)->allocate(&(this->graph), this->max_size, core);
			} else
				emb_lists.getLocal(i)->allocate(&(this->graph), this->max_size, this->max_degree);
		}
	}

	void solver () {
		//edge_process_opt();
		edge_parallel_solver();
		//vertex_process_opt();
		//vertex_process();
	}
	/*
	void vertex_process() {
		std::cout << "DFS vertex processing without advanced optimization\n";
		galois::do_all(galois::iterate(this->graph.begin(), this->graph.end()), 
		[&](const auto& vid) {
			EmbeddingListTy *emb_list = emb_lists.getLocal();
			emb_list->init_vertex(vid);
			//if (std::is_same<EmbeddingTy,>::value)
			if (is_single) ego_extend_single(1, *emb_list); // TODO: use constexpr
			else ego_extend_multi(1, *emb_list);
			emb_list->clear_labels(vid);
		}, galois::chunk_size<1>(), galois::steal(), galois::loopname("DfsVertexNaiveSolver"));
		if (!is_single) motif_count();
	}
	//*/
	void edge_parallel_solver() {
		std::cout << "DFS edge processing without advanced optimization\n";
		std::cout << "Number of single-edge embeddings: " << edge_list.size() << "\n";
		galois::do_all(galois::iterate(edge_list), [&](const SEdge &edge) {
			//std::cout << "Processing edge: " << edge.to_string() << "\n";
			EmbeddingListTy *emb_list = emb_lists.getLocal();
			if (is_single) {
				if (!degree_filter(edge.src, edge.dst)) {
					emb_list->init_edge(edge);
					if (use_ccode) {
						extend_single(1, *emb_list);
						emb_list->clear_labels(edge.src);
					} else {
						//ego_extend_single_no_labeling(1, *emb_list);
					}
				}
			} else {
				emb_list->init_edge(edge);
				if (use_formula) {
					//ego_extend_opt(1, *emb_list); // egonet DFS with formula
					//solve_motif_equations(*emb_list);
					//if (this->max_size == 4)
					//	emb_list->clear_labels(edge.dst);
				} else {
					//std::cout << "multi-pattern extension without using formula\n";
					extend_multi(1, *emb_list); // egonet DFS
					//ego_extend_multi_non_canonical(1, *emb_list); // egonet DFS
					//ego_extend_sgl(1, *emb_list);
					//ego_extend_sgl_auto(1, *emb_list);
					//ego_extend_sgl_naive(1, *emb_list);
					emb_list->clear_labels(edge.dst);
					emb_list->clear_labels(edge.src);
				}
			}
		}, galois::chunk_size<1>(), galois::steal(), galois::loopname("EdgeSolver"));
		if (!is_single) motif_count();
	}

	bool degree_filter(VertexId src, VertexId dst) {
		//if (this->graph.get_degree(edge.src) >= this->max_size-1 
			//&& this->graph.get_degree(edge.dst) >= this->max_size-2) {
		//if (this->graph.get_degree(edge.src) < this->max_size-1 
		//	|| this->graph.get_degree.(edge.dst) < this->max_size-2) { 
		//	removed_edges += 1; return; }
		return false;
	}
	/*
	void vertex_process_opt() {
		std::cout << "DFS vertex processing using advanced optimization\n";
		galois::for_each(galois::iterate(this->graph.begin(), this->graph.end()), 
		[&](const auto &vid, auto &ctx) {
			EmbeddingListTy *emb_list = emb_lists.getLocal();
			emb_list->init_vertex(vid);
			if (is_single)
				dfs_extend(1, *emb_list);
		}, galois::chunk_size<1>(), galois::steal(), galois::loopname("DfsVertexSolver"));
		//if (!is_single) motif_count();
	}
	//*/
	/*
	// DFS extension for k-cliques
	void ego_extend_single_no_labeling(unsigned level, EmbeddingListTy &emb_list) {
		if (level == this->max_size-2) {
			for (size_t emb_id = 0; emb_id < emb_list.size(level); emb_id ++) {
				auto vid = emb_list.get_vertex(level, emb_id);
				if (level > 1) emb_list.push_history(vid);
				auto emb = emb_list.get_history();
				auto begin = this->graph.edge_begin(vid);
				auto end = this->graph.edge_end(vid);
				for (auto e = begin; e < end; e ++) {
					auto dst = this->graph.getEdgeDst(e);
					if (is_all_connected_dag(dst, emb, level))
						API::reduction(accumulators[0]);
				}
				if (level > 1) emb_list.pop_history();
			}
			return;
		}
		for (size_t emb_id = 0; emb_id < emb_list.size(level); emb_id ++) {
			auto vid = emb_list.get_vertex(level, emb_id);
			if (level > 1) emb_list.push_history(vid);
			auto emb = emb_list.get_history();
			auto begin = this->graph.edge_begin(vid);
			auto end = this->graph.edge_end(vid);
			emb_list.set_size(level+1, 0);
			for (auto e = begin; e < end; e ++) {
				auto dst = this->graph.getEdgeDst(e);
				if (is_all_connected_dag(dst, emb, level)) {
					auto start = emb_list.size(level+1);
					emb_list.set_vid(level+1, start, dst);
					emb_list.set_size(level+1, start+1);
				}
			}
			ego_extend_single_no_labeling(level+1, emb_list);
			if (level > 1) emb_list.pop_history();
		}
	}
	*/
	// DFS extension for k-cliques
	void extend_single(unsigned level, EmbeddingListTy &emb_list) {
		if (level == this->max_size-2) {
			for (size_t emb_id = 0; emb_id < emb_list.size(level); emb_id ++) {
				auto vid = emb_list.get_vertex(level, emb_id);
				//auto begin = emb_list.edge_begin(level, vid);
				//auto end = emb_list.edge_end(level, vid);
				auto begin = this->graph.edge_begin(vid);
				auto end = this->graph.edge_end(vid);
				for (auto e = begin; e < end; e ++) {
					auto dst = this->graph.getEdgeDst(e);
					//auto dst = emb_list.getEdgeDst(e);
					auto ccode = emb_list.get_label(dst);
					if (API::toAdd(level, dst, level, ccode, NULL))
					//if (level == ccode)
						API::reduction(accumulators[0]);
				}
			}
			return;
		}
		for (size_t emb_id = 0; emb_id < emb_list.size(level); emb_id ++) {
			auto vid = emb_list.get_vertex(level, emb_id);
			//auto begin = emb_list.edge_begin(level, vid);
			//auto end = emb_list.edge_end(level, vid);
			auto begin = this->graph.edge_begin(vid);
			auto end = this->graph.edge_end(vid);
			emb_list.set_size(level+1, 0);
			for (auto e = begin; e < end; e ++) {
				auto dst = this->graph.getEdgeDst(e);
				//auto dst = emb_list.getEdgeDst(e);
				auto ccode = emb_list.get_label(dst);
				if (API::toAdd(level, dst, level, ccode, NULL)) {
				//if (level == ccode) {
					auto start = emb_list.size(level+1);
					emb_list.set_vid(level+1, start, dst);
					emb_list.set_label(dst, level+1);
					emb_list.set_size(level+1, start+1);
				}
			}
			extend_single(level+1, emb_list);
			emb_list.reset_labels(level);
		}
	}
	/*
	// DFS extension for k-cliques using graph shrinking
	void dfs_extend(unsigned level, EmbeddingListTy &emb_list) {
		if (level == this->max_size-2) {
			for (size_t emb_id = 0; emb_id < emb_list.size(level); emb_id ++) {
				auto vid = emb_list.get_vertex(level, emb_id);
				auto begin = emb_list.edge_begin(level, vid);
				auto end = emb_list.edge_end(level, vid);
				for (auto e = begin; e < end; e ++) {
					//auto dst = this->graph.getEdgeDst(e);
					//auto ccode = emb_list.get_label(dst);
					//if (API::toAdd(level, dst, level, ccode, NULL))
						API::reduction(accumulators[0]);
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
				auto ccode = emb_list.get_label(dst);
				//if (API::toAdd(level, dst, level, ccode, NULL)) {
				if (ccode == level) { 
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

	void ego_extend_sgl(unsigned level, EmbeddingListTy &emb_list) {
		if (level == this->max_size-2) {
			for (size_t emb_id = 0; emb_id < emb_list.size(level); emb_id++) {
				unsigned last_vid = 0;
				if (level > 1) {
					last_vid = emb_list.get_vertex(level, emb_id);
					emb_list.push_history(last_vid);
					emb_list.update_labels(level, last_vid);
				}
				unsigned previous_pid = 0, src_idx = 0;
				if (level > 1) previous_pid = emb_list.get_pid(level, emb_id);
				if (level > 1) src_idx = emb_list.get_src(level, emb_id);
				for (unsigned element_id = 0; element_id < level+1; ++ element_id) {
					//if (!API::toExtend(level, element_id, NULL)) continue; // extend all
					auto src = emb_list.get_history(element_id);
					auto begin = this->graph.edge_begin(src);
					auto end = this->graph.edge_end(src);
					for (auto e = begin; e < end; e ++) {
						auto dst = this->graph.getEdgeDst(e);
						auto ccode = emb_list.get_label(dst);
						if (API::toAdd(level, dst, element_id, ccode, NULL)) { // add canonical
							if (is_tailed_triangle(previous_pid, src_idx, ccode))
								accumulators[0] += 1;
						}
					}
				}
				if (level > 1) emb_list.resume_labels(level, last_vid);
				if (level > 1) emb_list.pop_history();
			}
			return;
		}
		for (size_t emb_id = 0; emb_id < emb_list.size(level); emb_id ++) {
			if (level > 1) {
				unsigned last_vid = emb_list.get_vertex(level, emb_id);
				emb_list.push_history(last_vid);
				emb_list.update_labels(level, last_vid);
			}
			unsigned previous_pid = 0;
			if (level > 1) previous_pid = emb_list.get_pid(level, emb_id);
			emb_list.set_size(level+1, 0);
			for (unsigned element_id = 0; element_id < level+1; ++ element_id) {
				//if (!API::toExtend(level, element_id, NULL)) continue;
				auto src = emb_list.get_history(element_id);
				auto begin = this->graph.edge_begin(src);
				auto end = this->graph.edge_end(src);
				for (auto edge = begin; edge < end; edge ++) {
					auto dst = this->graph.getEdgeDst(edge);
					auto ccode = emb_list.get_label(dst);
					if (API::toAdd(level, dst, element_id, ccode, NULL)) {
						auto start = emb_list.size(level+1);
						assert(start < this->max_degree);
						emb_list.set_vid(level+1, start, dst);
						emb_list.set_size(level+1, start+1);
						//unsigned pid = getPattern(level, dst, emb_list, previous_pid, element_id);
						unsigned pid = find_pattern_id_dfs(
							level, dst, emb_list, previous_pid, element_id);
						emb_list.set_pid(level+1, start, pid);
						emb_list.set_src(level+1, start, element_id);
					}
				}
			}
			ego_extend_sgl(level+1, emb_list);
			if (level > 1) emb_list.pop_history();
		}
	}
	void ego_extend_sgl_auto(unsigned level, EmbeddingListTy &emb_list) {
		if (level == this->max_size-2) {
			for (size_t emb_id = 0; emb_id < emb_list.size(level); emb_id++) {
				unsigned last_vid = 0;
				if (level > 1) {
					last_vid = emb_list.get_vertex(level, emb_id);
					emb_list.push_history(last_vid);
					emb_list.update_labels(level, last_vid);
				}
				auto src = emb_list.get_history(level);
				#ifdef DIAMOND
				for (size_t emb_id = 0; emb_id < emb_list.size(level); emb_id++) {
					auto dst = emb_list.get_vertex(level, emb_id);
					if (dst != src && emb_list.get_label(dst) == 3)
						accumulators[0] += 1;
				}
				#else
				auto begin = this->graph.edge_begin(src);
				auto end = this->graph.edge_end(src);
				for (auto e = begin; e < end; e ++) {
					auto dst = this->graph.getEdgeDst(e);
					if (dst == emb_list.get_history(0) || dst == emb_list.get_history(1)) continue;
					//if (level > 1 && dst == emb_list.get_history(2)) continue;
					if (emb_list.get_label(dst) == 4) // tailed_triangle
						accumulators[0] += 1;
				}
				#endif
				if (level > 1) emb_list.resume_labels(level, last_vid);
				if (level > 1) emb_list.pop_history();
			}
			return;
		}
		for (size_t emb_id = 0; emb_id < emb_list.size(level); emb_id ++) {
			if (level > 1) {
				unsigned last_vid = emb_list.get_vertex(level, emb_id);
				emb_list.push_history(last_vid);
				emb_list.update_labels(level, last_vid);
			}
			emb_list.set_size(level+1, 0);
			auto src = emb_list.get_history(level);
			auto begin = this->graph.edge_begin(src);
			auto end = this->graph.edge_end(src);
			for (auto edge = begin; edge < end; edge ++) {
				auto dst = this->graph.getEdgeDst(edge);
				//#ifdef DIAMOND
				if (emb_list.get_label(dst) == 3) { // triangles
				//#else // cycle
				//if (emb_list.get_label(dst) != 3) { // wedges
				//#endif
					auto start = emb_list.size(level+1);
					emb_list.set_vid(level+1, start, dst);
					emb_list.set_size(level+1, start+1);
				}
			}
			ego_extend_sgl_auto(level+1, emb_list);
			if (level > 1) emb_list.pop_history();
		}
	}

	void ego_extend_sgl_naive(unsigned level, EmbeddingListTy &emb_list) {
		if (level == this->max_size-2) {
			for (size_t emb_id = 0; emb_id < emb_list.size(level); emb_id++) {
				unsigned last_vid = 0;
				if (level > 1) {
					last_vid = emb_list.get_vertex(level, emb_id);
					emb_list.push_history(last_vid);
					emb_list.update_labels(level, last_vid);
				}
				unsigned previous_pid = 0, src_idx = 0;
				if (level > 1) previous_pid = emb_list.get_pid(level, emb_id);
				if (level > 1) src_idx = emb_list.get_src(level, emb_id);
				for (unsigned element_id = 0; element_id < level+1; ++ element_id) {
					auto src = emb_list.get_history(element_id);
					auto begin = this->graph.edge_begin(src);
					auto end = this->graph.edge_end(src);
					for (auto e = begin; e < end; e ++) {
						auto dst = this->graph.getEdgeDst(e);
						if (dst == emb_list.get_history(0) || 
							dst == emb_list.get_history(1)) continue;
						if (level > 1 && dst == emb_list.get_history(2)) continue;
						#ifdef DIAMOND
						if (is_diamond(previous_pid, emb_list.get_label(dst)))
						#else
						if (is_tailed_triangle(previous_pid, src_idx, emb_list.get_label(dst)))
						#endif
							accumulators[0] += 1;
					}
				}
				if (level > 1) emb_list.resume_labels(level, last_vid);
				if (level > 1) emb_list.pop_history();
			}
			return;
		}
		for (size_t emb_id = 0; emb_id < emb_list.size(level); emb_id ++) {
			if (level > 1) {
				unsigned last_vid = emb_list.get_vertex(level, emb_id);
				emb_list.push_history(last_vid);
				emb_list.update_labels(level, last_vid);
			}
			emb_list.set_size(level+1, 0);
			unsigned previous_pid = 0;
			if (level > 1) previous_pid = emb_list.get_pid(level, emb_id);
			for (unsigned element_id = 0; element_id < level+1; ++ element_id) {
				auto src = emb_list.get_history(element_id);
				auto begin = this->graph.edge_begin(src);
				auto end = this->graph.edge_end(src);
				for (auto edge = begin; edge < end; edge ++) {
					auto dst = this->graph.getEdgeDst(edge);
					if (dst == emb_list.get_history(0) || 
						dst == emb_list.get_history(1)) continue;
					if (level > 1 && dst == emb_list.get_history(2)) continue;
					auto start = emb_list.size(level+1);
					emb_list.set_vid(level+1, start, dst);
					emb_list.set_size(level+1, start+1);
					unsigned pid = find_pattern_id_dfs(
						level, dst, emb_list, previous_pid, element_id);
					emb_list.set_pid(level+1, start, pid);
					emb_list.set_src(level+1, start, element_id);
				}
			}
			ego_extend_sgl_naive(level+1, emb_list);
			if (level > 1) emb_list.pop_history();
		}
	}
	//*/

	void extend_multi(unsigned level, EmbeddingListTy &emb_list) {
		if (level == this->max_size-2) {
			for (size_t emb_id = 0; emb_id < emb_list.size(level); emb_id++) {
				unsigned last_vid = 0;
				if (level > 1) {
					last_vid = emb_list.get_vertex(level, emb_id);
					emb_list.push_history(last_vid);
					emb_list.update_labels(level, last_vid);
				}
				EmbeddingTy emb(level+1);
				emb_list.get_embedding(level, emb);
				//std::cout << "emb: " << emb << "\n";
				unsigned previous_pid = 0, src_idx = 0;
				if (level > 1) previous_pid = emb_list.get_pid(level, emb_id);
				if (level > 1) src_idx = emb_list.get_src(level, emb_id);
				for (unsigned element_id = 0; element_id < level+1; ++ element_id) {
					if (!API::toExtend(level, element_id, &emb)) continue; // extend all
					auto src = emb.get_vertex(element_id);
					auto begin = this->graph.edge_begin(src);
					auto end = this->graph.edge_end(src);
					for (auto e = begin; e < end; e ++) {
						auto dst = this->graph.getEdgeDst(e);
						auto ccode = emb_list.get_label(dst);
						//std::cout << "\t idx=" << element_id << ", src=" << src << ", dst=" << dst << ", ccode=" << unsigned(ccode) << "\n";
						if (API::toAdd(level, dst, element_id, ccode, &emb)) {
							//unsigned pid = getPattern(level, 
								//dst, emb_list, previous_pid, src_idx);
							// get pattern id using the labels
							unsigned pid = find_pattern_id_dfs(
								level, dst, emb_list, previous_pid, src_idx);
							//std::cout << "\t\t pid = " << pid << "\n";
							API::reduction(accumulators[pid]);
						}
					}
				}
				if (level > 1) emb_list.resume_labels(level, last_vid);
				if (level > 1) emb_list.pop_history();
			}
			return;
		}
		for (size_t emb_id = 0; emb_id < emb_list.size(level); emb_id ++) {
			if (level > 1) {
				unsigned last_vid = emb_list.get_vertex(level, emb_id);
				emb_list.push_history(last_vid);
				emb_list.update_labels(level, last_vid);
			}
			EmbeddingTy emb(level+1);
			emb_list.get_embedding(level, emb);
			//std::cout << "emb: " << emb << "\n";
			unsigned previous_pid = 0;
			if (level > 1) previous_pid = emb_list.get_pid(level, emb_id);
			emb_list.set_size(level+1, 0);
			for (unsigned element_id = 0; element_id < level+1; ++ element_id) {
				if (!API::toExtend(level, element_id, &emb)) continue;
				auto src = emb.get_vertex(element_id);
				//auto src = emb_list.get_history(element_id);
				auto begin = this->graph.edge_begin(src);
				auto end = this->graph.edge_end(src);
				for (auto edge = begin; edge < end; edge ++) {
					auto dst = this->graph.getEdgeDst(edge);
					auto ccode = emb_list.get_label(dst);
					if (API::toAdd(level, dst, element_id, ccode, &emb)) {
						auto start = emb_list.size(level+1);
						assert(start < this->max_degree);
						emb_list.set_vid(level+1, start, dst);
						emb_list.set_size(level+1, start+1);
						//unsigned pid = getPattern(level, 
							//dst, emb_list, previous_pid, element_id);
						unsigned pid = find_pattern_id_dfs(
							level, dst, emb_list, previous_pid, element_id);
						emb_list.set_pid(level+1, start, pid);
						emb_list.set_src(level+1, start, element_id);
					}
				}
			}
			extend_multi(level+1, emb_list);
			if (level > 1) emb_list.pop_history();
		}
	}

	void ego_extend_multi_non_canonical(unsigned level, EmbeddingListTy &emb_list) {
		if (level == this->max_size-2) {
			for (size_t emb_id = 0; emb_id < emb_list.size(level); emb_id++) {
				unsigned last_vid = 0;
				if (level > 1) {
					last_vid = emb_list.get_vertex(level, emb_id);
					emb_list.push_history(last_vid);
					emb_list.update_labels(level, last_vid);
				}
				unsigned previous_pid = 0, src_idx = 0;
				if (level > 1) previous_pid = emb_list.get_pid(level, emb_id);
				if (level > 1) src_idx = emb_list.get_src(level, emb_id);
				for (unsigned element_id = 0; element_id < level+1; ++ element_id) {
					auto src = emb_list.get_history(element_id);
					auto begin = this->graph.edge_begin(src);
					auto end = this->graph.edge_end(src);
					for (auto e = begin; e < end; e ++) {
						auto dst = this->graph.getEdgeDst(e);
						if (dst == emb_list.get_history(0) || 
							dst == emb_list.get_history(1)) continue;
						if (level > 1 && dst == emb_list.get_history(2)) continue;
						unsigned pid = find_pattern_id_dfs(
							level, dst, emb_list, previous_pid, src_idx);
						API::reduction(accumulators[pid]);
					}
				}
				if (level > 1) emb_list.pop_history();
				if (level > 1) emb_list.resume_labels(level, last_vid);
			}
			return;
		}
		for (size_t emb_id = 0; emb_id < emb_list.size(level); emb_id ++) {
			if (level > 1) {
				unsigned last_vid = emb_list.get_vertex(level, emb_id);
				emb_list.push_history(last_vid);
				emb_list.update_labels(level, last_vid);
			}
			unsigned previous_pid = 0;
			if (level > 1) previous_pid = emb_list.get_pid(level, emb_id);
			emb_list.set_size(level+1, 0);
			for (unsigned element_id = 0; element_id < level+1; ++ element_id) {
				auto src = emb_list.get_history(element_id);
				auto begin = this->graph.edge_begin(src);
				auto end = this->graph.edge_end(src);
				for (auto edge = begin; edge < end; edge ++) {
					auto dst = this->graph.getEdgeDst(edge);
					if (dst == emb_list.get_history(0) || 
						dst == emb_list.get_history(1)) continue;
					if (level > 1 && dst == emb_list.get_history(2)) continue;
					auto start = emb_list.size(level+1);
					emb_list.set_vid(level+1, start, dst);
					emb_list.set_size(level+1, start+1);
					unsigned pid = find_pattern_id_dfs(
						level, dst, emb_list, previous_pid, element_id);
					emb_list.set_pid(level+1, start, pid);
					emb_list.set_src(level+1, start, element_id);
				}
			}
			ego_extend_multi_non_canonical(level+1, emb_list);
			if (level > 1) emb_list.pop_history();
		}
	}
/*
	void ego_extend_opt(unsigned level, EmbeddingListTy &emb_list) {
		if (level == this->max_size-2) {
			for (size_t emb_id = 0; emb_id < emb_list.size(level); emb_id ++) {
				auto src = emb_list.get_vid(level, emb_id);
				unsigned previous_pid = 0;
				if (level > 1) previous_pid = emb_list.get_pid(level, emb_id);
				auto begin = this->graph.edge_begin(src);
				auto end = this->graph.edge_end(src);
				for (auto e = begin; e < end; e ++) {
					auto dst = this->graph.getEdgeDst(e);
					auto ccode = emb_list.get_label(dst);
					if (API::toAdd(level, dst, level, ccode, NULL)) {
						API::reduction(level, src, dst, ccode, previous_pid);
					}
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
			for (auto e : this->graph.edges(src)) {
				auto dst = this->graph.getEdgeDst(e);
				auto ccode = emb_list.get_label(dst);
				if (API::toAdd(level, dst, level, ccode, NULL)) {
					auto start = emb_list.size(level+1);
					emb_list.set_vid(level+1, start, dst);
					emb_list.set_size(level+1, start+1);
					API::update(level, dst, start, ccode, previous_pid);
				}
			}
			ego_extend_opt(level+1, emb_list);
			if (level > 1) emb_list.pop_history();
		}
	}
//*/
	void solve_motif_equations(EmbeddingListTy &emb_list) {
		VertexId u = emb_list.get_vid(0, 0);
		VertexId v = emb_list.get_vid(1, 0);
		Ulong tri_count = emb_list.get_tri_count();
		//Ulong wed_count = emb_list.get_wed_count();
		assert(this->max_size < 5);
		unsigned deg_v = this->graph.get_degree(v);
		unsigned deg_u = this->graph.get_degree(u);
		if (this->max_size == 3) {
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
	void solve_3motif_equations(EmbeddingListTy &emb_list) {
		VertexId u = emb_list.get_vid(0, 0);
		VertexId v = emb_list.get_vid(1, 0);
		Ulong tri_count = emb_list.get_tri_count();
		unsigned deg_v = this->graph.get_degree(v);
		unsigned deg_u = this->graph.get_degree(u);
		accumulators[0] += tri_count;
		accumulators[1] += deg_v - tri_count - 1 + deg_u - tri_count - 1;
	}
	void solve_4motif_equations(EmbeddingListTy &emb_list) {
		VertexId u = emb_list.get_vid(0, 0);
		VertexId v = emb_list.get_vid(1, 0);
		Ulong tri_count = emb_list.get_tri_count();
		unsigned deg_v = this->graph.get_degree(v);
		unsigned deg_u = this->graph.get_degree(u);
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
	Ulong get_total_count() { return accumulators[0].reduce(); }

protected:
	int npatterns;
	bool is_directed;
	unsigned core;
	UlongAccu removed_edges;
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

	inline bool is_diamond(unsigned previous_pid, unsigned qcode) {
		if ((previous_pid == 0 && (qcode == 3 || qcode == 5 || qcode == 6)) ||
			(previous_pid == 1 && qcode == 7)) return true;
		return false;
	}

	inline bool is_4cycle(unsigned previous_pid, unsigned src_idx, unsigned qcode) {
		if (previous_pid == 1) {
			if (src_idx == 0) {
				if (qcode == 6) return true;
			} else {
				if (qcode == 5) return true;
			}
		}
		return false;
	}

	inline bool is_tailed_triangle(unsigned previous_pid, unsigned src_idx, unsigned qcode) {
		if (previous_pid == 0 && qcode < 5 && qcode != 3) return true; 
		if (previous_pid == 1) {
			if (src_idx == 0) {
				if (qcode == 3 || qcode == 5) return true;
			} else {
				if (qcode == 3 || qcode == 6) return true;
			}
		}
		return false;
	}

	inline unsigned find_pattern_id_dfs(unsigned level, VertexId vid, EmbeddingListTy& emb_list, unsigned previous_pid = 0, unsigned src_idx = 0) {
		unsigned pid = 0;
		//VertexId vid = emb_list.get_vid(level, pos);
		if (level == 1) { // count 3-motifs
			if (emb_list.get_label(vid) == 3) {
				pid = 0; // triangle
			} else {
				pid = 1; //wedge 
			}
		} else if (level == 2) { // count 4-motifs
			if (previous_pid == 0) { // extending a triangle
				if (emb_list.get_label(vid) == 7) {
					pid = 5; // clique
				} else if (emb_list.get_label(vid) == 3 || emb_list.get_label(vid) == 5 || emb_list.get_label(vid) == 6) {
					pid = 4; // diamond
				} else pid = 3; // tailed-triangle
			} else {
				if (emb_list.get_label(vid) == 7) {
					pid = 4; // diamond
				} else if (src_idx == 0) {
					if (emb_list.get_label(vid) == 6) pid = 2; // 4-cycle
					else if (emb_list.get_label(vid) == 3 || emb_list.get_label(vid) == 5) pid = 3; // tailed-triangle
					else if (emb_list.get_label(vid) == 1) pid = 1; // 3-star
					else pid = 0 ; // 4-chain
				} else {
					if (emb_list.get_label(vid) == 5) pid = 2; // 4-cycle
					else if (emb_list.get_label(vid) == 3 || emb_list.get_label(vid) == 6) pid = 3; // tailed-triangle
					else if (emb_list.get_label(vid) == 2) pid = 1; // 3-star
					else pid = 0; // 4-chain
				}
			}
		}
		return pid;
	}
};
