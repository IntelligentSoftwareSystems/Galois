#pragma once
#include "pangolin/miner.h"
#include "pangolin/ptypes.h"
#include "edgelist.h"
#include "embedding_list_dfs.h"

template <typename API, bool enable_dag=false, bool is_single=true, 
	bool use_ccode=true, bool use_local_graph=false, bool use_pcode=false, 
	bool do_local_counting=false, bool edge_par=true, bool is_clique=true>
class VertexMinerDFS : public Miner<SimpleElement,BaseEmbedding,enable_dag> {
public:
typedef EmbeddingList<is_single,use_ccode,use_pcode,use_local_graph,do_local_counting,is_clique> EmbeddingListTy;
typedef galois::substrate::PerThreadStorage<EmbeddingListTy> EmbeddingLists;
	VertexMinerDFS(unsigned max_sz, int nt, unsigned slevel = 1) : 
		Miner<SimpleElement,BaseEmbedding,enable_dag>(max_sz, nt), npatterns(1), starting_level(slevel) {
		if (is_single) {
			accumulators.resize(1);
			accumulators[0].reset();
			removed_edges.reset();
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
		init_counter();
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
	}
	void initialize(std::string pattern_filename) {
		core = this->max_degree;
		//if (edge_par || use_local_graph) 
		init_edgelist();
		init_emb_list();
		if (is_single && !is_clique) {
			if (pattern_filename == "") {
				std::cout << "need specify pattern file name using -p\n";
				exit(1);
			}
			unsigned pid = this->read_pattern(pattern_filename);
			//unsigned pid = this->read_pattern(pattern_filename, "gr", true);
			std::cout << "pattern id = " << pid << "\n";
			set_input_pattern(pid);
		}
	}
	void set_input_pattern(unsigned pid) {
		input_pid = pid;
	} 
	void init_edgelist(bool symmetrize = false) {
		edge_list.init(this->graph, enable_dag, symmetrize);
		if (use_local_graph) { // TODO: use constexpr
			// rebuild the graph to minimize the max_degree to save memory for the local graph
			core = edge_list.generate_graph(this->graph);
		}
	}
	void init_emb_list() {
		for (int i = 0; i < this->num_threads; i++) {
			emb_lists.getLocal(i)->allocate(&(this->graph), this->max_size, core, npatterns);
		}
	}
	/*
	inline VertexId get_query_vertex(unsigned id) { return id; }
	inline VertexId get_matching_order(unsigned id) { return id; }
	VertexId get_query_vertex(unsigned id) { return matching_order[id]; }
	inline VertexId get_matching_order(unsigned id) { return matching_order_map[id]; }
	std::vector<VertexId> matching_order;
	std::vector<VertexId> matching_order_map;
	std::vector<VertexId> automorph_group_id;
	// Read the preset file to hardcode the presets
	void read_presets() {
		matching_order_map.resize(ms);
		automorph_group_id.resize(ms);
		std::ifstream ifile;
		ifile.open(preset_filename);
		if (!ifile) printf("Error in reading file %s\n", preset_filename.c_str());
		VertexId x;
		for (size_t i = 0; i< max_size; ++i) {
			ifile >> x;
			matching_order[i] = x;
			if(debug) std::cout << "matching_order[" << i << "] = " << x << "\n";
		}
		for (size_t i = 0; i < max_size; ++i) {
			ifile >> x;
			matching_order_map[i] = x;
			if(debug) std::cout << "matching_map[" << i << "] = " << x << "\n";
		}
		for (size_t i = 0; i < max_size; ++i) {
			ifile >> x;
			automorph_group_id[i] = x;
			if(debug) std::cout << "automorph_group_id[" << i << "] = " << x << "\n";
		}
		ifile.close();
	}

	void ordered_vertex_parallel_solver() {
		VertexId curr_qnode = get_query_vertex(0);
		galois::do_all(galois::iterate(this->graph.begin(), this->graph.end()), [&](const auto& src) {
			auto emb_list = emb_lists.getLocal();
			if (this->graph.get_degree(src) < this->pattern.get_degree(curr_qnode)) return;
			emb_list->init_vertex(src);
			extend_ordered(1, 0, *emb_list);
		}, galois::chunk_size<1>(), galois::steal(), galois::loopname("VertexParallelSolver"));
	}
	//*/

	void solver () {
		if (edge_par)
			edge_parallel_solver();
			//ordered_edge_parallel_solver();
		else
			vertex_parallel_solver();
	}

	void ordered_edge_parallel_solver() {
		std::cout << "DFS ordered edge parallel processing\n";
		galois::do_all(galois::iterate(edge_list), [&](const SEdge &edge) {
			if (this->graph.get_degree(edge.src) < this->pattern.get_degree(0)) return;
			if (this->graph.get_degree(edge.dst) < this->pattern.get_degree(1)) return;
			auto emb_list = emb_lists.getLocal();
			if (edge.src < edge.dst) { 
				emb_list->init_edge(edge);
				extend_ordered(starting_level, *emb_list);
				emb_list->clear_labels(edge.dst);
				emb_list->clear_labels(edge.src);
			}
		}, galois::chunk_size<1>(), galois::steal(), galois::loopname("EdgeParallelSolver"));
	}

	void vertex_parallel_solver() {
		std::cout << "DFS vertex parallel processing\n";
		//galois::for_each(galois::iterate(this->graph.begin(), this->graph.end()), [&](const auto &vid, auto &ctx) {
		galois::do_all(galois::iterate(this->graph.begin(), this->graph.end()), [&](const auto& vid) {
			EmbeddingListTy *emb_list = emb_lists.getLocal();
			//std::cout << "Processing vertex " << vid << "\n";
			emb_list->init_vertex(vid);
			if (is_single) { // TODO: use constexpr
				if (is_clique) {
					extend_clique(starting_level, *emb_list); 
				} else {
					//extend_single(starting_level, *emb_list); 
					extend_single_nc(starting_level, *emb_list);
				}
			} else {
				extend_multi(starting_level, *emb_list);
			}
			if (!use_local_graph) emb_list->clear_labels(vid);
		}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("VertexParallelSolver"));
		//}, galois::chunk_size<1>(), galois::steal(), galois::no_conflicts(), galois::loopname("VertexParallelSolver"));
		if (!is_single) motif_count();
	}

	void edge_parallel_solver() {
		std::cout << "DFS edge parallel processing\n";
		std::cout << "Number of single-edge embeddings: " << edge_list.size() 
			<< ", staring from level " << starting_level << "\n";
		galois::do_all(galois::iterate(edge_list), [&](const SEdge &edge) {
			//std::cout << "Processing edge: " << edge.to_string() << "\n";
			EmbeddingListTy *emb_list = emb_lists.getLocal();
			if (is_single) {
				if (!degree_filter(edge.src, edge.dst)) {
					emb_list->init_edge(edge);
					if (use_ccode) { // TODO: use constexpr
						if (is_clique) { // TODO: use constexpr
							extend_clique(starting_level, *emb_list);
						} else {
							extend_single(starting_level, *emb_list);
							//extend_single_nc(starting_level, *emb_list);
							//ego_extend_sgl_naive(1, *emb_list);
						}
						if (!use_local_graph) emb_list->clear_labels(edge.src);
						if (!use_local_graph && !is_clique) emb_list->clear_labels(edge.dst);
					} else {
						extend_single_naive(starting_level, *emb_list);
					}
				}
			} else {
				emb_list->init_edge(edge);
				if (do_local_counting) {
					extend_multi_local(1, *emb_list);
					solve_motif_equations(*emb_list);
					if (this->max_size > 3) {
						emb_list->clear_labels(edge.dst);
						emb_list->clear_labels(edge.src);
					}
				} else {
					extend_multi(1, *emb_list);
					emb_list->clear_labels(edge.dst);
					emb_list->clear_labels(edge.src);
				}
			}
		}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("EdgeParallelSolver"));
		if (!is_single) motif_count();
	}

	//bool degree_filter(VertexId src, VertexId dst) {
	bool degree_filter(VertexId, VertexId) {
		//if (this->graph.get_degree(edge.src) >= this->max_size-1 
			//&& this->graph.get_degree(edge.dst) >= this->max_size-2) {
		//if (this->graph.get_degree(edge.src) < this->max_size-1 
		//	|| this->graph.get_degree.(edge.dst) < this->max_size-2) { 
		//	removed_edges += 1; return; }
		//std::cout << "\n\tremoved_edges = " << removed_edges.reduce() << "\n";
		return false;
	}
	///*
	// do not use ccode, need check the input graph for connectivity
	void extend_single_naive(unsigned level, EmbeddingListTy &emb_list) {
		if (level == this->max_size-2) {
			for (size_t emb_id = 0; emb_id < emb_list.size(level); emb_id ++) {
				auto vid = emb_list.get_vertex(level, emb_id);
				if (level > 1) emb_list.push_history(vid);
				auto begin = this->graph.edge_begin(vid);
				auto end = this->graph.edge_end(vid);
				for (auto e = begin; e < end; e ++) {
					auto dst = this->graph.getEdgeDst(e);
					if (this->is_all_connected_dag(dst, emb_list.get_history(), level))
						API::reduction(accumulators[0]);
				}
				if (level > 1) emb_list.pop_history();
			}
			return;
		}
		for (size_t emb_id = 0; emb_id < emb_list.size(level); emb_id ++) {
			auto vid = emb_list.get_vertex(level, emb_id);
			if (level > 1) emb_list.push_history(vid);
			auto begin = this->graph.edge_begin(vid);
			auto end = this->graph.edge_end(vid);
			emb_list.set_size(level+1, 0);
			for (auto e = begin; e < end; e ++) {
				auto dst = this->graph.getEdgeDst(e);
				if (this->is_all_connected_dag(dst, emb_list.get_history(), level)) {
					auto start = emb_list.size(level+1);
					emb_list.set_vid(level+1, start, dst);
					emb_list.set_size(level+1, start+1);
				}
			}
			extend_single_naive(level+1, emb_list);
			if (level > 1) emb_list.pop_history();
		}
	}
	//*/
	// DFS extension for k-cliques
	// it is easy to identify whether the input pattern is a clique,
	// and we automatically enable all optimizations 
	// when we find that the given pattern is a clique
	void extend_clique(unsigned level, EmbeddingListTy &emb_list) {
		if (level == this->max_size-2) {
			for (size_t emb_id = 0; emb_id < emb_list.size(level); emb_id ++) {
				auto vid = emb_list.get_vertex(level, emb_id);
				auto begin = emb_list.edge_begin(level, vid);
				auto end = emb_list.edge_end(level, vid);
				//auto begin = this->graph.edge_begin(vid);
				//auto end = this->graph.edge_end(vid);
				for (auto e = begin; e < end; e ++) {
					//auto dst = this->graph.getEdgeDst(e);
					auto dst = emb_list.getEdgeDst(e);
					auto ccode = emb_list.get_label(dst);
					if (API::toAdd(level, this->max_size, dst, level, ccode, NULL))
						API::reduction(accumulators[0]);
				}
			}
			return;
		}
		for (size_t emb_id = 0; emb_id < emb_list.size(level); emb_id ++) {
			auto vid = emb_list.get_vertex(level, emb_id);
			auto begin = emb_list.edge_begin(level, vid);
			auto end = emb_list.edge_end(level, vid);
			//auto begin = this->graph.edge_begin(vid);
			//auto end = this->graph.edge_end(vid);
			emb_list.set_size(level+1, 0);
			for (auto e = begin; e < end; e ++) {
				//auto dst = this->graph.getEdgeDst(e);
				auto dst = emb_list.getEdgeDst(e);
				auto ccode = emb_list.get_label(dst);
				if (API::toAdd(level, this->max_size, dst, level, ccode, NULL)) {
					auto start = emb_list.size(level+1);
					emb_list.set_vid(level+1, start, dst);
					emb_list.set_label(dst, level+1);
					emb_list.set_size(level+1, start+1);
					if (use_local_graph) emb_list.init_egonet_degree(level+1, dst);
				}
			}
			if (use_local_graph) emb_list.update_egonet(level);
			extend_clique(level+1, emb_list);
			emb_list.reset_labels(level);
		}
	}

	void extend_single(unsigned level, EmbeddingListTy &emb_list) {
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
					if (!API::toExtend(level, element_id, NULL)) continue;
					auto src = emb_list.get_history(element_id);
					auto begin = this->graph.edge_begin(src);
					auto end = this->graph.edge_end(src);
					for (auto e = begin; e < end; e ++) {
						auto dst = this->graph.getEdgeDst(e);
						auto ccode = emb_list.get_label(dst);
						if (API::toAdd(level, this->max_size, dst, element_id, 
								ccode, emb_list.get_history_ptr())) {
							unsigned pid = API::getPattern(level, this->max_size,
								src, dst, ccode, previous_pid, src_idx, NULL);
							//std::cout << "\t\t\t\t pcode=" << previous_pid << ", pid=" << pid << "\n";
							if (pid == input_pid)
								API::reduction(accumulators[0]);
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
				if (!API::toExtend(level, element_id, NULL)) continue;
				auto src = emb_list.get_history(element_id);
				auto begin = this->graph.edge_begin(src);
				auto end = this->graph.edge_end(src);
				for (auto edge = begin; edge < end; edge ++) {
					auto dst = this->graph.getEdgeDst(edge);
					auto ccode = emb_list.get_label(dst);
					if (API::toAdd(level, this->max_size, dst, element_id, 
							ccode, emb_list.get_history_ptr())) {
						auto start = emb_list.size(level+1);
						assert(start < this->max_degree);
						emb_list.set_vid(level+1, start, dst);
						emb_list.set_size(level+1, start+1);
						unsigned pid = API::getPattern(level, this->max_size,
								src, dst, ccode, previous_pid, element_id, NULL);
						//std::cout << "\t level=" << level << ", pcode=" << previous_pid << ", pid=" << pid << "\n";
						emb_list.set_pid(level+1, start, pid);
						emb_list.set_src(level+1, start, element_id);
					}
				}
			}
			extend_single(level+1, emb_list);
			if (level > 1) emb_list.pop_history();
		}
	}
	
	void extend_single_nc(unsigned level, EmbeddingListTy &emb_list) {
		if (level == this->max_size-2) {
			for (size_t emb_id = 0; emb_id < emb_list.size(level); emb_id++) {
				unsigned last_vid = 0;
				if (level > 1) {
					last_vid = emb_list.get_vertex(level, emb_id);
					emb_list.push_history(last_vid);
					emb_list.update_labels(level, last_vid);
				}
				//std::cout << "\t\t"; emb_list.print_history(); std::cout << "\n";
				auto src = emb_list.get_history(level);
				//#ifdef DIAMOND
				//for (size_t emb_id = 0; emb_id < emb_list.size(level); emb_id++) {
				//	auto dst = emb_list.get_vertex(level, emb_id);
				//#else
				auto begin = this->graph.edge_begin(src);
				auto end = this->graph.edge_end(src);
				for (auto e = begin; e < end; e ++) {
					auto dst = this->graph.getEdgeDst(e);
				//#endif
					auto ccode = emb_list.get_label(dst);
					//std::cout << "\t\t\t src=" << src << ", dst=" << dst << ", ccode=" << unsigned(ccode) << "\n";
					if (API::toAdd(level, this->max_size, dst, level, 
						ccode, emb_list.get_history_ptr())) {
						//std::cout << "\t\t\t\t found\n";
						API::reduction(accumulators[0]);
					}
				}
				if (level > 1) emb_list.resume_labels(level, last_vid);
				if (level > 1) emb_list.pop_history();
			}
			return;
		}
		for (size_t emb_id = 0; emb_id < emb_list.size(level); emb_id ++) {
			unsigned last_vid = 0;
			if (level > 1 || !edge_par) {
				last_vid = emb_list.get_vertex(level, emb_id);
				emb_list.push_history(last_vid);
				emb_list.update_labels(level, last_vid);
			}
			emb_list.set_size(level+1, 0);
			auto src = emb_list.get_history(level);
			//std::cout << "level=" << level << ", emb: "; emb_list.print_history();
			//std::cout << ", src=" << src << "\n";
			auto begin = this->graph.edge_begin(src);
			auto end = this->graph.edge_end(src);
			for (auto edge = begin; edge < end; edge ++) {
				auto dst = this->graph.getEdgeDst(edge);
				auto ccode = emb_list.get_label(dst);
				//std::cout << "\t dst=" << dst << ", ccode=" << unsigned(ccode) << "\n";
				if (API::toAdd(level, this->max_size, dst, level, 
						ccode, emb_list.get_history_ptr())) {
					auto start = emb_list.size(level+1);
					emb_list.set_vid(level+1, start, dst);
					emb_list.set_size(level+1, start+1);
				}
			}
			extend_single_nc(level+1, emb_list);
			if (level > 1 || !edge_par) emb_list.pop_history();
			if (!edge_par) emb_list.resume_labels(level, last_vid);
		}
	}
	/*
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
				unsigned previous_pid = 0, src_idx = 0;
				if (level > 1) previous_pid = emb_list.get_pid(level, emb_id);
				if (level > 1) src_idx = emb_list.get_src(level, emb_id);
				for (unsigned element_id = 0; element_id < level+1; ++ element_id) {
					if (!API::toExtend(level, element_id, NULL)) continue;
					auto src = emb_list.get_history(element_id);
					auto begin = this->graph.edge_begin(src);
					auto end = this->graph.edge_end(src);
					for (auto e = begin; e < end; e ++) {
						auto dst = this->graph.getEdgeDst(e);
						auto ccode = emb_list.get_label(dst);
						if (API::toAdd(level, this->max_size, dst, element_id, 
								ccode, emb_list.get_history_ptr())) {
							unsigned pid = API::getPattern(level, this->max_size,
								src, dst, ccode, previous_pid, src_idx, NULL);
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
			unsigned previous_pid = 0;
			if (level > 1) previous_pid = emb_list.get_pid(level, emb_id);
			emb_list.set_size(level+1, 0);
			for (unsigned element_id = 0; element_id < level+1; ++ element_id) {
				if (!API::toExtend(level, element_id, NULL)) continue;
				auto src = emb_list.get_history(element_id);
				auto begin = this->graph.edge_begin(src);
				auto end = this->graph.edge_end(src);
				for (auto edge = begin; edge < end; edge ++) {
					auto dst = this->graph.getEdgeDst(edge);
					auto ccode = emb_list.get_label(dst);
					if (API::toAdd(level, this->max_size, dst, element_id, 
							ccode, emb_list.get_history_ptr())) {
						auto start = emb_list.size(level+1);
						assert(start < this->max_degree);
						emb_list.set_vid(level+1, start, dst);
						emb_list.set_size(level+1, start+1);
						unsigned pid = API::getPattern(level, this->max_size,
							src, dst, ccode, previous_pid, element_id, NULL);
						emb_list.set_pid(level+1, start, pid);
						emb_list.set_src(level+1, start, element_id);
					}
				}
			}
			extend_multi(level+1, emb_list);
			if (level > 1) emb_list.pop_history();
		}
	}

	void extend_multi_local(unsigned level, EmbeddingListTy &emb_list) {
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
					if (API::toAdd(level, this->max_size, dst, level, 
							ccode, emb_list.get_history_ptr())) {
						unsigned pid = API::getPattern(level, this->max_size, 
							src, dst, ccode, previous_pid, level, NULL);
						API::local_reduction(level, pid, emb_list.local_counters[pid]);
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
			auto begin = this->graph.edge_begin(src);
			auto end = this->graph.edge_end(src);
			for (auto e = begin; e < end; e ++) {
				auto dst = this->graph.getEdgeDst(e);
				auto ccode = emb_list.get_label(dst);
				if (API::toAdd(level, this->max_size, dst, level, 
						ccode, emb_list.get_history_ptr())) {
					auto start = emb_list.size(level+1);
					emb_list.set_vid(level+1, start, dst);
					emb_list.set_size(level+1, start+1);
					unsigned pid = API::getPattern(level, this->max_size,
							src, dst, ccode, previous_pid, level, NULL);
					emb_list.set_pid(level+1, start, pid);
					API::local_reduction(level, pid, emb_list.local_counters[pid]);
				}
			}
			extend_multi_local(level+1, emb_list);
			if (level > 1) emb_list.pop_history();
		}
	}

	inline void extend_ordered(unsigned level, EmbeddingListTy &emb_list) {
		unsigned n = level + 1;
		unsigned last_vid = 0;
		if (n == this->max_size-1) {
			//std::cout << "\t\t level " << level << ": "; 
			//emb_list.print_history(); std::cout << "\n";
			if (level > 1) {
				last_vid = emb_list.get_history(level);
				//emb_list.push_history(last_vid);
				emb_list.update_labels(level, last_vid);
			}
			for (auto q_edge : this->pattern.edges(n)) {
				VertexId q_dst = this->pattern.getEdgeDst(q_edge);
				unsigned q_order = q_dst;
				//std::cout << "\t\t order=" << q_order << "\n";
				if (q_dst < n) {
					auto d_src = emb_list.get_history(q_order);
					//std::cout << "\t\t src=" << d_src << "\n";
					for (auto d_edge : this->graph.edges(d_src)) {
						auto d_dst = this->graph.getEdgeDst(d_edge);
						auto ccode = emb_list.get_label(d_dst);
						//std::cout << "\t\t dst=" << d_dst << ", ccode=" << unsigned(ccode) << "\n";
						if (this->graph.get_degree(d_dst) < this->pattern.get_degree(n)) continue;
						if (API::toAdd(level, this->max_size, d_dst, q_order, 
								ccode, emb_list.get_history_ptr())) {
							//std::cout << "\t\t\t subgraph macthed: dst=" << d_dst << "\n";
							API::reduction(accumulators[0]);
						}
					}
					break;
				}
			}
			if (level > 1) emb_list.resume_labels(level, last_vid);
			if (level > 1) emb_list.pop_history();
			return;
		}
		for (auto q_edge : this->pattern.edges(n)) {
			VertexId q_dst = this->pattern.getEdgeDst(q_edge);
			unsigned q_order = q_dst; //get_matching_order(q_dst); // using query vertex id to get its matching order
			//std::cout << "level " << level << ": "; 
			//emb_list.print_history();
			//std::cout << ", order=" << q_order << "\n";
			if (q_order < n) {
				auto d_src = emb_list.get_history(q_order);
				//std::cout << "src=" << d_src << "\n";
				//emb_list.set_size(level+1, 0);
				for (auto d_edge : this->graph.edges(d_src)) {
					auto d_dst = this->graph.getEdgeDst(d_edge);
					auto ccode = emb_list.get_label(d_dst);
					//std::cout << "\t dst=" << d_dst << ", ccode=" << unsigned(ccode) << "\n";
					if (API::toAdd(level, this->max_size, d_dst, q_order, 
							ccode, emb_list.get_history_ptr())) {
						//auto start = emb_list.size(level+1);
						//emb_list.set_vid(level+1, start, d_dst);
						//emb_list.set_idx(level+1, start, pos);
						//emb_list.set_size(level+1, start+1);
						//std::cout << "\t pushing vertex " << d_dst << " to stack\n";
						emb_list.push_history(d_dst);
						extend_ordered(level+1, emb_list);
						if (level > 1) emb_list.pop_history();
					}
				}
				break;
			}
		}
		//extend_ordered(level+1, emb_list);
		//if (level > 1) emb_list.pop_history();
	}

	// compute global counts using user-defined formula
	// TODO: make it a virtual function to be override by the user
	void solve_motif_equations(EmbeddingListTy &emb_list) {
		VertexId u = emb_list.get_vid(0, 0);
		VertexId v = emb_list.get_vid(1, 0);
		Ulong tri_count = emb_list.local_counters[0];
		unsigned deg_v = this->graph.get_degree(v);
		unsigned deg_u = this->graph.get_degree(u);
		if (this->max_size == 3) {
			accumulators[0] += tri_count;
			accumulators[1] += deg_v - tri_count - 1 + deg_u - tri_count - 1;
		} else if (this->max_size == 4) {
			accumulators[2] += emb_list.local_counters[2]; // 4-cycle
			accumulators[5] += emb_list.local_counters[5]; // 4-clique
			Ulong star3_count = (deg_v - tri_count - 1) + (deg_u - tri_count - 1);
			accumulators[4] += (tri_count * (tri_count - 1) / 2); // diamond
			accumulators[0] += tri_count * star3_count; // tailed_triangles
			accumulators[3] += (deg_v - tri_count - 1) * (deg_u - tri_count - 1); // 4-path
			accumulators[1] += (deg_v - tri_count - 1) * (deg_v - tri_count - 2) / 2; // 3-star
			accumulators[1] += (deg_u - tri_count - 1) * (deg_u - tri_count - 2) / 2;
		} else { 
			exit(1);
		}
	}

	void motif_count() {
		if (do_local_counting) {
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
		} else {
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
		}
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
	int npatterns;           // number of patterns; 1 for single-pattern problem
	bool is_directed;        // is the input graph a directed graph (true for a DAG)
	unsigned core;           // the core value of the input graph; for estimating and pre-allocating memory
	unsigned nedges_pattern; // number of edges in the pattern (used for single-pattern problem only)
	unsigned starting_level; // in which level the search starts from 
	unsigned input_pid;
	UlongAccu removed_edges; // number of edges removed by degree filtering
	EmbeddingLists emb_lists;// the embedding lists; one list for each thread
	EdgeList edge_list;      // the edge list to hold the single-edge embeddings
	std::vector<UlongAccu> accumulators; // counters for counting embeddings of each pattern

	// dedicated counters for 4-motifs
	Ulong total_3_tris;
	Ulong total_3_path;
	Ulong total_4_clique;
	Ulong total_4_diamond;
	Ulong total_4_tailed_tris;
	Ulong total_4_cycle;
	Ulong total_3_star;
	Ulong total_4_path;

	// generated functions for subgraph counting/listing 
	// they are currently writtern by hand. Should be easy to generated automatically
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

