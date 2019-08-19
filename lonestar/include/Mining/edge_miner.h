#ifndef EDGE_MINER_H
#define EDGE_MINER_H
#include "miner.h"
#include "domain_support.h"

typedef std::pair<unsigned, unsigned> InitPattern;
typedef QuickPattern<EdgeEmbedding, ElementType> QPattern;
typedef CanonicalGraph<EdgeEmbedding, ElementType> CPattern;
///*
typedef std::unordered_map<QPattern, Frequency> QpMapFreq; // quick pattern map (mapping quick pattern to its frequency)
typedef std::unordered_map<CPattern, Frequency> CgMapFreq; // canonical pattern map (mapping canonical pattern to its frequency)

typedef std::map<InitPattern, DomainSupport*> InitMap;
typedef std::unordered_map<QPattern, DomainSupport*> QpMapDomain; // quick pattern map (mapping quick pattern to its domain support)
typedef std::unordered_map<CPattern, DomainSupport*> CgMapDomain; // canonical pattern map (mapping canonical pattern to its domain support)

typedef std::unordered_map<unsigned, unsigned> FreqMap;
typedef std::unordered_map<unsigned, bool> DomainMap;
//*/
/*
typedef galois::gstl::Map<InitPattern, DomainSupport> InitMap;
//typedef galois::gstl::Map<QPattern, Frequency> QpMapFreq; // mapping quick pattern to its frequency
//typedef galois::gstl::Map<CPattern, Frequency> CgMapFreq; // mapping canonical pattern to its frequency
typedef galois::gstl::UnorderedMap<QPattern, DomainSupport> QpMapDomain; // mapping quick pattern to its domain support
typedef galois::gstl::UnorderedMap<CPattern, DomainSupport> CgMapDomain; // mapping canonical pattern to its domain support
//typedef galois::gstl::Map<unsigned, unsigned> FreqMap;
typedef galois::gstl::UnorderedMap<unsigned, bool> DomainMap;
//*/
typedef galois::substrate::PerThreadStorage<InitMap> LocalInitMap;
typedef galois::substrate::PerThreadStorage<QpMapFreq> LocalQpMapFreq; // PerThreadStorage: thread-local quick pattern map
typedef galois::substrate::PerThreadStorage<CgMapFreq> LocalCgMapFreq; // PerThreadStorage: thread-local canonical pattern map
typedef galois::substrate::PerThreadStorage<QpMapDomain> LocalQpMapDomain;
typedef galois::substrate::PerThreadStorage<CgMapDomain> LocalCgMapDomain;

class EdgeMiner : public Miner {
public:
	EdgeMiner(Graph *g) { graph = g; construct_edgemap(); }
	virtual ~EdgeMiner() {}
	// given an embedding, extend it with one more edge, and if it is not automorphism, insert the new embedding into the task queue
	void extend_edge(EdgeEmbeddingQueue &in_queue, EdgeEmbeddingQueue &out_queue) {
		//if (show) std::cout << "\n----------------------------- Extending -----------------------------\n";
		if (show) std::cout << "\n------------------------- Step 1: Extending -------------------------\n";
		// for each embedding in the worklist, do the edge-extension operation
		galois::do_all(galois::iterate(in_queue),
			[&](const EmbeddingType& emb) {
				unsigned n = emb.size();
				// get the number of distinct vertices in the embedding
				VertexSet vert_set;
				if (n > 3)
					for (unsigned i = 0; i < n; i ++) vert_set.insert(emb.get_vertex(i));
				// for each vertex in the embedding
				for (unsigned i = 0; i < n; ++i) {
					VertexId src = emb.get_vertex(i);
					// make sure each distinct vertex is extended only once
					if (emb.get_key(i) == 0) {
						// try edge extension
						for (auto e : graph->edges(src)) {
							GNode dst = graph->getEdgeDst(e);
							BYTE existed = 0;
							#ifdef ENABLE_LABEL
							if (is_frequent_edge[*e])
							#endif
								// check if this is automorphism
								if (!is_edge_automorphism(n, emb, i, src, dst, existed, vert_set)) {
									auto dst_label = 0, edge_label = 0;
									#ifdef ENABLE_LABEL
									dst_label = graph->getData(dst);
									//edge_label = graph->getEdgeData(e); // TODO: enable this for FSM
									#endif
									ElementType new_element(dst, (BYTE)existed, edge_label, dst_label, (BYTE)i);
									EdgeEmbedding new_emb(emb);
									new_emb.push_back(new_element);
									// insert the new (extended) embedding into the worklist
									out_queue.push_back(new_emb);
								}
						}
					}
				}
			},
			galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::no_conflicts(),
			galois::wl<galois::worklists::PerSocketChunkFIFO<CHUNK_SIZE>>(),
			galois::loopname("Extending")
		);
	}
	void extend_edge(unsigned level, EmbeddingList& emb_list) {
		if (show) std::cout << "\n----------------------------- Extending -----------------------------\n";
		//if (show) std::cout << "\n------------------------- Step 1: Extending -------------------------\n";
		UintList num_new_emb(emb_list.size());
		galois::do_all(galois::iterate((size_t)0, emb_list.size()),
			[&](const size_t& pos) {
				EdgeEmbedding emb(level+1);
				get_embedding(level, pos, emb_list, emb);
				num_new_emb[pos] = 0;
				unsigned n = emb.size();
				VertexSet vert_set;
				if (n > 3)
					for (unsigned i = 0; i < n; i ++) vert_set.insert(emb.get_vertex(i));
				for (unsigned i = 0; i < n; ++i) {
					VertexId src = emb.get_vertex(i);
					if (emb.get_key(i) == 0) { // TODO: need to fix this
						for (auto e : graph->edges(src)) {
							GNode dst = graph->getEdgeDst(e);
							BYTE existed = 0;
							if (is_frequent_edge[*e])
								if (!is_edge_automorphism(n, emb, i, src, dst, existed, vert_set))
									num_new_emb[pos] ++;
						}
					}
				}
				emb.clean();
			},
			galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::no_conflicts(),
			galois::wl<galois::worklists::PerSocketChunkFIFO<CHUNK_SIZE>>(),
			galois::loopname("Extending-alloc")
		);
		Ulong new_size = std::accumulate(num_new_emb.begin(), num_new_emb.end(), (Ulong)0);
		if (show) std::cout << "new_size = " << new_size << "\n";
		assert(new_size < 4294967296); // TODO: currently do not support vector size larger than 2^32
		UintList indices = parallel_prefix_sum(num_new_emb);
		new_size = indices[indices.size()-1];
		emb_list.add_level(new_size);
		galois::do_all(galois::iterate((size_t)0, emb_list.size(level)),
			[&](const size_t& pos) {
				EdgeEmbedding emb(level+1);
				get_embedding(level, pos, emb_list, emb);
				unsigned start = indices[pos];
				unsigned n = emb.size();
				VertexSet vert_set;
				if (n > 3)
					for (unsigned i = 0; i < n; i ++) vert_set.insert(emb.get_vertex(i));
				for (unsigned i = 0; i < n; ++i) {
					VertexId src = emb.get_vertex(i);
					if (emb.get_key(i) == 0) {
						for (auto e : graph->edges(src)) {
							GNode dst = graph->getEdgeDst(e);
							BYTE existed = 0;
							if (is_frequent_edge[*e])
								if (!is_edge_automorphism(n, emb, i, src, dst, existed, vert_set)) {
									emb_list.set_idx(level+1, start, pos);
									emb_list.set_his(level+1, start, i);
									emb_list.set_vid(level+1, start++, dst);
								}
						}
					}
				}
			},
			galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::no_conflicts(),
			galois::wl<galois::worklists::PerSocketChunkFIFO<CHUNK_SIZE>>(),
			galois::loopname("Extending-write")
		);
	}
	inline unsigned init_aggregator() {
		init_map.clear();
		/*for (auto src : *graph) {
			auto& src_label = graph->getData(src);
			for (auto e : graph->edges(src)) {
				GNode dst = graph->getEdgeDst(e);
				//auto elabel = graph.getEdgeData(e);
				auto& dst_label = graph->getData(dst);
				if (src_label <= dst_label) {
					initpattern key = get_init_pattern(src_label, dst_label);
					if (init_map.find(key) == init_map.end()) {
						init_map[key].first.resize(2);
						std::fill(init_map[key].first.begin(), init_map[key].first.end(), 0);
						init_map[key].second.resize(2);
					}
					init_map[key].second[0].insert(src);
					init_map[key].second[1].insert(dst);
					for (unsigned i = 0; i < 2; i ++) {
						if (init_map[key].second[i].size() >= threshold) {
							init_map[key].first[i] = 1;
							init_map[key].second[i].clear();
						}
					}
				}
			}
		}*/
		galois::do_all(galois::iterate(graph->begin(), graph->end()),
			[&](const GNode& src) {
				InitMap *lmap = init_localmaps.getLocal();
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
			galois::wl<galois::worklists::PerSocketChunkFIFO<CHUNK_SIZE>>(),
			galois::loopname("InitAggregation")
		);
		merge_init_map();
		std::cout << "Number of single-edge patterns: " << init_map.size() << "\n";
		unsigned count = 0;
		for (auto it = init_map.begin(); it != init_map.end(); ++it)
			if (it->second->get_support()) count ++;
		return count; // return number of frequent single-edge patterns
	}
	// aggregate embeddings into quick patterns
	inline void quick_aggregate_each(const EdgeEmbedding& emb, QpMapFreq& qp_map) {
		// turn this embedding into its quick pattern
		QPattern qp(emb);
		// update frequency for this quick pattern
		if (qp_map.find(qp) != qp_map.end()) {
			// if this quick pattern already exists, increase its count
			qp_map[qp] += 1;
			qp.clean();
		// otherwise add this quick pattern into the map, and set the count as one
		} else qp_map[qp] = 1;
	}
	inline void quick_aggregate(EdgeEmbeddingQueue& queue) {
		//if (show) std::cout << "\n---------------------------- Aggregating ----------------------------\n";
		if (show) std::cout << "\n------------------------ Step 2: Aggregating ------------------------\n";
		for (auto i = 0; i < numThreads; i++) qp_localmaps.getLocal(i)->clear();
		galois::do_all(galois::iterate(queue),
			[&](EmbeddingType &emb) {
				QpMapDomain *lmap = qp_localmaps.getLocal();
				unsigned n = emb.size();
				QPattern qp(emb);
				bool qp_existed = false;
				auto it = lmap->find(qp);
				if (it == lmap->end()) {
					(*lmap)[qp] = new DomainSupport(n);
					(*lmap)[qp]->set_threshold(threshold);
					emb.set_qpid(qp.get_id());
				} else {
					qp_existed = true;
					emb.set_qpid((it->first).get_id());
				}
				DomainSupport *support = (*lmap)[qp];
				for (unsigned i = 0; i < n; i ++) {
					if (support->has_domain_reached_support(i) == false)
						support->add_vertex(i, emb.get_vertex(i));
				}
				if (qp_existed) qp.clean();
			},
			galois::chunk_size<CHUNK_SIZE>(), galois::steal(),
			galois::no_conflicts(), galois::wl<galois::worklists::PerSocketChunkFIFO<CHUNK_SIZE>>(),
			galois::loopname("QuickAggregation")
		);
	}
	inline void quick_aggregate(unsigned level, EmbeddingList& emb_list) {
		//if (show) std::cout << "\n---------------------------- Aggregating ----------------------------\n";
		if (show) std::cout << "\n------------------------ Step 2: Aggregating ------------------------\n";
		for (auto i = 0; i < numThreads; i++) qp_localmaps.getLocal(i)->clear();
		galois::do_all(galois::iterate((size_t)0, emb_list.size()),
			[&](const size_t& pos) {
				QpMapDomain *lmap = qp_localmaps.getLocal();
				EdgeEmbedding emb(level+1);
				get_embedding(level, pos, emb_list, emb);
				unsigned n = emb.size();
				QPattern qp(emb, true);
				bool qp_existed = false;
				auto it = lmap->find(qp);
				if (it == lmap->end()) {
					(*lmap)[qp] = new DomainSupport(n);
					(*lmap)[qp]->set_threshold(threshold);
					emb_list.set_pid(pos, qp.get_id());
				} else {
					qp_existed = true;
					emb_list.set_pid(pos, (it->first).get_id());
				}
				for (unsigned i = 0; i < n; i ++) {
					if ((*lmap)[qp]->has_domain_reached_support(i) == false)
						(*lmap)[qp]->add_vertex(i, emb.get_vertex(i));
				}
				if (qp_existed) qp.clean();
			},
			galois::chunk_size<CHUNK_SIZE>(), galois::steal(),
			galois::no_conflicts(), galois::wl<galois::worklists::PerSocketChunkFIFO<CHUNK_SIZE>>(),
			galois::loopname("QuickAggregation")
		);
	}
	inline void canonical_aggregate_each(std::pair<QPattern, Frequency> element, CgMapFreq &cg_map) {
		// turn the quick pattern into its canonical pattern
		CPattern cg(element.first);
		element.first.clean();
		// if this pattern already exists, increase its count
		if (cg_map.find(cg) != cg_map.end()) cg_map[cg] += element.second;
		// otherwise add this pattern into the map, and set the count as 'freq'
		else cg_map[cg] = element.second;
		cg.clean();
	}
	// aggregate quick patterns into canonical patterns.
	// construct id_map from quick pattern ID (qp_id) to canonical pattern ID (cg_id)
	void canonical_aggregate() {
		id_map.clear();
		for (auto i = 0; i < numThreads; i++) cg_localmaps.getLocal(i)->clear();
		galois::do_all(galois::iterate(qp_map),
			[&](std::pair<QPattern, DomainSupport*> element) {
				CgMapDomain *lmap = cg_localmaps.getLocal();
				unsigned num_domains = element.first.get_size();
				CPattern cg(element.first);
				int qp_id = element.first.get_id();
				int cg_id = cg.get_id();
				slock.lock();
				id_map.insert(std::make_pair(qp_id, cg_id));
				slock.unlock();
				auto it = lmap->find(cg);
				if (it == lmap->end()) {
					(*lmap)[cg] = new DomainSupport(num_domains);
					(*lmap)[cg]->set_threshold(threshold);
					element.first.set_cgid(cg.get_id());
				} else {
					element.first.set_cgid((it->first).get_id());
				}
				VertexPositionEquivalences equivalences;
				element.first.get_equivalences(equivalences);
				for (unsigned i = 0; i < num_domains; i ++) {
					if ((*lmap)[cg]->has_domain_reached_support(i) == false) {
						unsigned qp_idx = cg.get_quick_pattern_index(i);
						assert(qp_idx >= 0 && qp_idx < num_domains);
						UintSet equ_set = equivalences.get_equivalent_set(qp_idx);
						for (unsigned idx : equ_set) {
							DomainSupport *support = element.second;
							if (support->has_domain_reached_support(idx) == false) {
								bool reached_threshold = (*lmap)[cg]->add_vertices(i, support->domain_sets[idx]);
								if (reached_threshold) break;
							} else {
								(*lmap)[cg]->set_domain_frequent(i);
								break;
							}
						}
					}
				}
				cg.clean();
			},
			galois::chunk_size<CHUNK_SIZE>(), galois::steal(),
			galois::no_conflicts(), galois::wl<galois::worklists::PerSocketChunkFIFO<CHUNK_SIZE>>(),
			galois::loopname("CanonicalAggregation")
		);
	}
	inline void merge_qp_map(LocalQpMapFreq &qp_localmap, QpMapFreq &qp_map) {
		for (auto i = 0; i < numThreads; i++) {
			for (auto element : *qp_localmap.getLocal(i)) {
				if (qp_map.find(element.first) != qp_map.end())
					qp_map[element.first] += element.second;
				else qp_map[element.first] = element.second;
			}
		}
	}
	inline void merge_cg_map(LocalCgMapFreq &localmaps, CgMapFreq &cg_map) {
		for (auto i = 0; i < numThreads; i++) {
			for (auto element : *localmaps.getLocal(i)) {
				if (cg_map.find(element.first) != cg_map.end())
					cg_map[element.first] += element.second;
				else cg_map[element.first] = element.second;
			}
		}
	}
	inline void merge_init_map() {
		init_map = *(init_localmaps.getLocal(0));
		for (auto i = 1; i < numThreads; i++) {
			for (auto element : *init_localmaps.getLocal(i)) {
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
	}
	inline void merge_qp_map(unsigned num_domains) {
		qp_map.clear();
		qp_map = *(qp_localmaps.getLocal(0));
		for (auto i = 1; i < numThreads; i++) {
			const QpMapDomain *lmap = qp_localmaps.getLocal(i);
			for (auto element : *lmap) {
				if (qp_map.find(element.first) == qp_map.end())
					qp_map[element.first] = element.second;
			}
			galois::do_all(galois::iterate(*lmap),
				[&](std::pair<QPattern, DomainSupport*> element) {
					DomainSupport *support = element.second;
					for (unsigned i = 0; i < num_domains; i ++) {
						if (!qp_map[element.first]->has_domain_reached_support(i) && qp_map[element.first] != support) {
							if (support->has_domain_reached_support(i))
								qp_map[element.first]->set_domain_frequent(i);
							else qp_map[element.first]->add_vertices(i, support->domain_sets[i]);
						}
					}
				},
				galois::chunk_size<CHUNK_SIZE>(), galois::steal(),
				galois::no_conflicts(), galois::wl<galois::worklists::PerSocketChunkFIFO<CHUNK_SIZE>>(),
				galois::loopname("MergeQuickPatterns")
			);
		}
	}
	inline void merge_cg_map(unsigned num_domains) {
		cg_map.clear();
		cg_map = *(cg_localmaps.getLocal(0));
		for (auto i = 1; i < numThreads; i++) {
			const CgMapDomain *lmap = cg_localmaps.getLocal(i);
			for (auto element : *lmap) {
				if (cg_map.find(element.first) == cg_map.end())
					cg_map[element.first] = element.second;
			}
			galois::do_all(galois::iterate(*lmap),
				[&](std::pair<CPattern, DomainSupport*> element) {
					DomainSupport *support = element.second;
					for (unsigned i = 0; i < num_domains; i ++) {
						if (!cg_map[element.first]->has_domain_reached_support(i) && cg_map[element.first] != support) {
							if (support->has_domain_reached_support(i))
								cg_map[element.first]->set_domain_frequent(i);
							else cg_map[element.first]->add_vertices(i, support->domain_sets[i]);
						}
					}
				},
				galois::chunk_size<CHUNK_SIZE>(), galois::steal(),
				galois::no_conflicts(), galois::wl<galois::worklists::PerSocketChunkFIFO<CHUNK_SIZE>>(),
				galois::loopname("MergeCanonicalPatterns")
			);
		}
	}

	// Filtering for FSM
#ifdef ENABLE_LABEL
	inline void init_filter(EdgeEmbeddingQueue &out_queue) {
		is_frequent_edge.resize(graph->sizeEdges());
		std::fill(is_frequent_edge.begin(), is_frequent_edge.end(), 0);
		galois::do_all(galois::iterate(graph->begin(), graph->end()),
			[&](const GNode& src) {
				auto& src_label = graph->getData(src);
				for (auto e : graph->edges(src)) {
					GNode dst = graph->getEdgeDst(e);
					if(src < dst) {
						auto& dst_label = graph->getData(dst);
						InitPattern key = get_init_pattern(src_label, dst_label);
						if (init_map[key]->get_support()) { // check the support of this pattern
							EdgeEmbedding new_emb;
							new_emb.push_back(ElementType(src, 0, src_label));
							new_emb.push_back(ElementType(dst, 0, dst_label));
							out_queue.push_back(new_emb);
							unsigned eid = edge_map[OrderedEdge(dst,src)];
							__sync_bool_compare_and_swap(&is_frequent_edge[*e], 0, 1);
							__sync_bool_compare_and_swap(&is_frequent_edge[eid], 0, 1);
							slock.unlock();
						}
					}
				}
			},
			galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::no_conflicts(),
			galois::wl<galois::worklists::PerSocketChunkFIFO<CHUNK_SIZE>>(),
			galois::loopname("InitFilter")
		);
		std::cout << "Number of frequent edges: " << count(is_frequent_edge.begin(), is_frequent_edge.end(), 1) << "\n";
	}
	inline void init_filter(EmbeddingList& emb_list) {
		UintList is_frequent_emb(emb_list.size(), 0);
		galois::do_all(galois::iterate((size_t)0, emb_list.size()),
			[&](const size_t& pos) {
				VertexId src = emb_list.get_idx(1, pos);
				VertexId dst = emb_list.get_vid(1, pos);
				auto& src_label = graph->getData(src);
				auto& dst_label = graph->getData(dst);
				InitPattern key = get_init_pattern(src_label, dst_label);
				if (init_map[key]->get_support()) is_frequent_emb[pos] = 1;
			},
			galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::no_conflicts(),
			galois::wl<galois::worklists::PerSocketChunkFIFO<CHUNK_SIZE>>(),
			galois::loopname("InitFilter")
		);

		assert(emb_list.size()*2 == graph->sizeEdges()); // symmetric graph
		is_frequent_edge.resize(graph->sizeEdges());
		std::fill(is_frequent_edge.begin(), is_frequent_edge.end(), 0);
		galois::do_all(galois::iterate((size_t)0, emb_list.size()),
			[&](const size_t& pos) {
				if (is_frequent_emb[pos]) {
					VertexId src = emb_list.get_idx(1, pos);
					VertexId dst = emb_list.get_vid(1, pos);
					unsigned eid0 = edge_map[OrderedEdge(src,dst)];
					unsigned eid1 = edge_map[OrderedEdge(dst,src)];
					__sync_bool_compare_and_swap(&is_frequent_edge[eid0], 0, 1);
					__sync_bool_compare_and_swap(&is_frequent_edge[eid1], 0, 1);
				}
			},
			galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::no_conflicts(),
			galois::wl<galois::worklists::PerSocketChunkFIFO<CHUNK_SIZE>>(),
			galois::loopname("InitFrquentEdges")
		);
		std::cout << "Number of frequent edges: " << count(is_frequent_edge.begin(), is_frequent_edge.end(), 1) << "\n";
	
		UintList indices = parallel_prefix_sum(is_frequent_emb);
		//VertexList vid_list0 = emb_list.get_vid_list(0);
		VertexList vid_list0 = emb_list.get_idx_list(1);
		VertexList vid_list1 = emb_list.get_vid_list(1);
		galois::do_all(galois::iterate((size_t)0, emb_list.size()),
			[&](const size_t& pos) {
				if (is_frequent_emb[pos]) {
					VertexId src = vid_list0[pos];
					VertexId dst = vid_list1[pos];
					unsigned start = indices[pos];
					//emb_list.set_vid(0, start, src);
					emb_list.set_vid(1, start, dst);
					//emb_list.set_idx(1, start, start);
					emb_list.set_idx(1, start, src);
				}
			},
			galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::no_conflicts(),
			galois::wl<galois::worklists::PerSocketChunkFIFO<CHUNK_SIZE>>(),
			galois::loopname("InitEmbeddingList")
		);
		emb_list.remove_tail(indices.back());
	}
#endif
	// Check if the pattern of a given embedding is frequent, if yes, insert it to the queue
	inline void filter(EdgeEmbeddingQueue &in_queue, EdgeEmbeddingQueue &out_queue) {
		if (show) std::cout << "\n-------------------------- Step 3: Filter ---------------------------\n";
		galois::do_all(galois::iterate(in_queue),
			[&](const EdgeEmbedding &emb) {
				unsigned qp_id = emb.get_qpid();
				unsigned cg_id = id_map.at(qp_id);
				if (domain_support_map.at(cg_id)) out_queue.push_back(emb);
			},
			galois::chunk_size<CHUNK_SIZE>(), galois::steal(), 
			galois::no_conflicts(), galois::wl<galois::worklists::PerSocketChunkFIFO<CHUNK_SIZE>>(),
			galois::loopname("Filter")
		);
	}
	inline void filter(unsigned level, EmbeddingList &emb_list) {
		if (show) std::cout << "\n-------------------------- Step 3: Filter ---------------------------\n";
		UintList is_frequent_emb(emb_list.size(), 0);
		galois::do_all(galois::iterate((size_t)0, emb_list.size()),
			[&](const size_t& pos) {
				unsigned qp_id = emb_list.get_pid(pos);
				unsigned cg_id = id_map.at(qp_id);
				if (domain_support_map.at(cg_id))
					is_frequent_emb[pos] = 1;
			},
			galois::chunk_size<CHUNK_SIZE>(), galois::steal(), 
			galois::no_conflicts(), galois::wl<galois::worklists::PerSocketChunkFIFO<CHUNK_SIZE>>(),
			galois::loopname("Filter-alloc")
		);
		UintList indices = parallel_prefix_sum(is_frequent_emb);
		VertexList vid_list = emb_list.get_vid_list(level);
		UintList idx_list = emb_list.get_idx_list(level);
		ByteList his_list = emb_list.get_his_list(level);
		galois::do_all(galois::iterate((size_t)0, emb_list.size()),
			[&](const size_t& pos) {
				if (is_frequent_emb[pos]) {
					unsigned start = indices[pos];
					VertexId vid = vid_list[pos];
					IndexTy idx = idx_list[pos];
					BYTE his = his_list[pos];
					emb_list.set_idx(level, start, idx);
					emb_list.set_vid(level, start, vid);
					emb_list.set_his(level, start, his);
				}
			},
			galois::chunk_size<CHUNK_SIZE>(), galois::steal(), 
			galois::no_conflicts(), galois::wl<galois::worklists::PerSocketChunkFIFO<CHUNK_SIZE>>(),
			galois::loopname("Filter-write")
		);
		emb_list.remove_tail(indices.back());
	}
	inline void set_threshold(const unsigned minsup) { threshold = minsup; }
	inline void printout_agg(const CgMapFreq &cg_map) {
		for (auto it = cg_map.begin(); it != cg_map.end(); ++it)
			std::cout << "{" << it->first << " --> " << it->second << std::endl;
	}
	inline void printout_agg() {
		if (show) std::cout << "num_patterns: " << cg_map.size() << " num_quick_patterns: " << qp_map.size() << "\n";
		BoolVec support(cg_map.size());
		int i = 0;
		for (auto it = cg_map.begin(); it != cg_map.end(); ++it) {
			support[i] = it->second->get_support();
			i ++;
		}
		i = 0;
		for (auto it = cg_map.begin(); it != cg_map.end(); ++it) {
			std::cout << "{" << it->first << " --> " << support[i] << std::endl;
			i ++;
		}
	}
	inline unsigned support_count() {
		domain_support_map.clear();
		unsigned count = 0;
		for (auto it = cg_map.begin(); it != cg_map.end(); ++it) {
			bool support = it->second->get_support();
			domain_support_map.insert(std::make_pair(it->first.get_id(), support));
			if (support) count ++;
		}
		return count;
	}
	// construct edge-map for later use. May not be necessary if Galois has this support
	void construct_edgemap() {
		for (auto src : *graph) {
			for (auto e : graph->edges(src)) {
				GNode dst = graph->getEdgeDst(e);
				OrderedEdge edge(src, dst);
				edge_map.insert(std::pair<OrderedEdge, unsigned>(edge, *e));
			}
		}
	}

private:
	unsigned threshold;
	InitMap init_map;
	UintMap id_map;
	FreqMap freq_support_map;
	DomainMap domain_support_map;
	galois::gstl::Map<OrderedEdge, unsigned> edge_map;
	std::set<std::pair<VertexId,VertexId> > freq_edge_set;
	std::vector<unsigned> is_frequent_edge;
	LocalInitMap init_localmaps; // initialization map, only used for once, no need to clear
	LocalQpMapDomain qp_localmaps; // quick pattern local map for each thread
	LocalCgMapDomain cg_localmaps; // canonical pattern local map for each thread
	QpMapDomain qp_map; // quick pattern map
	CgMapDomain cg_map; // canonical graph map
	galois::substrate::SimpleLock slock;

	inline InitPattern get_init_pattern(BYTE src_label, BYTE dst_label) {
		if (src_label <= dst_label) return std::make_pair(src_label, dst_label);
		else return std::make_pair(dst_label, src_label);
	}
	inline void get_embedding(unsigned level, unsigned pos, const EmbeddingList& emb_list, EdgeEmbedding &emb) {
		VertexId vid = emb_list.get_vid(level, pos);
		IndexTy idx = emb_list.get_idx(level, pos);
		BYTE his = emb_list.get_his(level, pos);
		BYTE lab = graph->getData(vid);
		//emb.set_element(level, ElementType(vid, 0, lab, his));
		ElementType ele(vid, 0, lab, his);
		emb.set_element(level, ele);
		for (unsigned l = 1; l < level; l ++) {
			vid = emb_list.get_vid(level-l, idx);
			his = emb_list.get_his(level-l, idx);
			lab = graph->getData(vid);
			//emb.set_element(level-l, ElementType(vid, 0, lab, his));
			ElementType ele(vid, 0, lab, his);
			emb.set_element(level-l, ele);
			idx = emb_list.get_idx(level-l, idx);
		}
		lab = graph->getData(idx);
		ElementType ele0(idx, 0, lab, 0);
		emb.set_element(0, ele0);
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
			//if (!existed && max_size < 4) return true;
		} else {
			std::cout << "Error: should go to detailed check\n";
		}
		return false;
	}
	bool is_edge_automorphism(unsigned size, const EdgeEmbedding& emb, BYTE history, VertexId src, VertexId dst, BYTE& existed, const VertexSet& vertex_set) {
		if (size < 3) return is_quick_automorphism(size, emb, history, src, dst, existed);
		// check with the first element
		if (dst <= emb.get_vertex(0)) return true;
		if (history == 0 && dst <= emb.get_vertex(1)) return true;
		// check loop edge
		if (dst == emb.get_vertex(emb.get_history(history))) return true;
		if (vertex_set.find(dst) != vertex_set.end()) existed = 1;
		// check to see if there already exists the vertex added; 
		// if so, just allow to add edge which is (smaller id -> bigger id)
		if (existed && src > dst) return true;
		std::pair<VertexId, VertexId> added_edge(src, dst);
		for (unsigned index = history + 1; index < emb.size(); ++index) {
			std::pair<VertexId, VertexId> edge;
			edge.first = emb.get_vertex(emb.get_history(index));
			edge.second = emb.get_vertex(index);
			//assert(edge.first != edge.second);
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

#endif // EDGE_MINER_HPP_
