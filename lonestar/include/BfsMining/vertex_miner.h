#ifndef VERTEX_MINER_H
#define VERTEX_MINER_H
#include "miner.h"
#include "embedding_list.h"
#include "quick_pattern.h"
#include "canonical_graph.h"

typedef std::unordered_map<BaseEmbedding, Frequency> SimpleMap;
typedef QuickPattern<EdgeInducedEmbedding<StructuralElement>, StructuralElement> StrQPattern; // structural quick pattern
typedef CanonicalGraph<EdgeInducedEmbedding<StructuralElement>, StructuralElement> StrCPattern; // structural canonical pattern
typedef std::unordered_map<StrQPattern, Frequency> StrQpMapFreq; // mapping structural quick pattern to its frequency
typedef std::unordered_map<StrCPattern, Frequency> StrCgMapFreq; // mapping structural canonical pattern to its frequency
typedef galois::substrate::PerThreadStorage<StrQpMapFreq> LocalStrQpMapFreq;
typedef galois::substrate::PerThreadStorage<StrCgMapFreq> LocalStrCgMapFreq;
typedef galois::gstl::Vector<BaseEmbedding> BaseEmbeddingBuffer;
typedef galois::gstl::Vector<VertexEmbedding> VertexEmbeddingBuffer;

class Status {
protected:
	std::vector<uint8_t> visited;
public:
	Status() {}
	~Status() {}
	void init(unsigned size) {
		visited.resize(size);
		reset();
	}
	void reset() {
		std::fill(visited.begin(), visited.end(), 0);
	}
	void set(VertexId pos, uint8_t value) { visited[pos] = value; }
	uint8_t get(VertexId pos) { return visited[pos]; }
};

typedef galois::substrate::PerThreadStorage<Status> StatusMT; // multi-threaded

class VertexMiner : public Miner {
public:
	VertexMiner(Graph *g) {
		graph = g;
		degree_counting();
	}
	virtual ~VertexMiner() {}
	void set_max_size(unsigned size = 3) { max_size = size; }
	void set_num_patterns(int np = 1) {
		npatterns = np;
		if (npatterns == 1) total_num.reset();
		else {
			accumulators.resize(npatterns);
			for (int i = 0; i < npatterns; i++) accumulators[i].reset();
			//std::cout << max_size << "-motif has " << npatterns << " patterns in total\n";
			#ifdef USE_MAP
			for (auto i = 0; i < numThreads; i++) qp_localmaps.getLocal(i)->clear();
			#endif
		}
	}
	void clean() {
		is_wedge.clear();
		accumulators.clear();
		qp_map.clear();
		cg_map.clear();
		for (auto i = 0; i < numThreads; i++) qp_localmaps.getLocal(i)->clear();
		for (auto i = 0; i < numThreads; i++) cg_localmaps.getLocal(i)->clear();
	}

	// Pangolin APIs
	virtual void init() {}
	// toExtend
	virtual bool toExtend(unsigned n, const BaseEmbedding &emb, unsigned pos) {
		return true;
	}
	// toAdd (only add non-automorphisms)
	virtual bool toAdd(unsigned n, const BaseEmbedding &emb, VertexId dst, unsigned pos) {
		return !is_vertexInduced_automorphism<BaseEmbedding>(n, emb, pos, dst);
	}
	virtual bool toExtend(unsigned n, const VertexEmbedding &emb, unsigned pos) {
		return true;
	}
	// toAdd (only add non-automorphisms)
	virtual bool toAdd(unsigned n, const VertexEmbedding &emb, VertexId dst, unsigned pos) {
		return !is_vertexInduced_automorphism<VertexEmbedding>(n, emb, pos, dst);
	}
	virtual unsigned getPattern(unsigned n, unsigned i, VertexId dst, const VertexEmbedding &emb, unsigned pos) {
		return 0;
	}
	virtual void print_output() {
	}

	// Given an embedding, extend it with one more vertex. Used for cliques. (fast)
	inline void extend_vertex(BaseEmbeddingQueue &in_queue, BaseEmbeddingQueue &out_queue) {
		//galois::runtime::profileVtune([&] () {
		galois::do_all(galois::iterate(in_queue), [&](const BaseEmbedding& emb) {
			unsigned n = emb.size();
			for (unsigned i = 0; i < n; ++i) {
				if(!toExtend(n, emb, i)) continue;
				auto src = emb.get_vertex(i);
				for (auto e : graph->edges(src)) {
					GNode dst = graph->getEdgeDst(e);
					if (toAdd(n, emb, dst, i)) {
						if (n < max_size-1) { // generate a new embedding and add it to the next queue
							BaseEmbedding new_emb(emb);
							new_emb.push_back(dst);
							out_queue.push_back(new_emb);
						} else {
							if (debug) {
								BaseEmbedding new_emb(emb);
								new_emb.push_back(dst);
								std::cout << new_emb << "\n";
							}
							total_num += 1; // if size = max_size, no need to add to the queue, just accumulate
						}
					}
				}
			}
		}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("Extending"));
		//}, "ExtendingVtune");
	}
	/*
	inline void extend_vertex_lazy(BaseEmbeddingQueue &in_queue, BaseEmbeddingQueue &out_queue) {
		std::cout << "lazy extension\n";
		galois::do_all(galois::iterate(in_queue), [&](const BaseEmbedding& emb) {
			unsigned n = emb.size();
			auto src = emb.get_vertex(n-1);
			UintList buffer; 
			for (auto e : graph->edges(src)) {
				auto dst = graph->getEdgeDst(e);
				buffer.push_back(dst);
			}
			for (auto dst : buffer) {
				if (toAdd(n, emb, dst, n-1)) {
					if (n < max_size-1) {
						BaseEmbedding new_emb(emb);
						new_emb.push_back(dst);
						out_queue.push_back(new_emb);
					} else total_num += 1;
				}
			}
		}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("Extending"));
	}
	//*/
	// Given an embedding, extend it with one more vertex. Used for vertex-induced motif
	inline void extend_vertex(VertexEmbeddingQueue &in_queue, VertexEmbeddingQueue &out_queue) {
		//galois::runtime::profilePapi([&] () {
		// for each embedding in the task queue, do vertex-extension
		galois::do_all(galois::iterate(in_queue), [&](const VertexEmbedding& emb) {
			unsigned n = emb.size();
			#ifdef USE_MAP
			StrQpMapFreq *qp_lmap;
			if (n >= 4) qp_lmap = qp_localmaps.getLocal();
			#endif
			for (unsigned i = 0; i < n; ++i) {
				if (!toExtend(n, emb, i)) continue;
				auto src = emb.get_vertex(i);
				for (auto e : graph->edges(src)) {
					auto dst = graph->getEdgeDst(e);
					if (toAdd(n, emb, dst, i)) {
						if (n < max_size-1) {
							VertexEmbedding new_emb(emb);
							#ifdef USE_PID
							if (n == 2 && max_size == 4) new_emb.set_pid(find_motif_pattern_id(n, i, dst, emb));
							#endif
							new_emb.push_back(dst);
							out_queue.push_back(new_emb);
						} else {
							if (n < 4) {
								unsigned pid = find_motif_pattern_id(n, i, dst, emb);
								accumulators[pid] += 1;
							} else {
								#ifdef USE_MAP
								std::vector<bool> connected;
								get_connectivity(n, i, dst, emb, connected);
								StrQPattern qp(n+1, connected);
								if (qp_lmap->find(qp) != qp_lmap->end()) {
									(*qp_lmap)[qp] += 1;
									qp.clean();
								} else (*qp_lmap)[qp] = 1;
								#endif
							}
						}
					}
				}
			}
		}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("Extending"));
		//}, "ExtendingPapi");
	}
	// extension for vertex-induced motif
	inline void extend_vertex_multi(unsigned level, EmbeddingList& emb_list, size_t chunk_begin, size_t chunk_end) {
		auto cur_size = emb_list.size();
		size_t begin = 0, end = cur_size;
		if (level == 1) {
			begin = chunk_begin; end = chunk_end; cur_size = end - begin;
			std::cout << "\t chunk_begin = " << chunk_begin << ", chunk_end " << chunk_end << "\n";
		}
		if (show) std::cout << "\t number of current embeddings in level " << level << ": " << cur_size << "\n";
		UintList num_new_emb(cur_size); // TODO: for large graph, wo need UlongList
		//UlongList num_new_emb(cur_size);
		//galois::runtime::profilePapi([&] () {
		galois::do_all(galois::iterate(begin, end), [&](const size_t& pos) {
			unsigned n = level+1;
			#ifdef USE_MAP
			StrQpMapFreq *qp_lmap;
			if (n >= 4) qp_lmap = qp_localmaps.getLocal();
			#endif
			VertexEmbedding emb(n);
			get_embedding<VertexEmbedding>(level, pos, emb_list, emb);
			if (n < max_size-1) num_new_emb[pos-begin] = 0;
			if (n == 3 && max_size == 4) emb.set_pid(emb_list.get_pid(pos));
			for (unsigned i = 0; i < n; ++i) {
				auto src = emb.get_vertex(i);
				for (auto e : graph->edges(src)) {
					GNode dst = graph->getEdgeDst(e);
					if (toAdd(n, emb, dst, i)) {
						if (n < max_size-1) {
							num_new_emb[pos-begin] ++;
						} else {
							#ifdef USE_MAP
							//unsigned pid  = getPattern(n, i, dst, emb, pos);
							#ifdef USE_CUSTOM
							if (n < 4) {
								unsigned pid = find_motif_pattern_id(n, i, dst, emb, pos);
								accumulators[pid] += 1;
							} else
							#endif
								quick_reduce(n, i, dst, emb, qp_lmap);
							#endif
						}
					}
				}
			}
		}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("Extending-alloc"));
		//}, "ExtendingAllocPapi");
		if (level == max_size-2) return;
		//UintList indices = parallel_prefix_sum<unsigned>(num_new_emb);
		UlongList indices = parallel_prefix_sum<unsigned,Ulong>(num_new_emb);
		num_new_emb.clear();
		Ulong new_size = indices.back();
		//std::cout << "number of new embeddings: " << new_size << "\n";
		emb_list.add_level(new_size);
		#ifdef USE_WEDGE
		if (level == 1 && max_size == 4) {
			is_wedge.resize(emb_list.size());
			std::fill(is_wedge.begin(), is_wedge.end(), 0);
		}
		#endif
		//galois::runtime::profilePapi([&] () {
		galois::do_all(galois::iterate(begin, end), [&](const size_t& pos) {
			VertexEmbedding emb(level+1);
			get_embedding<VertexEmbedding>(level, pos, emb_list, emb);
			auto start = indices[pos-begin];
			auto n = emb.size();
			for (unsigned i = 0; i < n; ++i) {
				auto src = emb.get_vertex(i);
				for (auto e : graph->edges(src)) {
					GNode dst = graph->getEdgeDst(e);
					if (toAdd(n, emb, dst, i)) {
						assert(start < indices.back());
						if (n == 2 && max_size == 4)
							emb_list.set_pid(start, find_motif_pattern_id(n, i, dst, emb, start));
						emb_list.set_idx(level+1, start, pos);
						emb_list.set_vid(level+1, start++, dst);
					}
				}
			}
		}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("Extending-insert"));
		//}, "ExtendingInsertPapi");
		indices.clear();
	}
	// extension for vertex-induced clique
	inline void extend_vertex_single(unsigned level, EmbeddingList& emb_list, size_t chunk_begin, size_t chunk_end) {
		auto cur_size = emb_list.size();
		size_t begin = 0, end = cur_size;
		if (level == 1) {
			begin = chunk_begin; end = chunk_end; cur_size = end - begin;
			std::cout << "\t chunk_begin = " << chunk_begin << ", chunk_end " << chunk_end << "\n";
		}
		std::cout << "\t number of current embeddings in level " << level << ": " << cur_size << "\n";
		UintList num_new_emb(cur_size);
		//galois::runtime::profilePapi([&] () {
		galois::do_all(galois::iterate(begin, end), [&](const size_t& pos) {
			BaseEmbedding emb(level+1);
			get_embedding<BaseEmbedding>(level, pos, emb_list, emb);
			auto vid = emb_list.get_vid(level, pos);
			num_new_emb[pos-begin] = 0;
			for (auto e : graph->edges(vid)) {
				GNode dst = graph->getEdgeDst(e);
				if (toAdd(level+1, emb, dst, level)) {
					if (level < max_size-2) num_new_emb[pos-begin] ++;
					else total_num += 1;
				}
			}
		}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("Extending-alloc"));
		//}, "ExtendingAllocPapi");
		if (level == max_size-2) return;
		//UintList indices = parallel_prefix_sum<unsigned>(num_new_emb);
		UlongList indices = parallel_prefix_sum<unsigned,Ulong>(num_new_emb);
		num_new_emb.clear();
		Ulong new_size = indices.back();
		std::cout << "\t number of new embeddings: " << new_size << "\n";
		emb_list.add_level(new_size);
		//galois::runtime::profilePapi([&] () {
		galois::do_all(galois::iterate(begin, end), [&](const size_t& pos) {
			BaseEmbedding emb(level+1);
			get_embedding<BaseEmbedding>(level, pos, emb_list, emb);
			auto vid = emb_list.get_vid(level, pos);
			unsigned start = indices[pos-begin];
			for (auto e : graph->edges(vid)) {
				GNode dst = graph->getEdgeDst(e);
				if (toAdd(level+1, emb, dst, level)) {
					emb_list.set_idx(level+1, start, pos);
					emb_list.set_vid(level+1, start++, dst);
				}
			}
		}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("Extending-insert"));
		//}, "ExtendingInsertPapi");
		indices.clear();
	}
	// quick pattern reduction
	inline void quick_reduce(unsigned n, unsigned i, VertexId dst, const VertexEmbedding &emb, StrQpMapFreq *qp_lmap) {
		std::vector<bool> connected;
		get_connectivity(n, i, dst, emb, connected);
		StrQPattern qp(n+1, connected);
		if (qp_lmap->find(qp) != qp_lmap->end()) {
			(*qp_lmap)[qp] += 1;
			qp.clean();
		} else (*qp_lmap)[qp] = 1;
	}
	// canonical pattern reduction
	inline void canonical_reduce() {
		for (auto i = 0; i < numThreads; i++) cg_localmaps.getLocal(i)->clear();
		galois::do_all(galois::iterate(qp_map),
			[&](auto& element) {
				StrCgMapFreq* cg_map = cg_localmaps.getLocal();
				StrCPattern cg(element.first);
				if (cg_map->find(cg) != cg_map->end()) (*cg_map)[cg] += element.second;
				else (*cg_map)[cg] = element.second;
				cg.clean();

			},
			galois::chunk_size<CHUNK_SIZE>(),
			#ifdef ENABLE_STEAL
			galois::steal(),
			#endif
			galois::loopname("CanonicalReduction")
		);
		qp_map.clear();
	}
	inline void merge_qp_map() {
		qp_map.clear();
		for (unsigned i = 0; i < qp_localmaps.size(); i++) {
			StrQpMapFreq qp_lmap = *qp_localmaps.getLocal(i);
			for (auto element : qp_lmap) {
				if (qp_map.find(element.first) != qp_map.end())
					qp_map[element.first] += element.second;
				else qp_map[element.first] = element.second;
			}
		}
	}
	inline void merge_cg_map() {
		cg_map.clear();
		for (unsigned i = 0; i < cg_localmaps.size(); i++) {
			StrCgMapFreq cg_lmap = *cg_localmaps.getLocal(i);
			for (auto element : cg_lmap) {
				if (cg_map.find(element.first) != cg_map.end())
					cg_map[element.first] += element.second;
				else cg_map[element.first] = element.second;
			}
		}
	}

	// Utilities
	//inline unsigned get_total_num_cliques() { return num_cliques; }
	Ulong get_total_count() { return total_num.reduce(); }
	void printout_motifs() {
		std::cout << std::endl;
		if (accumulators.size() == 2) {
			std::cout << "\ttriangles\t" << accumulators[0].reduce() << std::endl;
			std::cout << "\t3-chains\t" << accumulators[1].reduce() << std::endl;
		} else if (accumulators.size() == 6) {
			std::cout << "\t4-paths --> " << accumulators[0].reduce() << std::endl;
			std::cout << "\t3-stars --> " << accumulators[1].reduce() << std::endl;
			std::cout << "\t4-cycles --> " << accumulators[2].reduce() << std::endl;
			std::cout << "\ttailed-triangles --> " << accumulators[3].reduce() << std::endl;
			std::cout << "\tdiamonds --> " << accumulators[4].reduce() << std::endl;
			std::cout << "\t4-cliques --> " << accumulators[5].reduce() << std::endl;
		} else {
			if (max_size < 9) {
				std::cout << std::endl;
				//for (auto it = p_map.begin(); it != p_map.end(); ++it)
				//	std::cout << "{" << it->first << "} --> " << it->second << std::endl;
			} else {
				std::cout << std::endl;
				for (auto it = cg_map.begin(); it != cg_map.end(); ++it)
					std::cout << it->first << " --> " << it->second << std::endl;
			}
		}
		//std::cout << std::endl;
	}
	void tc_solver() {
		galois::do_all(galois::iterate(graph->begin(), graph->end()),
			[&](const GNode& src) {
				for (auto e : graph->edges(src)) {
					GNode dst = graph->getEdgeDst(e);
					total_num += intersect(src, dst);
				}
			},
			galois::chunk_size<CHUNK_SIZE>(), galois::steal(), 
			galois::loopname("TriangleCouting")
		);
	}

	void tc_solver(EmbeddingList& emb_list) {
		galois::do_all(galois::iterate((size_t)0, emb_list.size()),
			[&](const size_t& id) {
				total_num += intersect_dag(emb_list.get_idx(1,id), emb_list.get_vid(1,id));
			},
			galois::chunk_size<CHUNK_SIZE>(), galois::steal(), 
			galois::loopname("TriangleCouting")
		);
	}

private:
	StrQpMapFreq qp_map; // quick patterns map for counting the frequency
	StrCgMapFreq cg_map; // canonical graph map for couting the frequency
	LocalStrQpMapFreq qp_localmaps; // quick patterns local map for each thread
	LocalStrCgMapFreq cg_localmaps; // canonical graph local map for each thread
	galois::substrate::SimpleLock slock;
	template <typename EmbeddingTy>
	inline void get_embedding(unsigned level, unsigned pos, const EmbeddingList& emb_list, EmbeddingTy &emb) {
		auto vid = emb_list.get_vid(level, pos);
		auto idx = emb_list.get_idx(level, pos);
		ElementType ele(vid);
		emb.set_element(level, ele);
		// backward constructing the embedding
		for (unsigned l = 1; l < level; l ++) {
			auto u = emb_list.get_vid(level-l, idx);
			ElementType ele(u);
			emb.set_element(level-l, ele);
			idx = emb_list.get_idx(level-l, idx);
		}
		ElementType ele0(idx);
		emb.set_element(0, ele0);
	}

protected:
	int npatterns;
	UlongAccu total_num;
	std::vector<UlongAccu> accumulators;
};

#endif // VERTEX_MINER_HPP_
