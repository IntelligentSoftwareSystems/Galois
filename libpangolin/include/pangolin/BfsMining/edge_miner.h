#ifndef EDGE_MINER_H
#define EDGE_MINER_H
#include "pangolin/miner.h"
#include "pangolin/quick_pattern.h"
#include "pangolin/canonical_graph.h"
#include "pangolin/domain_support.h"
#include "pangolin/BfsMining/embedding_list.h"

template <typename ElementTy, typename EmbeddingTy, typename API, bool report_num_pattern>
class EdgeMiner : public Miner<ElementTy,EmbeddingTy,false> {
typedef EmbeddingList<ElementTy,EmbeddingTy> EmbeddingListTy;
typedef QuickPattern<EmbeddingTy, ElementTy> QPattern;
typedef CanonicalGraph<EmbeddingTy, ElementTy> CPattern;
// quick pattern map (mapping quick pattern to its frequency)
typedef std::unordered_map<QPattern, Frequency> QpMapFreq;
// canonical pattern map (mapping canonical pattern to its frequency)
typedef std::unordered_map<CPattern, Frequency> CgMapFreq;
// quick pattern map (mapping quick pattern to its domain support)
typedef std::unordered_map<QPattern, DomainSupport*> QpMapDomain;
// canonical pattern map (mapping canonical pattern to its domain support)
typedef std::unordered_map<CPattern, DomainSupport*> CgMapDomain;
// PerThreadStorage: thread-local quick pattern map
typedef galois::substrate::PerThreadStorage<QpMapFreq> LocalQpMapFreq;
// PerThreadStorage: thread-local canonical pattern map
typedef galois::substrate::PerThreadStorage<CgMapFreq> LocalCgMapFreq;
typedef galois::substrate::PerThreadStorage<QpMapDomain> LocalQpMapDomain;
typedef galois::substrate::PerThreadStorage<CgMapDomain> LocalCgMapDomain;

public:
	EdgeMiner(unsigned max_sz, int nt) :
		Miner<ElementTy,EmbeddingTy,false>(max_sz, nt) {}
	virtual ~EdgeMiner() {}
	void clean() {
		edge_map.clear();
		freq_edge_set.clear();
		is_frequent_edge.clear();
		clean_maps();
	}
	void clean_maps() {
		id_map.clear();
		domain_support_map.clear();
		for (auto ele : qp_map) ele.second->clean();
		for (auto ele : cg_map) ele.second->clean();
		for (auto ele : init_map) ele.second->clean();
		qp_map.clear();
		cg_map.clear();
		init_map.clear();
		for (auto i = 0; i < this->num_threads; i++) {
			auto qp_map_ptr = qp_localmaps.getLocal(i);
			for (auto ele : *qp_map_ptr) ele.second->clean();
			qp_map_ptr->clear();
			auto cg_map_ptr = cg_localmaps.getLocal(i);
			for (auto ele : *cg_map_ptr) ele.second->clean();
			cg_map_ptr->clear();
			auto init_map_ptr = init_pattern_maps.getLocal(i);
			for (auto ele : *init_map_ptr) ele.second->clean();
			init_map_ptr->clear();
		}
	}
	void initialize(std::string pattern_filename) {
		init_emb_list();
	}
	void init_emb_list() {
		this->emb_list.init(this->graph, this->max_size+1);
		construct_edgemap();
	}
	void inc_total_num(int value) { total_num += value; }
	void solver() {
		std::cout << "Mininum support: " << threshold << "\n";
		unsigned level = 1;
		//this->emb_list.printout_embeddings(1);
		int num_freq_patterns = init_aggregator();
		if (num_freq_patterns == 0) {
			std::cout << "No frequent pattern found\n\n";
			return;
		}
		inc_total_num(num_freq_patterns);
		std::cout << "Number of frequent single-edge patterns: " << num_freq_patterns << "\n";
		init_filter();
		//this->emb_list.printout_embeddings(level);

		while (1) {
			extend_edge(level);
			level ++;
			//this->emb_list.printout_embeddings(level, debug);
			quick_aggregate(level);
			merge_qp_map(level+1);
			canonical_aggregate();
			merge_cg_map(level+1);
			num_freq_patterns = support_count();
			//std::cout << "num_frequent_patterns: " << num_freq_patterns << "\n";
			//printout_agg();
			inc_total_num(num_freq_patterns);
			if (num_freq_patterns == 0) break;
			if (level == this->max_size) break;
			filter(level);
			//this->emb_list.printout_embeddings(level, debug);
			clean_maps();
		}
	}

	void extend_edge(unsigned level) {
		UintList num_new_emb(this->emb_list.size());
		galois::do_all(galois::iterate((size_t)0, this->emb_list.size()), [&](const size_t& pos) {
			EmbeddingTy emb(level+1);
			get_embedding(level, pos, emb);
			num_new_emb[pos] = 0;
			unsigned n = emb.size();
			VertexSet vert_set;
			if (n > 3)
				for (unsigned i = 0; i < n; i ++) vert_set.insert(emb.get_vertex(i));
			for (unsigned i = 0; i < n; ++i) {
				auto src = emb.get_vertex(i);
				if (emb.get_key(i) == 0) { // TODO: need to fix this
					for (auto e : this->graph.edges(src)) {
						GNode dst = this->graph.getEdgeDst(e);
						BYTE existed = 0;
						if (is_frequent_edge[*e])
							if (API::toAdd(n, emb, i, src, dst, existed, vert_set))
								num_new_emb[pos] ++;
					}
				}
			}
			emb.clean();
		}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("Extending-alloc"));
		Ulong new_size = std::accumulate(num_new_emb.begin(), num_new_emb.end(), (Ulong)0);
		UlongList indices = parallel_prefix_sum<unsigned,Ulong>(num_new_emb);
		new_size = indices[indices.size()-1];
		this->emb_list.add_level(new_size);
		galois::do_all(galois::iterate((size_t)0, this->emb_list.size(level)), [&](const size_t& pos) {
			EmbeddingTy emb(level+1);
			get_embedding(level, pos, emb);
			unsigned start = indices[pos];
			unsigned n = emb.size();
			VertexSet vert_set;
			if (n > 3)
				for (unsigned i = 0; i < n; i ++) vert_set.insert(emb.get_vertex(i));
			for (unsigned i = 0; i < n; ++i) {
				auto src = emb.get_vertex(i);
				if (emb.get_key(i) == 0) {
					for (auto e : this->graph.edges(src)) {
						GNode dst = this->graph.getEdgeDst(e);
						BYTE existed = 0;
						if (is_frequent_edge[*e]) {
							if (API::toAdd(n, emb, i, src, dst, existed, vert_set)) {
								this->emb_list.set_idx(level+1, start, pos);
								this->emb_list.set_his(level+1, start, i);
								this->emb_list.set_vid(level+1, start++, dst);
							}
						}
					}
				}
			}
		}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("Extending-write"));
	}
	inline unsigned init_aggregator() {
		init_map.clear();
		galois::do_all(galois::iterate(this->graph.begin(), this->graph.end()), [&](const GNode& src) {
			InitMap *lmap = init_pattern_maps.getLocal();
			auto& src_label = this->graph.getData(src);
			for (auto e : this->graph.edges(src)) {
				GNode dst = this->graph.getEdgeDst(e);
				auto& dst_label = this->graph.getData(dst);
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
		}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("InitAggregation"));
		merge_init_map();
		std::cout << "Number of single-edge patterns: " << init_map.size() << "\n";
		unsigned count = 0;
		for (auto it = init_map.begin(); it != init_map.end(); ++it)
			if (it->second->get_support()) count ++;
		return count; // return number of frequent single-edge patterns
	}
	inline void quick_aggregate(unsigned level) {
		for (auto i = 0; i < this->num_threads; i++) qp_localmaps.getLocal(i)->clear();
		galois::do_all(galois::iterate((size_t)0, this->emb_list.size()), [&](const size_t& pos) {
			QpMapDomain *lmap = qp_localmaps.getLocal();
			EmbeddingTy emb(level+1);
			get_embedding(level, pos, emb);
			unsigned n = emb.size();
			QPattern qp(emb, true);
			bool qp_existed = false;
			auto it = lmap->find(qp);
			if (it == lmap->end()) {
				(*lmap)[qp] = new DomainSupport(n);
				(*lmap)[qp]->set_threshold(threshold);
				this->emb_list.set_pid(pos, qp.get_id());
			} else {
				qp_existed = true;
				this->emb_list.set_pid(pos, (it->first).get_id());
			}
			for (unsigned i = 0; i < n; i ++) {
				if ((*lmap)[qp]->has_domain_reached_support(i) == false)
					(*lmap)[qp]->add_vertex(i, emb.get_vertex(i));
			}
			if (qp_existed) qp.clean();
		}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("QuickAggregation"));
	}
	// aggregate quick patterns into canonical patterns.
	// construct id_map from quick pattern ID (qp_id) to canonical pattern ID (cg_id)
	void canonical_aggregate() {
		id_map.clear();
		for (auto i = 0; i < this->num_threads; i++) cg_localmaps.getLocal(i)->clear();
		galois::do_all(galois::iterate(qp_map), [&](std::pair<QPattern, DomainSupport*> element) {
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
		}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("CanonicalAggregation"));
	}
	inline void merge_qp_map(LocalQpMapFreq &qp_localmap, QpMapFreq &qp_map) {
		for (auto i = 0; i < this->num_threads; i++) {
			for (auto element : *qp_localmap.getLocal(i)) {
				if (qp_map.find(element.first) != qp_map.end())
					qp_map[element.first] += element.second;
				else qp_map[element.first] = element.second;
			}
		}
	}
	inline void merge_cg_map(LocalCgMapFreq &localmaps, CgMapFreq &cg_map) {
		for (auto i = 0; i < this->num_threads; i++) {
			for (auto element : *localmaps.getLocal(i)) {
				if (cg_map.find(element.first) != cg_map.end())
					cg_map[element.first] += element.second;
				else cg_map[element.first] = element.second;
			}
		}
	}
	inline void merge_init_map() {
		init_map = *(init_pattern_maps.getLocal(0));
		for (auto i = 1; i < this->num_threads; i++) {
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
	}
	inline void merge_qp_map(unsigned num_domains) {
		qp_map.clear();
		qp_map = *(qp_localmaps.getLocal(0));
		for (auto i = 1; i < this->num_threads; i++) {
			const QpMapDomain *lmap = qp_localmaps.getLocal(i);
			for (auto element : *lmap) {
				if (qp_map.find(element.first) == qp_map.end())
					qp_map[element.first] = element.second;
			}
			galois::do_all(galois::iterate(*lmap), [&](std::pair<QPattern, DomainSupport*> element) {
				DomainSupport *support = element.second;
				for (unsigned i = 0; i < num_domains; i ++) {
					if (!qp_map[element.first]->has_domain_reached_support(i) && 
						qp_map[element.first] != support) {
						if (support->has_domain_reached_support(i))
							qp_map[element.first]->set_domain_frequent(i);
						else qp_map[element.first]->add_vertices(i, support->domain_sets[i]);
					}
				}
			}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), 
			galois::loopname("MergeQuickPatterns"));
		}
	}
	inline void merge_cg_map(unsigned num_domains) {
		cg_map.clear();
		cg_map = *(cg_localmaps.getLocal(0));
		for (auto i = 1; i < this->num_threads; i++) {
			const CgMapDomain *lmap = cg_localmaps.getLocal(i);
			for (auto element : *lmap) {
				if (cg_map.find(element.first) == cg_map.end())
					cg_map[element.first] = element.second;
			}
			galois::do_all(galois::iterate(*lmap), [&](std::pair<CPattern, DomainSupport*> element) {
				DomainSupport *support = element.second;
				for (unsigned i = 0; i < num_domains; i ++) {
					if (!cg_map[element.first]->has_domain_reached_support(i) && 
						cg_map[element.first] != support) {
						if (support->has_domain_reached_support(i))
							cg_map[element.first]->set_domain_frequent(i);
						else cg_map[element.first]->add_vertices(i, support->domain_sets[i]);
					}
				}
			}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(),
			galois::loopname("MergeCanonicalPatterns"));
		}
	}

	// Filtering for FSM
	inline void init_filter() {
		UintList is_frequent_emb(this->emb_list.size(), 0);
		galois::do_all(galois::iterate((size_t)0, this->emb_list.size()), [&](const size_t& pos) {
			auto src = this->emb_list.get_idx(1, pos);
			auto dst = this->emb_list.get_vid(1, pos);
			auto& src_label = this->graph.getData(src);
			auto& dst_label = this->graph.getData(dst);
			InitPattern key = get_init_pattern(src_label, dst_label);
			if (init_map[key]->get_support()) is_frequent_emb[pos] = 1;
		}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("InitFilter"));

		assert(this->emb_list.size()*2 == this->graph.sizeEdges()); // symmetric graph
		is_frequent_edge.resize(this->graph.sizeEdges());
		std::fill(is_frequent_edge.begin(), is_frequent_edge.end(), 0);
		galois::do_all(galois::iterate((size_t)0, this->emb_list.size()), [&](const size_t& pos) {
			if (is_frequent_emb[pos]) {
				auto src = this->emb_list.get_idx(1, pos);
				auto dst = this->emb_list.get_vid(1, pos);
				auto eid0 = edge_map[OrderedEdge(src,dst)];
				auto eid1 = edge_map[OrderedEdge(dst,src)];
				//std::cout << "src=" << src << ", dst=" << dst
				//	<< ", eid_sd=" << eid0 << ", eid_ds=" << eid1 << "\n";
				__sync_bool_compare_and_swap(&is_frequent_edge[eid0], 0, 1);
				__sync_bool_compare_and_swap(&is_frequent_edge[eid1], 0, 1);
			}
		}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("InitFrquentEdges"));
		std::cout << "Number of frequent edges: " << 
			count(is_frequent_edge.begin(), is_frequent_edge.end(), 1) << "\n";
	
		UintList indices = parallel_prefix_sum(is_frequent_emb);
		VertexList vid_list0 = this->emb_list.get_idx_list(1);
		VertexList vid_list1 = this->emb_list.get_vid_list(1);
		galois::do_all(galois::iterate((size_t)0, this->emb_list.size()), [&](const size_t& pos) {
			if (is_frequent_emb[pos]) {
				auto src = vid_list0[pos];
				auto dst = vid_list1[pos];
				auto start = indices[pos];
				this->emb_list.set_vid(1, start, dst);
				this->emb_list.set_idx(1, start, src);
			}
		}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("InitEmbeddingList"));
		this->emb_list.remove_tail(indices.back());
	}

	// Check if the pattern of a given embedding is frequent, if yes, insert it to the queue
	inline void filter(unsigned level) {
		UintList is_frequent_emb(this->emb_list.size(), 0);
		galois::do_all(galois::iterate((size_t)0, this->emb_list.size()), [&](const size_t& pos) {
			unsigned qp_id = this->emb_list.get_pid(pos);
			unsigned cg_id = id_map.at(qp_id);
			if (domain_support_map.at(cg_id))
				is_frequent_emb[pos] = 1;
		}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("Filter-alloc"));
		UlongList indices = parallel_prefix_sum<unsigned,Ulong>(is_frequent_emb);
		VertexList vid_list = this->emb_list.get_vid_list(level);
		UintList idx_list = this->emb_list.get_idx_list(level);
		ByteList his_list = this->emb_list.get_his_list(level);
		galois::do_all(galois::iterate((size_t)0, this->emb_list.size()), [&](const size_t& pos) {
			if (is_frequent_emb[pos]) {
				auto start = indices[pos];
				auto vid = vid_list[pos];
				auto idx = idx_list[pos];
				auto his = his_list[pos];
				this->emb_list.set_idx(level, start, idx);
				this->emb_list.set_vid(level, start, vid);
				this->emb_list.set_his(level, start, his);
			}
		}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("Filter-write"));
		this->emb_list.remove_tail(indices.back());
	}
	void set_threshold(const unsigned minsup) { threshold = minsup; }
	inline void printout_agg(const CgMapFreq &cg_map) {
		for (auto it = cg_map.begin(); it != cg_map.end(); ++it)
			std::cout << "{" << it->first << " --> " << it->second << std::endl;
	}
	inline void printout_agg() {
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
		//std::cout << "Constructing edge map ...\n";
		for (auto src : this->graph) {
			for (auto e : this->graph.edges(src)) {
				GNode dst = this->graph.getEdgeDst(e);
				OrderedEdge edge(src, dst);
				//std::cout << "src=" << src << ", dst=" << dst
				//	<< ", eid=" << *e << "\n";
				edge_map.insert(std::pair<OrderedEdge, unsigned>(edge, *e));
			}
		}
	}

protected:
	int total_num; // total number of frequent patterns
	unsigned threshold;
	EmbeddingListTy emb_list;

private:
	InitMap init_map;
	UintMap id_map;
	DomainMap domain_support_map;
	galois::gstl::Map<OrderedEdge, unsigned> edge_map;
	std::set<std::pair<VertexId,VertexId> > freq_edge_set;
	std::vector<unsigned> is_frequent_edge;
	InitMaps init_pattern_maps; // initialization map, only used for once, no need to clear
	LocalQpMapDomain qp_localmaps; // quick pattern local map for each thread
	LocalCgMapDomain cg_localmaps; // canonical pattern local map for each thread
	QpMapDomain qp_map; // quick pattern map
	CgMapDomain cg_map; // canonical graph map
	galois::substrate::SimpleLock slock;

	inline InitPattern get_init_pattern(BYTE src_label, BYTE dst_label) {
		if (src_label <= dst_label) return std::make_pair(src_label, dst_label);
		else return std::make_pair(dst_label, src_label);
	}
	inline void get_embedding(unsigned level, unsigned pos, EmbeddingTy &emb) {
		auto vid = this->emb_list.get_vid(level, pos);
		auto idx = this->emb_list.get_idx(level, pos);
		auto his = this->emb_list.get_his(level, pos);
		auto lab = this->graph.getData(vid);
		ElementTy ele(vid, 0, lab, his);
		emb.set_element(level, ele);
		for (unsigned l = 1; l < level; l ++) {
			vid = this->emb_list.get_vid(level-l, idx);
			his = this->emb_list.get_his(level-l, idx);
			lab = this->graph.getData(vid);
			ElementTy ele(vid, 0, lab, his);
			emb.set_element(level-l, ele);
			idx = this->emb_list.get_idx(level-l, idx);
		}
		lab = this->graph.getData(idx);
		ElementTy ele0(idx, 0, lab, 0);
		emb.set_element(0, ele0);
	}
};

#endif // EDGE_MINER_HPP_
