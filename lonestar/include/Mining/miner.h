#ifndef MINER_HPP_
#define MINER_HPP_
#include "quick_pattern.h"
#include "canonical_graph.h"
#include "galois/substrate/PerThreadStorage.h"
#include "galois/substrate/SimpleLock.h"

// We provide two types of 'support': frequency and domain support.
// Frequency is used for counting, e.g. motif counting.
// Domain support, a.k.a, the minimum image-based support, is used for FSM. It has the anti-monotonic property.
typedef unsigned Frequency;
typedef QuickPattern<EdgeEmbedding, ElementType> QPattern;
typedef QuickPattern<VertexEmbedding, SimpleElement> QPSimple;
typedef CanonicalGraph<EdgeEmbedding, ElementType> CPattern;
typedef CanonicalGraph<VertexEmbedding, SimpleElement> CPSimple;
typedef HashIntSets DomainSupport;
typedef std::unordered_map<QPattern, Frequency> QpMapFreq; // mapping quick pattern to its frequency
typedef std::unordered_map<QPSimple, Frequency> QpSMapFreq; // mapping quick pattern to its frequency
typedef std::unordered_map<CPattern, Frequency> CgMapFreq; // mapping canonical pattern to its frequency
typedef std::unordered_map<QPattern, DomainSupport> QpMapDomain; // mapping quick pattern to its domain support
typedef std::unordered_map<CPattern, DomainSupport> CgMapDomain; // mapping canonical pattern to its domain support
typedef std::unordered_map<BaseEmbedding, Frequency> SimpleMap;
typedef std::unordered_map<unsigned, Frequency> Map3Motif;
typedef galois::substrate::PerThreadStorage<QpMapDomain> LocalQpMapDomain;
typedef galois::substrate::PerThreadStorage<CgMapDomain> LocalCgMapDomain;
typedef galois::substrate::PerThreadStorage<QpMapFreq> LocalQpMapFreq;
typedef galois::substrate::PerThreadStorage<CgMapFreq> LocalCgMapFreq;
typedef galois::substrate::PerThreadStorage<SimpleMap> LocalSimpleMap;
typedef galois::substrate::PerThreadStorage<Map3Motif> LocalMap;
typedef galois::GAccumulator<unsigned> UintAccu;

class Miner {
public:
	Miner(Graph *g) {
		graph = g;
		num_cliques = 0;
		degree_counting();
	}
	virtual ~Miner() {};
	// insert single-edge embeddings into the embedding queue (worklist)
	inline void init(EmbeddingQueueT &queue) {
		if (show) printf("\n=============================== Init ================================\n\n");
		galois::do_all(galois::iterate(graph->begin(), graph->end()),
			[&](const GNode& src) {
#ifdef ENABLE_LABEL
				auto& src_label = graph->getData(src);
#endif
				for (auto e : graph->edges(src)) {
					GNode dst = graph->getEdgeDst(e);
					if(src < dst) {
#ifdef ENABLE_LABEL
						auto& dst_label = graph->getData(dst);
#endif
						EmbeddingT new_emb;
#ifdef ENABLE_LABEL
						new_emb.push_back(ElementType(src, 0, src_label));
						new_emb.push_back(ElementType(dst, 0, dst_label));
#else
						new_emb.push_back(ElementType(src));
						new_emb.push_back(ElementType(dst));
#endif
						queue.push_back(new_emb);
					}
				}
			},
			galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::no_conflicts(),
			galois::wl<galois::worklists::PerSocketChunkFIFO<CHUNK_SIZE>>(),
			galois::loopname("Initialization")
		);
	}
	// given an embedding, extend it with one more edge, and if it is not automorphism, insert the new embedding into the task queue
	void extend_edge(unsigned max_size, EdgeEmbedding emb, EdgeEmbeddingQueue &queue) {
		unsigned size = emb.size();
		// get the number of distinct vertices in the embedding
		std::unordered_set<VertexId> vertices_set;
		vertices_set.reserve(size);
		for(unsigned i = 0; i < size; i ++) vertices_set.insert(emb.get_vertex(i));
		std::unordered_set<VertexId> set; // uesd to make sure each distinct vertex is expanded only once
		// for each vertex in the embedding
		for(unsigned i = 0; i < size; ++i) {
			VertexId id = emb.get_vertex(i);
			assert(id >= 0 && id < graph->size());
			if(set.find(id) == set.end()) {
				set.insert(id);
				// try edge extension
				for(auto e : graph->edges(id)) {
					GNode dst = graph->getEdgeDst(e);
					auto dst_label = 0, edge_label = 0;
					#ifdef ENABLE_LABEL
					dst_label = graph->getData(dst);
					//edge_label = graph->getEdgeData(e); // TODO: enable this for FSM
					#endif
					auto num_vertices = vertices_set.size();
					bool vertex_existed = true;
					if(vertices_set.find(dst) == vertices_set.end()) {
						num_vertices ++;
						vertex_existed = false;
					}
					// number of vertices must be smaller than k.
					// check if this is automorphism
					if(num_vertices <= max_size && !is_edgeInduced_automorphism(emb, i, id, dst, vertex_existed)) {
						ElementType new_element(dst, (BYTE)num_vertices, edge_label, dst_label, (BYTE)i);
						// insert the new extended embedding into the queue
						emb.push_back(new_element);
						queue.push_back(emb);
						emb.pop_back();
					}
				}
			}
		}
	}
	// Given an embedding, extend it with one more vertex. Used for vertex-induced k-motif
	void extend_vertex_motif(const VertexEmbedding &emb, VertexEmbeddingQueue &queue) {
		unsigned n = emb.size();
		for (unsigned i = 0; i < n; ++i) {
			VertexId src = emb.get_vertex(i);
			for (auto e : graph->edges(src)) {
				GNode dst = graph->getEdgeDst(e);
				if (!is_vertexInduced_automorphism(emb, i, src, dst)) {
					VertexEmbedding new_emb(emb);
					unsigned pid = find_motif_pattern_id(n, i, dst, emb);
					//if (emb.get_qpid() == 0) accumulators[6] += 1;
					//else accumulators[7] += 1;
					new_emb.push_back(dst);
					new_emb.set_qpid(pid);
					queue.push_back(new_emb);
					//emb.pop_back();
				}
			}
		}
	}
	// Given an embedding, extend it with one more vertex. Used for k-cliques
	void extend_vertex(BaseEmbedding emb, BaseEmbeddingQueue &queue) {
		unsigned n = emb.size();
		for(unsigned i = 0; i < n; ++i) {
			VertexId id = emb.get_vertex(i);
			for(auto e : graph->edges(id)) {
				GNode dst = graph->getEdgeDst(e);
				// extend vertex in ascending order to avoid unnecessary enumeration
				if(dst > emb.get_vertex(n-1)) {
					emb.push_back(dst);
					queue.push_back(emb);
					emb.pop_back();
				}
			}
		}
	}
	// Given an embedding, extend it with one more vertex. Used for k-cliques.
	void extend_vertex_clique(BaseEmbedding emb, BaseEmbeddingQueue &queue, UintAccu &num, bool need_update = true) {
		unsigned n = emb.size();
		VertexId src = emb.get_vertex(n-1); // toExpand
		for (auto e : graph->edges(src)) {
			GNode dst = graph->getEdgeDst(e);
			if (dst > src) { // toAdd
				if (is_all_connected(dst, emb)) {
					if (need_update) {
						emb.push_back(dst);
						queue.push_back(emb);
						emb.pop_back();
					} else num += 1;
				}
			}
		}
	}
	void aggregate_clique(BaseEmbeddingQueue &in_queue, BaseEmbeddingQueue &out_queue) {
		SimpleMap simple_agg;
		for (const BaseEmbedding emb : in_queue) {
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
	// check each embedding to find the cliques
	void aggregate_clique_each(BaseEmbedding emb, SimpleMap& sm, BaseEmbeddingQueue &out_queue) {
		auto it = sm.find(emb);
		if(it != sm.end()) {
			// check if this is a clique
			if(it->second == it->first.size() - 2) {
				out_queue.push_back(emb);
				sm.erase(it);
			}
			else sm[emb] += 1;
		}
		else sm[emb] = 1;
	}
	void aggregate_motif_each(const VertexEmbedding &emb, std::vector<UintAccu> &accumulators) {
		unsigned n = emb.size();
		for (unsigned i = 0; i < n; ++i) {
			VertexId src = emb.get_vertex(i);
			for (auto e : graph->edges(src)) {
				GNode dst = graph->getEdgeDst(e);
				if (!is_vertexInduced_automorphism(emb, i, src, dst)) {
					unsigned pid = find_motif_pattern_id(n, i, dst, emb);
					accumulators[pid] += 1;
				}
			}
		}
	}
	void quick_aggregate(EdgeEmbeddingQueue &queue, QpMapFreq &qp_map) {
		for (auto emb : queue) {
			QPattern qp(emb);
			if (qp_map.find(qp) != qp_map.end()) {
				qp_map[qp] += 1;
				qp.clean();
			} else qp_map[qp] = 1;
		}
	}
	void quick_aggregate(EdgeEmbeddingQueue &queue, QpMapDomain &qp_map) {
		for (auto emb : queue) {
			QPattern qp(emb);
			if (qp_map.find(qp) != qp_map.end()) {
				for (unsigned i = 0; i < emb.size(); i ++)
					qp_map[qp][i].insert(emb.get_vertex(i));
				qp.clean();
			} else {
				qp_map[qp].resize(emb.size());
				for (unsigned i = 0; i < emb.size(); i ++)
					qp_map[qp][i].insert(emb.get_vertex(i));
			}
		}
	}
	// aggregate embeddings into quick patterns
	inline void quick_aggregate_each(EdgeEmbedding& emb, QpMapFreq& qp_map) {
		// turn this embedding into its quick pattern
		QPattern qp(emb);
		// update frequency for this quick pattern
		if (qp_map.find(qp) != qp_map.end()) {
			// if this quick pattern already exists, increase its count
			qp_map[qp] += 1;
			emb.set_qpid(qp.get_id());
			qp.clean();
		// otherwise add this quick pattern into the map, and set the count as one
		} else {
			qp_map[qp] = 1;
			emb.set_qpid(qp.get_id());
		}
	}
	/*
	inline void quick_aggregate_each(VertexEmbedding& emb, QpSMapFreq& qp_map) {
		QPSimple qp(emb);
		if (qp_map.find(qp) != qp_map.end()) {
			qp_map[qp] += 1;
			qp.clean();
		} else qp_map[qp] = 1;
	}
	*/
	inline void quick_aggregate_each(EdgeEmbedding& emb, QpMapDomain& qp_map) {
		QPattern qp(emb);
		bool qp_existed = false;
		auto it = qp_map.find(qp);
		if (it == qp_map.end()) {
			qp_map[qp].resize(emb.size());
			emb.set_qpid(qp.get_id());
		} else {
			qp_existed = true;
			emb.set_qpid((it->first).get_id());
		}
		for (unsigned i = 0; i < emb.size(); i ++)
			qp_map[qp][i].insert(emb.get_vertex(i));
		if (qp_existed) qp.clean();
	}
	void canonical_aggregate(QpMapFreq qp_map, CgMapFreq &cg_map) {
		for (auto it = qp_map.begin(); it != qp_map.end(); ++it) {
			QPattern qp = it->first;
			unsigned freq = it->second;
			CPattern* cg = turn_canonical_graph(qp, false);
			qp.clean();
			if (cg_map.find(*cg) != cg_map.end()) cg_map[*cg] += freq;
			else cg_map[*cg] = freq;
			delete cg;
		}
	}
	// aggregate quick patterns into canonical patterns.
	inline void canonical_aggregate_each(QPattern qp, Frequency freq, CgMapFreq &cg_map) {
		// turn the quick pattern into its canonical pattern
		CPattern* cg = turn_canonical_graph(qp, false);
		qp.clean();
		// if this pattern already exists, increase its count
		if (cg_map.find(*cg) != cg_map.end()) cg_map[*cg] += freq;
		// otherwise add this pattern into the map, and set the count as 'freq'
		else cg_map[*cg] = freq;
		delete cg;
	}
	// aggregate quick patterns into canonical patterns. Construct an id_map from quick pattern ID (qp_id) to canonical pattern ID (cg_id)
	void canonical_aggregate_each(QPattern qp, Frequency freq, CgMapFreq &cg_map, UintMap &id_map) {
		// turn the quick pattern into its canonical pattern
		CPattern* cg = turn_canonical_graph(qp, false);
		assert(cg != NULL);
		int qp_id = qp.get_id();
		int cg_id = cg->get_id();
		slock.lock();
		id_map.insert(std::make_pair(qp_id, cg_id));
		slock.unlock();
		qp.clean();
		// if this pattern already exists, increase its count
		auto it = cg_map.find(*cg);
		if (it != cg_map.end()) {
			cg_map[*cg] += freq;
			//qp.set_cgid(cg->get_id());
		// otherwise add this pattern into the map, and set the count as 'freq'
		} else {
			cg_map[*cg] = freq;
			//cg_map.insert(std::make_pair(*cg, freq));
			//qp.set_cgid((it->first).get_id());
		}
		delete cg;
	}
	void canonical_aggregate_each(QPattern qp, DomainSupport domainSets, CgMapDomain& cg_map, UintMap &id_map) {
		assert(qp.get_size() == domainSets.size());
		unsigned numDomains = qp.get_size();
		// turn the quick pattern into its canonical pattern
		CPattern* cg = turn_canonical_graph(qp, false);
		int qp_id = qp.get_id();
		int cg_id = cg->get_id();
		slock.lock();
		id_map.insert(std::make_pair(qp_id, cg_id));
		slock.unlock();
		auto it = cg_map.find(*cg);
		if (it == cg_map.end()) {
			cg_map[*cg].resize(numDomains);
			qp.set_cgid(cg->get_id());
		} else {
			qp.set_cgid((it->first).get_id());
		}
		for (unsigned i = 0; i < numDomains; i ++) {
			unsigned qp_idx = cg->get_quick_pattern_index(i);
			assert(qp_idx >= 0 && qp_idx < numDomains);
			cg_map[*cg][i].insert(domainSets[qp_idx].begin(), domainSets[qp_idx].end());
		}
		delete cg;
	}
	// check if the pattern of each embedding in the queue is frequent
	void filter(EdgeEmbeddingQueue &in_queue, CgMapFreq &cg_map, EdgeEmbeddingQueue &out_queue) {
		for (auto emb : in_queue) {
			QPattern qp(emb);
			//turn_quick_pattern_pure(emb, qp);
			CPattern* cf = turn_canonical_graph(qp, false);
			qp.clean();
			assert(cg_map.find(*cf) != cg_map.end());
			if(cg_map[*cf] >= threshold) out_queue.push_back(emb);
			delete cf;
		}
	}
	// filtering for FSM
	// check if the pattern of a given embedding is frequent, if yes, insert it to the queue
	void filter_each(EdgeEmbedding &emb, CgMapFreq &cg_map, EdgeEmbeddingQueue &out_queue) {
		// find the quick pattern of this embedding
		QPattern qp(emb);
		// find the pattern (canonical graph) of this embedding
		CPattern* cf = turn_canonical_graph(qp, false);
		qp.clean();
		//assert(cg_map.find(*cf) != cg_map.end());
		// compare the count of this pattern with the threshold
		// if the pattern is frequent, insert this embedding into the task queue
		if (cg_map[*cf] >= threshold) out_queue.push_back(emb);
		delete cf;
	}
	void filter(EdgeEmbeddingQueue &in_queue, CgMapDomain &cg_map, EdgeEmbeddingQueue &out_queue) {
		for (auto emb : in_queue) {
			QPattern qp(emb);
			CPattern* cf = turn_canonical_graph(qp, false);
			qp.clean();
			assert(cg_map.find(*cf) != cg_map.end());
			bool is_frequent = true;
			unsigned numOfDomains = cg_map[*cf].size();
			for (unsigned i = 0; i < numOfDomains; i ++) {
				if (cg_map[*cf][i].size() < threshold) {
					is_frequent = false;
					break;
				}
			}
			if (is_frequent) out_queue.push_back(emb);
			delete cf;
		}
	}
	void filter_each(EdgeEmbedding &emb, CgMapDomain &cg_map, EdgeEmbeddingQueue &out_queue) {
		QPattern qp(emb);
		CPattern* cf = turn_canonical_graph(qp, false);
		qp.clean();
		//assert(cg_map.find(*cf) != cg_map.end());
		bool is_frequent = true;
		unsigned numOfDomains = cg_map[*cf].size();
		for (unsigned i = 0; i < numOfDomains; i ++) {
			if (cg_map[*cf][i].size() < threshold) {
				is_frequent = false;
				break;
			}
		}
		if (is_frequent) out_queue.push_back(emb);
		delete cf;
	}
	inline void filter(EdgeEmbeddingQueue &in_queue, const UintMap id_map, const UintMap support_map, EdgeEmbeddingQueue &out_queue) {
		for (auto emb : in_queue) {
			unsigned qp_id = emb.get_qpid();
			unsigned cg_id = id_map.at(qp_id);
			if (support_map.at(cg_id) >= threshold) out_queue.push_back(emb);
		}
	}
	inline void filter_each(EdgeEmbedding emb, const UintMap id_map, const UintMap support_map, EdgeEmbeddingQueue &out_queue) {
		unsigned qp_id = emb.get_qpid();
		unsigned cg_id = id_map.at(qp_id);
		if (support_map.at(cg_id) >= threshold) out_queue.push_back(emb);
	}
	void set_threshold(unsigned minsup) { threshold = minsup; }
	unsigned get_total_num_cliques() { return num_cliques; }
	void printout_agg(const CgMapFreq cg_map) {
		for (auto it = cg_map.begin(); it != cg_map.end(); ++it)
			std::cout << "{" << it->first << " --> " << it->second << std::endl;
	}
	void printout_motifs(std::vector<UintAccu> &accumulators) {
		std::cout << std::endl;
		if (accumulators.size() == 2) {
			std::cout << "\ttriangles --> " << accumulators[0].reduce() << std::endl;
			std::cout << "\t3-chains --> " << accumulators[1].reduce() << std::endl;
		} else {
			std::cout << "\t4-paths --> " << accumulators[0].reduce() << std::endl;
			std::cout << "\t3-stars --> " << accumulators[1].reduce() << std::endl;
			std::cout << "\t4-cycles --> " << accumulators[2].reduce() << std::endl;
			std::cout << "\ttailed-triangles --> " << accumulators[3].reduce() << std::endl;
			std::cout << "\tdiamonds --> " << accumulators[4].reduce() << std::endl;
			std::cout << "\t4-cliques --> " << accumulators[5].reduce() << std::endl;
		}
		std::cout << std::endl;
	}
	unsigned support_count(const CgMapDomain cg_map, UintMap &support_map) {
		unsigned count = 0;
		for (auto it = cg_map.begin(); it != cg_map.end(); ++it) {
			unsigned support = get_support(it->second);
			support_map.insert(std::make_pair(it->first.get_id(), support));
			if (support >= threshold) count ++;
		}
		return count;
	}
	unsigned support_count(const CgMapFreq cg_map, UintMap &support_map) {
		unsigned count = 0;
		for (auto it = cg_map.begin(); it != cg_map.end(); ++it) {
			unsigned support = it->second;
			support_map.insert(std::make_pair(it->first.get_id(), support));
			if (support >= threshold) count ++;
		}
		return count;
	}
	// counting the minimal image based support
	unsigned get_support(HashIntSets domainSets) {
		unsigned numDomains = domainSets.size();
		unsigned support = 0xFFFFFFFF;
		// get the minimal domain size
		for (unsigned j = 0; j < numDomains; j ++)
			if (domainSets[j].size() < support)
				support = domainSets[j].size();
		return support;
	}
	void printout_agg(const CgMapDomain cg_map) {
		std::vector<unsigned> support(cg_map.size());
		int i = 0;
		for (auto it = cg_map.begin(); it != cg_map.end(); ++it) {
			support[i] = get_support(it->second);
			i ++;
		}
		i = 0;
		for (auto it = cg_map.begin(); it != cg_map.end(); ++it) {
			std::cout << "{" << it->first << " --> " << support[i] << std::endl;
			i ++;
		}
	}

private:
	Graph *graph;
	unsigned threshold;
	unsigned num_cliques;
	galois::StatTimer Tconnect;
	std::vector<unsigned> degrees;
	galois::substrate::SimpleLock slock;

	inline bool is_all_connected(unsigned dst, BaseEmbedding emb) {
		unsigned n = emb.size();
		bool all_connected = true;
#if 0
		unsigned num_edges = 0;
		for(auto e2 : graph->edges(dst)) {
			GNode dst_dst = graph->getEdgeDst(e2);
			for(unsigned i = 0; i < n; ++i) {
				if (dst_dst == emb.get_vertex(i)) {
					num_edges ++;
					break;
				}
			}
		}
		if(num_edges != n) all_connected = false;
#else
		for(unsigned i = 0; i < n-1; ++i) {
			unsigned from = emb.get_vertex(i);
			if (!is_connected(from, dst)) {
				all_connected = false;
				break;
			}
		}
#endif
		return all_connected;
	}
	inline bool is_connected(unsigned from, unsigned to) {
		bool connected = false;
		if (degrees[from] < degrees[to]) {
			for(auto e : graph->edges(from)) {
				GNode dst = graph->getEdgeDst(e);
				if (dst == to) {
					connected = true;
					break;
				}
			}
		} else {
			for(auto e : graph->edges(to)) {
				GNode dst = graph->getEdgeDst(e);
				if (dst == from) {
					connected = true;
					break;
				}
			}
		}
		return connected;
	}
	void degree_counting() {
		degrees.resize(graph->size());
		galois::do_all(galois::iterate(graph->begin(), graph->end()),
			[&] (GNode v) {
				degrees[v] = std::distance(graph->edge_begin(v), graph->edge_end(v));
			},
			galois::loopname("DegreeCounting")
		);
	}
	inline bool is_vertexInduced_automorphism(const VertexEmbedding& emb, unsigned idx, VertexId src, VertexId dst) {
		unsigned n = emb.size();
		// the new vertex id should be larger than the first vertex id
		if (dst <= emb.get_vertex(0)) return true;
		// the new vertex should not already exist in the embedding
		for (unsigned i = 1; i < n; ++i)
			if (dst == emb.get_vertex(i)) return true;
		// the new vertex should not already be expanded by any previous vertex in the embedding
		for (unsigned i = 0; i < idx; ++i)
			if (is_connected(emb.get_vertex(i), dst)) return true;
		// the new vertex id should be larger than any vertex id after its source vertex in the embedding
		for (unsigned i = idx+1; i < n; ++i)
			if (dst < emb.get_vertex(i)) return true;
		return false;
	}
	inline bool is_edgeInduced_automorphism(const EdgeEmbedding & emb, BYTE history, VertexId src, VertexId dst, const bool vertex_existed) {
		//check with the first element
		if(dst < emb.get_vertex(0)) return true;
		//check loop edge
		if(dst == emb.get_vertex(emb.get_history(history))) return true;
		//check to see if there already exists the vertex added; if so, just allow to add edge which is (smaller id -> bigger id)
		if(vertex_existed && src > dst) return true;
		std::pair<VertexId, VertexId> added_edge(src, dst);
		for(unsigned index = history + 1; index < emb.size(); ++index) {
			std::pair<VertexId, VertexId> edge;
			getEdge(emb, index, edge);
			int cmp = compare(added_edge, edge);
			if(cmp <= 0) return true;
		}
		return false;
	}
	inline unsigned find_motif_pattern_id(unsigned n, unsigned idx, VertexId dst, const VertexEmbedding& emb) {
		unsigned pid = 0;
		if (n == 2) { // count 3-motifs
			pid = 1; // 3-chain
			if(idx == 0 && is_connected(emb.get_vertex(1), dst)) pid = 0; // triangle
		} else if (n == 3) { // count 4-motifs
			unsigned num_edges = 1;
			pid = emb.get_qpid();
			//std::cout << "Embedding: " << emb << ", " << "adding " << dst << " at " << idx << "\n";
			if (pid == 0) { // expanding a triangle
				for (unsigned j = 0; j < n; j ++)
					if (j != idx && is_connected(emb.get_vertex(j), dst)) num_edges ++;
				pid = num_edges + 2; // p3: tailed-triangle; p4: diamond; p5: 4-clique
				//if (pid == 3) std::cout << "tailed-triangle embedding: " << emb << ", " << "adding " << dst << " at " << idx << "\n";
				//if (pid == 4) std::cout << "diamond embedding: " << emb << ", " << "adding " << dst << " at " << idx << "\n";
				//if (pid == 5) std::cout << "4-clique embedding: " << emb << ", " << "adding " << dst << " at " << idx << "\n";
			} else { // expanding a 3-chain
				assert(pid == 1);
				std::vector<bool> connected(3, false);
				connected[idx] = true;
				for (unsigned j = 0; j < n; j ++) {
					if (j != idx && is_connected(emb.get_vertex(j), dst)) {
						num_edges ++;
						connected[j] = true;
					}
				}
				if (num_edges == 1) {
					unsigned center = is_connected(emb.get_vertex(1), emb.get_vertex(2)) ? 1 : 0;
					if (idx == center) {
						pid = 1; // p1: 3-star
						//std::cout << "3-star embedding: " << emb << ", " << "adding " << dst << " at " << idx << "\n";
					} else {
						pid = 0; // p0: 3-path
						//std::cout << "3-path embedding: " << emb << ", " << "adding " << dst << " at " << idx << "\n";
					}
				} else if (num_edges == 2) {
					unsigned center = is_connected(emb.get_vertex(1), emb.get_vertex(2)) ? 1 : 0;
					if (connected[center]) {
						pid = 3; // p3: tailed-triangle
						//std::cout << "tailed-triangle embedding: " << emb << ", " << "adding " << dst << " at " << idx << "\n";
					} else {
						pid = 2; // p2: 4-cycle
						//std::cout << "4-cycle embedding: " << emb << ", " << "adding " << dst << " at " << idx << "\n";
					}
				} else {
					pid = 4; // p4: diamond
					//std::cout << "diamond: " << emb << ", " << "adding " << dst << " at " << idx << "\n";
				}
			}
		} else {
			std::cout << "5-motif and beyond are not supported for now\n";
		}
		return pid;
	}
	inline bool edge_existed(const EdgeEmbedding & emb, BYTE history, VertexId src, VertexId dst) {
		std::pair<VertexId, VertexId> added_edge(src, dst);
		for(unsigned i = 1; i < emb.size(); ++i) {
			if(emb.get_vertex(i) == dst && emb.get_vertex(emb.get_history(i)) == src)
				return true;
		}
		return false;
	}
	static void report_aut(void* param, const unsigned n, const unsigned* aut) {
		assert(param);
		//fprintf((FILE*) param, "Generator: ");
		//bliss::print_permutation((FILE*) param, n, aut, 1);
		//fprintf((FILE*) param, "\n");
	}
/*
	bliss::AbstractGraph* turn_canonical_graph_bliss(QPattern & qp, const bool is_directed) {
		bliss::AbstractGraph* ag = 0;
		// read graph from quick pattern
		ag = readGraph(qp, is_directed);
		// turn to canonical form
		bliss::AbstractGraph* cf = turnCanonical(ag);
		delete ag;
		ag = 0;
		return cf;
	}
	bliss::AbstractGraph* turnCanonical(bliss::AbstractGraph* ag) {
		bliss::Stats stats;
		const unsigned * cl = ag->canonical_form(stats, &report_aut, stdout); // canonical labeling. This is expensive.
		bliss::AbstractGraph* cf = ag->permute(cl); //permute to canonical form
		return cf;
	}
//*/
	CPattern* turn_canonical_graph(QPattern & qp, const bool is_directed) {
		//bliss::AbstractGraph* cf = turn_canonical_graph_bliss(qp, is_directed);
		bliss::AbstractGraph* ag = readGraph(qp, is_directed);
		bliss::Stats stats;
		const unsigned * cl = ag->canonical_form(stats, &report_aut, stdout); // canonical labeling. This is expensive.
		bliss::AbstractGraph* cf = ag->permute(cl); //permute to canonical form
		//bliss::AbstractGraph* cf = turnCanonical(ag);
		delete ag;
		CPattern* cg = new CPattern(cf, is_directed);
		delete cf;
		return cg;
	}
	bliss::AbstractGraph* readGraph(QPattern & qp, bool opt_directed) {
		bliss::AbstractGraph* g = 0;
		//get the number of vertices
		std::unordered_map<VertexId, BYTE> vertices;
		for(unsigned index = 0; index < qp.get_size(); ++index) {
			auto element = qp.at(index);
#ifdef ENABLE_LABEL
			vertices[element.get_vid()] = element.get_vlabel();
#else
			vertices[element.get_vid()] = 0;
#endif
		}
		//std::cout << "Transforming quick_pattern: " << qp << ", num_vertices = " << vertices.size() << "\n";
		//construct graph
		const unsigned number_vertices = vertices.size();
		assert(!opt_directed);
		//if(opt_directed) g = new bliss::Digraph(vertices.size());
		//else
			g = new bliss::Graph(vertices.size());
		//set vertices
		for(unsigned i = 0; i < number_vertices; ++i)
			g->change_color(i, (unsigned)vertices[i + 1]);
		//read edges
		assert(qp.get_size() > 1);
		for(unsigned index = 1; index < qp.get_size(); ++index) {
			auto element = qp.at(index);
			//std::cout << "element: " << element << "\n";
			VertexId from, to;
			from = qp.at(element.get_his()).get_vid();
			to = element.get_vid();
			//std::cout << "add edge (" << from << "," << to << ")\n";
			g->add_edge(from - 1, to - 1, std::make_pair((unsigned)element.get_his(), index));
		}
		return g;
	}
	inline void getEdge(const EdgeEmbedding & emb, unsigned index, std::pair<VertexId, VertexId>& edge) {
		edge.first = emb.get_vertex(emb.get_history(index));
		edge.second = emb.get_vertex(index);
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

#endif // MINER_HPP_
