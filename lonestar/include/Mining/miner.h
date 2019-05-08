#ifndef MINER_HPP_
#define MINER_HPP_
#include "quick_pattern.h"
#include "canonical_graph.h"
#include "galois/Bag.h"
#include "galois/Galois.h"
#include "galois/substrate/PerThreadStorage.h"
#include "galois/substrate/SimpleLock.h"

// We provide two types of 'support': frequency and domain support.
// Frequency is used for counting, e.g. motif counting.
// Domain support, a.k.a, the minimum image-based support, is used for FSM. It has the anti-monotonic property.
typedef unsigned Frequency;
typedef HashIntSets DomainSupport;
typedef std::unordered_map<QuickPattern, Frequency> QpMapFreq; // mapping quick pattern to its frequency
typedef std::unordered_map<CanonicalGraph, Frequency> CgMapFreq; // mapping canonical pattern to its frequency
typedef std::unordered_map<QuickPattern, DomainSupport> QpMapDomain; // mapping quick pattern to its domain support
typedef std::unordered_map<CanonicalGraph, DomainSupport> CgMapDomain; // mapping canonical pattern to its domain support
typedef std::unordered_map<BaseEmbedding, Frequency> SimpleMap;
//typedef CTSL::HashMap<BaseEmbedding, Frequency> SimpleConcurrentMap;
//typedef utils::concurrent_map<BaseEmbedding, Frequency> SimpleConcurrentMap;
typedef galois::substrate::PerThreadStorage<QpMapDomain> LocalQpMapDomain;
typedef galois::substrate::PerThreadStorage<CgMapDomain> LocalCgMapDomain;
typedef galois::substrate::PerThreadStorage<QpMapFreq> LocalQpMapFreq;
typedef galois::substrate::PerThreadStorage<CgMapFreq> LocalCgMapFreq;
typedef galois::substrate::PerThreadStorage<SimpleMap> LocalSimpleMap;
typedef galois::InsertBag<Embedding> EmbeddingQueue;
typedef galois::InsertBag<int> IntQueue;
typedef galois::InsertBag<BaseEmbedding> BaseEmbeddingQueue;
//typedef std::vector<Embedding> EmbeddingVec;
typedef galois::gstl::Vector<Embedding> EmbeddingVec;
//typedef galois::gstl::Vector<EmbeddingQueue> EmbeddingLists;
typedef std::vector<EmbeddingQueue> EmbeddingLists;
typedef galois::gstl::Vector<int> IntVec; // quick pattern ID list, used for each canonical pattern
typedef galois::gstl::Vector<IntQueue> QpLists;

class Miner {
public:
	Miner(Graph *g, unsigned emb_size = 16) : embedding_size(emb_size) {
		graph = g;
		num_cliques = 0;
	}
	virtual ~Miner() {};
	// given an embedding, extend it with one more edge, and if it is not automorphism, insert the new embedding into the task queue
	void extend_edge(unsigned max_size, Embedding emb, EmbeddingQueue &queue) {
		unsigned size = emb.size();
		// get the number of distinct vertices in the embedding
		std::unordered_set<VertexId> vertices_set;
		vertices_set.reserve(size);
		for(unsigned i = 0; i < size; i ++) vertices_set.insert(emb[i].vertex_id);
		std::unordered_set<VertexId> set; // uesd to make sure each distinct vertex is expanded only once
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
					if(num_vertices <= max_size && !is_automorphism(emb, i, id, dst, vertex_existed)) {
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
	// extend edge when input graph is DAG (no automorphism check is required)
	void extend_edge_DAG(unsigned max_size, Embedding emb, EmbeddingQueue &queue) {
		unsigned size = emb.size();
		std::unordered_set<VertexId> vertices_set;
		vertices_set.reserve(size);
		for(unsigned i = 0; i < size; i ++) vertices_set.insert(emb[i].vertex_id);
		std::unordered_set<VertexId> set; // uesd to make sure each distinct vertex is expanded only once
		for(unsigned i = 0; i < size; ++i) {
			VertexId id = emb[i].vertex_id;
			assert(id >= 0 && id < graph->size());
			if(set.find(id) == set.end()) {
				set.insert(id);
				for(auto e : graph->edges(id)) {
					GNode dst = graph->getEdgeDst(e);
					auto dst_label = 0, edge_label = 0;
					#ifdef ENABLE_LABEL
					dst_label = graph->getData(dst);
					//edge_label = graph->getEdgeData(e);
					#endif
					auto num_vertices = vertices_set.size();
					bool vertex_existed = true;
					if(vertices_set.find(dst) == vertices_set.end()) {
						num_vertices ++;
						vertex_existed = false;
					}
					//if(num_vertices <= max_size) {
					if(num_vertices <= max_size && (!vertex_existed || !edge_existed(emb, i, id, dst))) {
						ElementType new_element(dst, (BYTE)num_vertices, edge_label, dst_label, (BYTE)i);
						emb.push_back(new_element);
						queue.push_back(emb);
						emb.pop_back();
					}
				}
			}
		}
	}
	// Given an embedding, extend it with one more vertex. Used for k-cliques
	void extend_vertex(BaseEmbedding emb, BaseEmbeddingQueue &queue) {
		unsigned n = emb.size();
		for(unsigned i = 0; i < n; ++i) {
			SimpleElement id = emb[i];
			for(auto e : graph->edges(id)) {
				GNode dst = graph->getEdgeDst(e);
				// extend vertex in ascending order to avoid unnecessary enumeration
				if(dst > emb[n-1]) {
					emb.push_back(dst);
					queue.push_back(emb);
					emb.pop_back();
				}
			}
		}
	}
	// Given an embedding, extend it with one more vertex. Used for k-cliques.
	void extend_vertex_clique(BaseEmbedding emb, BaseEmbeddingQueue &queue) {
		unsigned n = emb.size();
		SimpleElement src = emb[n-1];
		for(auto e1 : graph->edges(src)) {
			GNode dst = graph->getEdgeDst(e1);
			if(dst > src) {
				unsigned num_edges = 0;
				for(auto e2 : graph->edges(dst)) {
					GNode dst_dst = graph->getEdgeDst(e2);
					for(unsigned i = 0; i < n; ++i) {
						if (dst_dst == emb[i]) {
							num_edges ++;
							break;
						}
					}
				}
				if(num_edges == n) {
					emb.push_back(dst);
					queue.push_back(emb);
					emb.pop_back();
				}
			}
		}
	}
	void quick_aggregate(EmbeddingQueue &queue, QpMapFreq &qp_map) {
		for (auto emb : queue) {
			QuickPattern qp(emb);
			if (qp_map.find(qp) != qp_map.end()) {
				qp_map[qp] += 1;
				qp.clean();
			} else qp_map[qp] = 1;
		}
	}
	void quick_aggregate(EmbeddingQueue &queue, QpMapDomain &qp_map) {
		for (auto emb : queue) {
			QuickPattern qp(emb);
			if (qp_map.find(qp) != qp_map.end()) {
				for (unsigned i = 0; i < emb.size(); i ++)
					qp_map[qp][i].insert(emb[i].vertex_id);
				qp.clean();
			} else {
				qp_map[qp].resize(emb.size());
				for (unsigned i = 0; i < emb.size(); i ++)
					qp_map[qp][i].insert(emb[i].vertex_id);
			}
		}
	}
	// aggregate embeddings into quick patterns
	inline void quick_aggregate_each(Embedding& emb, QpMapFreq& qp_map) {
		// turn this embedding into its quick pattern
		QuickPattern qp(emb);
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
	inline void quick_aggregate_each(Embedding& emb, QpMapDomain& qp_map) {
		QuickPattern qp(emb);
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
			qp_map[qp][i].insert(emb[i].vertex_id);
		if (qp_existed) qp.clean();
	}
	void canonical_aggregate(QpMapFreq qp_map, CgMapFreq &cg_map) {
		for (auto it = qp_map.begin(); it != qp_map.end(); ++it) {
			QuickPattern qp = it->first;
			unsigned freq = it->second;
			CanonicalGraph* cg = turn_canonical_graph(qp, false);
			qp.clean();
			if (cg_map.find(*cg) != cg_map.end()) cg_map[*cg] += freq;
			else cg_map[*cg] = freq;
			delete cg;
		}
	}
	// aggregate quick patterns into canonical patterns.
	inline void canonical_aggregate_each(QuickPattern qp, Frequency freq, CgMapFreq &cg_map) {
		// turn the quick pattern into its canonical pattern
		CanonicalGraph* cg = turn_canonical_graph(qp, false);
		qp.clean();
		// if this pattern already exists, increase its count
		if (cg_map.find(*cg) != cg_map.end()) cg_map[*cg] += freq;
		// otherwise add this pattern into the map, and set the count as 'freq'
		else cg_map[*cg] = freq;
		delete cg;
	}
	// aggregate quick patterns into canonical patterns. Construct an id_map from QuickPattern ID (qp_id) to CanonicalGraph ID (cg_id)
	void canonical_aggregate_each(QuickPattern qp, Frequency freq, CgMapFreq &cg_map, UintMap &id_map) {
		// turn the quick pattern into its canonical pattern
		CanonicalGraph* cg = turn_canonical_graph(qp, false);
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
	void canonical_aggregate_each(QuickPattern qp, DomainSupport domainSets, CgMapDomain& cg_map, UintMap &id_map) {
		assert(qp.get_size() == domainSets.size());
		unsigned numDomains = qp.get_size();
		// turn the quick pattern into its canonical pattern
		CanonicalGraph* cg = turn_canonical_graph(qp, false);
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
	// check if the pattern of each embedding in the queue is frequent
	void filter(EmbeddingQueue &in_queue, CgMapFreq &cg_map, EmbeddingQueue &out_queue) {
		for (auto emb : in_queue) {
			QuickPattern qp(emb);
			//turn_quick_pattern_pure(emb, qp);
			CanonicalGraph* cf = turn_canonical_graph(qp, false);
			qp.clean();
			assert(cg_map.find(*cf) != cg_map.end());
			if(cg_map[*cf] >= threshold) out_queue.push_back(emb);
			delete cf;
		}
	}
	// filtering for FSM
	// check if the pattern of a given embedding is frequent, if yes, insert it to the queue
	void filter_each(Embedding &emb, CgMapFreq &cg_map, EmbeddingQueue &out_queue) {
		// find the quick pattern of this embedding
		QuickPattern qp(emb);
		// find the pattern (canonical graph) of this embedding
		CanonicalGraph* cf = turn_canonical_graph(qp, false);
		qp.clean();
		//assert(cg_map.find(*cf) != cg_map.end());
		// compare the count of this pattern with the threshold
		// if the pattern is frequent, insert this embedding into the task queue
		if (cg_map[*cf] >= threshold) out_queue.push_back(emb);
		delete cf;
	}
	void filter(EmbeddingQueue &in_queue, CgMapDomain &cg_map, EmbeddingQueue &out_queue) {
		for (auto emb : in_queue) {
			QuickPattern qp(emb);
			CanonicalGraph* cf = turn_canonical_graph(qp, false);
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
	void filter_each(Embedding &emb, CgMapDomain &cg_map, EmbeddingQueue &out_queue) {
		QuickPattern qp(emb);
		CanonicalGraph* cf = turn_canonical_graph(qp, false);
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
	inline void filter(EmbeddingQueue &in_queue, const UintMap id_map, const UintMap support_map, EmbeddingQueue &out_queue) {
		for (auto emb : in_queue) {
			unsigned qp_id = emb.get_qpid();
			unsigned cg_id = id_map.at(qp_id);
			if (support_map.at(cg_id) >= threshold) out_queue.push_back(emb);
		}
	}
	inline void filter_each(Embedding emb, const UintMap id_map, const UintMap support_map, EmbeddingQueue &out_queue) {
		unsigned qp_id = emb.get_qpid();
		unsigned cg_id = id_map.at(qp_id);
		if (support_map.at(cg_id) >= threshold) out_queue.push_back(emb);
	}
	void set_threshold(unsigned minsup) { threshold = minsup; }
	void update_embedding_size() { embedding_size += sizeof(ElementType); }
	void update_base_embedding_size() { embedding_size += sizeof(SimpleElement); }
	inline unsigned get_embedding_size() { return embedding_size; }
	unsigned get_total_num_cliques() { return num_cliques; }
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
	void printout_agg(const CgMapFreq cg_map) {
		for (auto it = cg_map.begin(); it != cg_map.end(); ++it)
			std::cout << "{" << it->first << " --> " << it->second << std::endl;
		//std::cout << "num_patterns: " << cg_map.size() << " num_embeddings: " 
		//	<< std::distance(queue.begin(), queue.end()) << "\n";
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
	unsigned embedding_size;
	unsigned threshold;
	Graph *graph;
	unsigned num_cliques;
	galois::substrate::SimpleLock slock;
#if 0
	std::vector<LabeledEdge> edge_list;
	void construct_edgelist() {
		for (GNode src : *graph) {
			auto& src_label = graph->getData(src);
			for (auto e : graph->edges(src)) {
				GNode dst = graph->getEdgeDst(e);
				auto& dst_label = graph->getData(dst);
				LabeledEdge edge(src, dst, src_label, dst_label);
				edge_list.push_back(edge);
			}
		}
		assert(edge_list.size() == graph->sizeEdges());
	}
	void turn_quick_pattern_pure(const Embedding & emb, QuickPattern & qp) {
		std::memcpy(qp.get_elements(), emb.data(), qp.get_size() * sizeof(ElementType));
		std::unordered_map<VertexId, VertexId> map;
		VertexId new_id = 1;
		for(unsigned i = 0; i < qp.get_size(); i++) {
			auto& element = qp.at(i);
			VertexId old_id = element.vertex_id;
			auto iterator = map.find(old_id);
			if(iterator == map.end()) {
				element.set_vertex_id(new_id);
				map[old_id] = new_id++;
			} else element.set_vertex_id(iterator->second);
		}
	}
#endif
	inline bool is_automorphism(Embedding & emb, BYTE history, VertexId src, VertexId dst, const bool vertex_existed) {
		//check with the first element
		if(dst < emb.front().vertex_id) return true;
		//check loop edge
		if(dst == emb[emb[history].history_info].vertex_id) return true;
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
	inline bool edge_existed(Embedding & emb, BYTE history, VertexId src, VertexId dst) {
		std::pair<VertexId, VertexId> added_edge(src, dst);
		for(unsigned i = 1; i < emb.size(); ++i) {
			if(emb[i].vertex_id == dst && emb[emb[i].history_info].vertex_id == src)
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
	bliss::AbstractGraph* turn_canonical_graph_bliss(QuickPattern & qp, const bool is_directed) {
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
	CanonicalGraph* turn_canonical_graph(QuickPattern & qp, const bool is_directed) {
		//bliss::AbstractGraph* cf = turn_canonical_graph_bliss(qp, is_directed);
		bliss::AbstractGraph* ag = readGraph(qp, is_directed);
		bliss::Stats stats;
		const unsigned * cl = ag->canonical_form(stats, &report_aut, stdout); // canonical labeling. This is expensive.
		bliss::AbstractGraph* cf = ag->permute(cl); //permute to canonical form
		//bliss::AbstractGraph* cf = turnCanonical(ag);
		delete ag;
		CanonicalGraph* cg = new CanonicalGraph(cf, is_directed);
		delete cf;
		return cg;
	}
	bliss::AbstractGraph* readGraph(QuickPattern & qp, bool opt_directed) {
		bliss::AbstractGraph* g = 0;
		//get the number of vertices
		std::unordered_map<VertexId, BYTE> vertices;
		for(unsigned index = 0; index < qp.get_size(); ++index) {
			auto element = qp.at(index);
#ifdef ENABLE_LABEL
			vertices[element.vertex_id] = element.vertex_label;
#else
			vertices[element.vertex_id] = 0;
#endif
		}
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
		//std::cout << "read edges\n";
		assert(qp.get_size() > 1);
		for(unsigned index = 1; index < qp.get_size(); ++index) {
			auto element = qp.at(index);
			VertexId from = qp.at(element.history_info).vertex_id;
			VertexId to = element.vertex_id;
			//std::cout << "add edge " << index << " ...\n";
			g->add_edge(from - 1, to - 1, std::make_pair((unsigned)element.history_info, index));
			//std::cout << "edge added\n";
		}
		//std::cout << "done read edges\n";
		return g;
	}
	inline void getEdge(Embedding & emb, unsigned index, std::pair<VertexId, VertexId>& edge) {
		auto ele = emb[index];
		edge.first = emb[ele.history_info].vertex_id;
		edge.second = ele.vertex_id;
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
void printout_embeddings(int level, Miner& miner, BaseEmbeddingQueue& queue, bool verbose = false) {
#else
void printout_embeddings(int level, Miner& miner, EmbeddingQueue& queue, bool verbose = false) {
#endif
	int num_embeddings = std::distance(queue.begin(), queue.end());
	unsigned embedding_size = miner.get_embedding_size();
	std::cout << "Number of embeddings in level " << level << ": " << num_embeddings << " (embedding_size=" << embedding_size << ")" << std::endl;
	if(verbose) for (auto embedding : queue) miner.printout_embedding(level, embedding);
}

#endif /* MINER_HPP_ */
