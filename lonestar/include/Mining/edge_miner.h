#ifndef EDGE_MINER_H
#define EDGE_MINER_H
#include "miner.h"
typedef HashIntSets DomainSupport;
typedef std::unordered_map<QPattern, DomainSupport> QpMapDomain; // mapping quick pattern to its domain support
typedef std::unordered_map<CPattern, DomainSupport> CgMapDomain; // mapping canonical pattern to its domain support
typedef galois::substrate::PerThreadStorage<QpMapDomain> LocalQpMapDomain;
typedef galois::substrate::PerThreadStorage<CgMapDomain> LocalCgMapDomain;

class EdgeMiner : public Miner {
public:
	EdgeMiner(Graph *g) {
		graph = g;
	}
	virtual ~EdgeMiner() {}
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
			CPattern cg(qp);
			qp.clean();
			if (cg_map.find(cg) != cg_map.end()) cg_map[cg] += freq;
			else cg_map[cg] = freq;
			//delete cg;
		}
	}
	// aggregate quick patterns into canonical patterns.
	inline void canonical_aggregate_each(QPattern qp, Frequency freq, CgMapFreq &cg_map) {
		// turn the quick pattern into its canonical pattern
		CPattern cg(qp);
		qp.clean();
		// if this pattern already exists, increase its count
		if (cg_map.find(cg) != cg_map.end()) cg_map[cg] += freq;
		// otherwise add this pattern into the map, and set the count as 'freq'
		else cg_map[cg] = freq;
		//delete cg;
	}
	// aggregate quick patterns into canonical patterns. Construct an id_map from quick pattern ID (qp_id) to canonical pattern ID (cg_id)
	void canonical_aggregate_each(QPattern qp, Frequency freq, CgMapFreq &cg_map, UintMap &id_map) {
		// turn the quick pattern into its canonical pattern
		CPattern cg(qp);
		//assert(cg != NULL);
		int qp_id = qp.get_id();
		int cg_id = cg.get_id();
		slock.lock();
		id_map.insert(std::make_pair(qp_id, cg_id));
		slock.unlock();
		qp.clean();
		// if this pattern already exists, increase its count
		auto it = cg_map.find(cg);
		if (it != cg_map.end()) {
			cg_map[cg] += freq;
		// otherwise add this pattern into the map, and set the count as 'freq'
		} else {
			cg_map[cg] = freq;
		}
		//delete cg;
	}
	void canonical_aggregate_each(QPattern qp, DomainSupport domainSets, CgMapDomain& cg_map, UintMap &id_map) {
		assert(qp.get_size() == domainSets.size());
		unsigned numDomains = qp.get_size();
		// turn the quick pattern into its canonical pattern
		CPattern cg(qp);
		int qp_id = qp.get_id();
		int cg_id = cg.get_id();
		slock.lock();
		id_map.insert(std::make_pair(qp_id, cg_id));
		slock.unlock();
		auto it = cg_map.find(cg);
		if (it == cg_map.end()) {
			cg_map[cg].resize(numDomains);
			qp.set_cgid(cg.get_id());
		} else {
			qp.set_cgid((it->first).get_id());
		}
		for (unsigned i = 0; i < numDomains; i ++) {
			unsigned qp_idx = cg.get_quick_pattern_index(i);
			assert(qp_idx >= 0 && qp_idx < numDomains);
			cg_map[cg][i].insert(domainSets[qp_idx].begin(), domainSets[qp_idx].end());
		}
		//delete cg;
	}
	// check if the pattern of each embedding in the queue is frequent
	void filter(EdgeEmbeddingQueue &in_queue, CgMapFreq &cg_map, EdgeEmbeddingQueue &out_queue) {
		for (auto emb : in_queue) {
			QPattern qp(emb);
			CPattern cg(qp);
			qp.clean();
			assert(cg_map.find(cg) != cg_map.end());
			if(cg_map[cg] >= threshold) out_queue.push_back(emb);
			//delete cf;
		}
	}
	// filtering for FSM
	// check if the pattern of a given embedding is frequent, if yes, insert it to the queue
	void filter_each(EdgeEmbedding &emb, CgMapFreq &cg_map, EdgeEmbeddingQueue &out_queue) {
		// find the quick pattern of this embedding
		QPattern qp(emb);
		// find the pattern (canonical graph) of this embedding
		CPattern cg(qp);
		qp.clean();
		// compare the count of this pattern with the threshold
		// if the pattern is frequent, insert this embedding into the task queue
		if (cg_map[cg] >= threshold) out_queue.push_back(emb);
		//delete cf;
	}
	void filter(EdgeEmbeddingQueue &in_queue, CgMapDomain &cg_map, EdgeEmbeddingQueue &out_queue) {
		for (auto emb : in_queue) {
			QPattern qp(emb);
			CPattern cg(qp);
			qp.clean();
			assert(cg_map.find(cg) != cg_map.end());
			bool is_frequent = true;
			unsigned numOfDomains = cg_map[cg].size();
			for (unsigned i = 0; i < numOfDomains; i ++) {
				if (cg_map[cg][i].size() < threshold) {
					is_frequent = false;
					break;
				}
			}
			if (is_frequent) out_queue.push_back(emb);
			//delete cf;
		}
	}
	void filter_each(EdgeEmbedding &emb, CgMapDomain &cg_map, EdgeEmbeddingQueue &out_queue) {
		QPattern qp(emb);
		CPattern cg(qp);
		qp.clean();
		//assert(cg_map.find(*cf) != cg_map.end());
		bool is_frequent = true;
		unsigned numOfDomains = cg_map[cg].size();
		for (unsigned i = 0; i < numOfDomains; i ++) {
			if (cg_map[cg][i].size() < threshold) {
				is_frequent = false;
				break;
			}
		}
		if (is_frequent) out_queue.push_back(emb);
		//delete cf;
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
	void printout_agg(const CgMapFreq cg_map) {
		for (auto it = cg_map.begin(); it != cg_map.end(); ++it)
			std::cout << "{" << it->first << " --> " << it->second << std::endl;
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

private:
	unsigned threshold;
	galois::substrate::SimpleLock slock;
/*
	static void report_aut(void* param, const unsigned n, const unsigned* aut) {
		assert(param);
		//fprintf((FILE*) param, "Generator: ");
		//bliss::print_permutation((FILE*) param, n, aut, 1);
		//fprintf((FILE*) param, "\n");
	}
	template <typename EmbeddingTy = EdgeEmbedding, typename ElementTy = ElementType>
	CanonicalGraph<EmbeddingTy,ElementTy>* turn_canonical_graph(QuickPattern<EmbeddingTy,ElementTy> & qp, bool is_directed = false) {
	//CPattern* turn_canonical_graph(QPattern & qp, bool is_directed = false) {
		assert(!is_directed);
		//bliss::AbstractGraph* ag = readGraph(qp);
	//}
	//bliss::AbstractGraph* readGraph(QPattern & qp) {
		bliss::AbstractGraph* ag = 0;
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
		//construct bliss graph
		const unsigned number_vertices = vertices.size();
		ag = new bliss::Graph(vertices.size());
		//set vertices
		for(unsigned i = 0; i < number_vertices; ++i)
			ag->change_color(i, (unsigned)vertices[i + 1]);
		//read edges
		assert(qp.get_size() > 1);
		for(unsigned index = 1; index < qp.get_size(); ++index) {
			auto element = qp.at(index);
			//std::cout << "element: " << element << "\n";
			VertexId from, to;
			from = qp.at(element.get_his()).get_vid();
			to = element.get_vid();
			//std::cout << "add edge (" << from << "," << to << ")\n";
			ag->add_edge(from - 1, to - 1, std::make_pair((unsigned)element.get_his(), index));
		}
		//return ag;
		bliss::Stats stats;
		const unsigned * cl = ag->canonical_form(stats, &report_aut, stdout); // canonical labeling. This is expensive.
		bliss::AbstractGraph* cf = ag->permute(cl); //permute to canonical form
		delete ag;
		CanonicalGraph<EmbeddingTy,ElementTy>* cg = new CanonicalGraph<EmbeddingTy,ElementTy>(cf);
		//CPattern* cg = new CPattern(cf);
		delete cf;
		return cg;
	}
*/
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
	inline bool edge_existed(const EdgeEmbedding & emb, BYTE history, VertexId src, VertexId dst) {
		std::pair<VertexId, VertexId> added_edge(src, dst);
		for(unsigned i = 1; i < emb.size(); ++i) {
			if(emb.get_vertex(i) == dst && emb.get_vertex(emb.get_history(i)) == src)
				return true;
		}
		return false;
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

#endif // EDGE_MINER_HPP_
