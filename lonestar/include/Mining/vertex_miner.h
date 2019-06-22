#ifndef VERTEX_MINER_H
#define VERTEX_MINER_H
#include "miner.h"
typedef std::unordered_map<BaseEmbedding, Frequency> SimpleMap;
typedef QuickPattern<EdgeInducedEmbedding<StructuralElement>, StructuralElement> StrQPattern; // structural quick pattern
typedef CanonicalGraph<EdgeInducedEmbedding<StructuralElement>, StructuralElement> StrCPattern; // structural canonical pattern
typedef std::unordered_map<StrQPattern, Frequency> StrQpMapFreq; // mapping structural quick pattern to its frequency
typedef std::unordered_map<StrCPattern, Frequency> StrCgMapFreq; // mapping structural canonical pattern to its frequency
typedef galois::substrate::PerThreadStorage<StrQpMapFreq> LocalStrQpMapFreq;
typedef galois::substrate::PerThreadStorage<StrCgMapFreq> LocalStrCgMapFreq;

class VertexMiner : public Miner {
public:
	VertexMiner(Graph *g) {
		graph = g;
		num_cliques = 0;
		degree_counting();
	}
	virtual ~VertexMiner() {}
	// Given an embedding, extend it with one more vertex. Used for vertex-induced k-motif
	inline void extend_vertex(const VertexEmbedding &emb, VertexEmbeddingQueue &queue) {
		unsigned n = emb.size();
		for (unsigned i = 0; i < n; ++i) {
			VertexId src = emb.get_vertex(i); // toExpand (expand every vertex in the embedding)
			for (auto e : graph->edges(src)) {
				GNode dst = graph->getEdgeDst(e);
				if (!is_vertexInduced_automorphism(emb, i, src, dst)) { // toAdd (only add non-automorphisms)
					VertexEmbedding new_emb(emb);
					if (n == 2 && k == 4) new_emb.set_pid(find_motif_pattern_id(n, i, dst, emb));
					new_emb.push_back(dst);
					queue.push_back(new_emb);
				}
			}
		}
	}
	// Given an embedding, extend it with one more vertex. Used for k-cliques (same as RStream: slow)
	inline void extend_vertex(const BaseEmbedding &emb, BaseEmbeddingQueue &queue) {
		unsigned n = emb.size();
		for(unsigned i = 0; i < n; ++i) {
			VertexId src = emb.get_vertex(i); // toExpand (expand every vertex in the embedding: slow)
			for(auto e : graph->edges(src)) {
				GNode dst = graph->getEdgeDst(e);
				// extend vertex in ascending order to avoid unnecessary enumeration
				if(dst > emb.get_vertex(n-1)) { // toAdd (only add vertx with larger ID)
					BaseEmbedding new_emb(emb);
					new_emb.push_back(dst);
					queue.push_back(new_emb);
				}
			}
		}
	}
	// Given an embedding, extend it with one more vertex. Used for k-cliques. (fast)
	inline void extend_vertex(const BaseEmbedding &emb, BaseEmbeddingQueue &queue, UintAccu &num, bool need_update = true) {
		unsigned n = emb.size();
		VertexId src = emb.get_vertex(n-1); // toExpand (only expand the last vertex in the embedding: fast)
		for (auto e : graph->edges(src)) {
			GNode dst = graph->getEdgeDst(e);
			if (dst > src && is_all_connected(dst, emb, n-1)) { // toAdd (only add vertex that is connected to all the vertices in the embedding)
				if (need_update) { // generate a new embedding and add it to the next queue
					BaseEmbedding new_emb(emb);
					new_emb.push_back(dst);
					queue.push_back(new_emb);
				} else num += 1; // if size = k, no need to add to the queue, just accumulate
			}
		}
	}
	/*
	void extend_vertex_all(UintAccu &num, bool need_update = true) {
		galois::do_all(galois::iterate(graph->begin(), graph->end()),
			[&](const GNode& src) {
				for (auto e : graph->edges(src)) {
					GNode dst = graph->getEdgeDst(e);
					if(src < dst) {
						BaseEmbedding emb(2);
						emb.set_element(0, src);
						emb.set_element(1, dst);
						for (auto e1 : graph->edges(dst)) {
							GNode dst_dst = graph->getEdgeDst(e1);
							//if (dst_dst > dst && is_connected(dst_dst, src))
							if (dst_dst > dst && is_all_connected(dst_dst, emb, 1))
								num += 1;
						}
					}
				}
			},
			galois::chunk_size<CHUNK_SIZE>(), galois::steal(), 
			galois::loopname("CliqueCouting")
		);
	}
	//*/
	inline void extend_vertex_each(unsigned level, unsigned pos, const EmbeddingList& emb_list, IndexList& num_emb, UintAccu &num, bool need_update = true) {
		VertexId vid = emb_list.get_vid(level, pos);
		IndexTy idx = emb_list.get_idx(level, pos);
		num_emb[pos] = 0;
		BaseEmbedding emb(level+1);
		emb.set_element(level, vid);
		for (unsigned l = 1; l <= level; l ++) {
			emb.set_element(level-l, emb_list.get_vid(level-l, idx));
			idx = emb_list.get_idx(level-l, idx);
		}
		for (auto e : graph->edges(vid)) {
			GNode dst = graph->getEdgeDst(e);
			if(vid < dst && is_all_connected(dst, emb, level)) {
				if (need_update) num_emb[pos] ++;
				else num += 1;
			}
		}
	}
	inline void extend_vertex_each(unsigned level, unsigned pos, EmbeddingList& emb_list, const IndexList& indices) {
		VertexId vid = emb_list.get_vid(level, pos);
		IndexTy idx = emb_list.get_idx(level, pos);
		unsigned start = indices[pos];
		BaseEmbedding emb;
		emb.push_back(vid);
		// backward constructing the embedding
		for (unsigned i = 1; i <= level; i ++) {
			emb.insert(emb.begin(), emb_list.get_vid(level-i, idx));
			idx = emb_list.get_idx(level-i, idx);
		}
		for (auto e : graph->edges(vid)) {
			GNode dst = graph->getEdgeDst(e);
			// check if it is a clique
			if (vid < dst && is_all_connected(dst, emb, level)) {
				emb_list.set_idx(level+1, start, pos);
				emb_list.set_vid(level+1, start++, dst);
			}
		}
	}
	inline void aggregate_all(BaseEmbeddingQueue &in_queue, BaseEmbeddingQueue &out_queue) {
		SimpleMap simple_agg;
		for (const BaseEmbedding emb : in_queue) {
			auto it = simple_agg.find(emb);
			if (it != simple_agg.end()) {
				if (it->second == it->first.size() - 2) {
					out_queue.push_back(emb);
					simple_agg.erase(it);
				}
				else simple_agg[emb] += 1;
			}
			else simple_agg[emb] = 1;
		}
	}
	// check each embedding to find the cliques (this method is used by RStream, slow!)
	inline void aggregate_each(const BaseEmbedding &emb, SimpleMap& sm, BaseEmbeddingQueue &out_queue) {
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
	inline void aggregate_each(const VertexEmbedding &emb, std::vector<UintAccu> &accumulators) {
		unsigned n = emb.size();
		for (unsigned i = 0; i < n; ++i) {
			VertexId src = emb.get_vertex(i);
			for (auto e : graph->edges(src)) {
				GNode dst = graph->getEdgeDst(e);
				if (!is_vertexInduced_automorphism(emb, i, src, dst)) {
					assert(n < 4);
					unsigned pid = find_motif_pattern_id(n, i, dst, emb);
					accumulators[pid] += 1;
				}
			}
		}
	}
	inline void aggregate_each(const VertexEmbedding &emb, UintMap &p_map) {
		unsigned n = emb.size();
		assert(n >= 4);
		for (unsigned i = 0; i < n; ++i) {
			VertexId src = emb.get_vertex(i);
			for (auto e : graph->edges(src)) {
				GNode dst = graph->getEdgeDst(e);
				if (!is_vertexInduced_automorphism(emb, i, src, dst)) {
					unsigned pid = find_motif_pattern_id(n, i, dst, emb);
					if (p_map.find(pid) != p_map.end()) p_map[pid] += 1;
					else p_map[pid] = 1;
				}
			}
		}
	}
	inline void quick_aggregate_each(const VertexEmbedding& emb, StrQpMapFreq& qp_map) {
		unsigned n = emb.size();
		for (unsigned i = 0; i < n; ++i) {
			VertexId src = emb.get_vertex(i);
			for (auto e : graph->edges(src)) {
				GNode dst = graph->getEdgeDst(e);
				if (!is_vertexInduced_automorphism(emb, i, src, dst)) {
					std::vector<bool> connected;
					get_connectivity(n, i, dst, emb, connected);
					StrQPattern qp(n+1, connected);
					if (qp_map.find(qp) != qp_map.end()) {
						qp_map[qp] += 1;
						qp.clean();
					} else {
						qp_map[qp] = 1;
					}
				}
			}
		}
	}
	inline void canonical_aggregate_each(const StrQPattern &qp, Frequency freq, StrCgMapFreq &cg_map) {
		StrCPattern cg(qp);
		//qp.clean();
		if (cg_map.find(cg) != cg_map.end()) cg_map[cg] += freq;
		else cg_map[cg] = freq;
		cg.clean();
	}
	inline void merge_qp_map(LocalStrQpMapFreq &qp_localmap, StrQpMapFreq &qp_map) {
		for (unsigned i = 0; i < qp_localmap.size(); i++) {
			StrQpMapFreq qp_lmap = *qp_localmap.getLocal(i);
			for (auto element : qp_lmap) {
				if (qp_map.find(element.first) != qp_map.end())
					qp_map[element.first] += element.second;
				else
					qp_map[element.first] = element.second;
			}
		}
	}
	inline void merge_cg_map(LocalStrCgMapFreq &cg_localmap, StrCgMapFreq &cg_map) {
		for (unsigned i = 0; i < cg_localmap.size(); i++) {
			StrCgMapFreq cg_lmap = *cg_localmap.getLocal(i);
			for (auto element : cg_lmap) {
				if (cg_map.find(element.first) != cg_map.end())
					cg_map[element.first] += element.second;
				else
					cg_map[element.first] = element.second;
			}
		}
	}
	inline unsigned get_total_num_cliques() { return num_cliques; }
	void printout_motifs(std::vector<UintAccu> &accumulators) {
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
			std::cout << "\ttoo many patterns to show\n";
		}
		std::cout << std::endl;
	}
	void printout_motifs(UintMap &p_map) {
		assert(p_map.size() == 21);
		std::cout << std::endl;
		for (auto it = p_map.begin(); it != p_map.end(); ++it)
			std::cout << "{" << it->first << "} --> " << it->second << std::endl;
		std::cout << std::endl;
	}
	void printout_motifs(StrCgMapFreq &cg_map) {
		std::cout << std::endl;
		for (auto it = cg_map.begin(); it != cg_map.end(); ++it)
			std::cout << it->first << " --> " << it->second << std::endl;
		std::cout << std::endl;
	}

private:
	unsigned num_cliques;
	galois::substrate::SimpleLock slock;
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
	inline unsigned find_motif_pattern_id(unsigned n, unsigned idx, VertexId dst, const VertexEmbedding& emb) {
		unsigned pid = 0;
		if (n == 2) { // count 3-motifs
			pid = 1; // 3-chain
			if(idx == 0 && is_connected(emb.get_vertex(1), dst)) pid = 0; // triangle
		} else if (n == 3) { // count 4-motifs
			unsigned num_edges = 1;
			pid = emb.get_pid();
			if (pid == 0) { // expanding a triangle
				for (unsigned j = 0; j < n; j ++)
					if (j != idx && is_connected(emb.get_vertex(j), dst)) num_edges ++;
				pid = num_edges + 2; // p3: tailed-triangle; p4: diamond; p5: 4-clique
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
					} else {
						pid = 0; // p0: 3-path
					}
				} else if (num_edges == 2) {
					unsigned center = is_connected(emb.get_vertex(1), emb.get_vertex(2)) ? 1 : 0;
					if (connected[center]) {
						pid = 3; // p3: tailed-triangle
					} else {
						pid = 2; // p2: 4-cycle
					}
				} else {
					pid = 4; // p4: diamond
				}
			}
		} else { // count 5-motif and beyond
			std::vector<bool> connected;
			get_connectivity(n, idx, dst, emb, connected);
			Matrix A(n+1, std::vector<MatType>(n+1, 0));
			gen_adj_matrix(n+1, connected, A);
			std::vector<MatType> c(n+1, 0);
			char_polynomial(n+1, A, c);
			bliss::UintSeqHash h;
			for (unsigned i = 0; i < n+1; ++i)
				h.update((unsigned)c[i]);
			pid = h.get_value();
		}
		return pid;
	}
};

#endif // VERTEX_MINER_HPP_
