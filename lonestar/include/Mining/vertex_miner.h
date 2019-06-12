#ifndef VERTEX_MINER_H
#define VERTEX_MINER_H
#include "miner.h"
typedef std::unordered_map<BaseEmbedding, Frequency> SimpleMap;

class VertexMiner : public Miner {
public:
	VertexMiner(Graph *g) {
		graph = g;
		num_cliques = 0;
		degree_counting();
	}
	virtual ~VertexMiner() {}
	// Given an embedding, extend it with one more vertex. Used for vertex-induced k-motif
	void extend_vertex_motif(const VertexEmbedding &emb, VertexEmbeddingQueue &queue) {
		unsigned n = emb.size();
		for (unsigned i = 0; i < n; ++i) {
			VertexId src = emb.get_vertex(i);
			for (auto e : graph->edges(src)) {
				GNode dst = graph->getEdgeDst(e);
				if (!is_vertexInduced_automorphism(emb, i, src, dst)) {
					VertexEmbedding new_emb(emb);
					if (n == 2 && k == 4) new_emb.set_pid(find_motif_pattern_id(n, i, dst, emb));
					new_emb.push_back(dst);
					queue.push_back(new_emb);
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
	void aggregate_motif_each(const VertexEmbedding &emb, std::vector<UintAccu> &accumulators, UintMap &pattern_map, unsigned &num_patterns) {
		unsigned n = emb.size();
		for (unsigned i = 0; i < n; ++i) {
			VertexId src = emb.get_vertex(i);
			for (auto e : graph->edges(src)) {
				GNode dst = graph->getEdgeDst(e);
				if (!is_vertexInduced_automorphism(emb, i, src, dst)) {
					unsigned pid = find_motif_pattern_id(n, i, dst, emb);
					if (n > 3) {
						slock.lock();
						if (pattern_map.find(pid) == pattern_map.end()) {
							pattern_map[pid] = num_patterns++;
						}
						slock.unlock();
						pid = pattern_map[pid];
					}
					assert(pid < 21);
					accumulators[pid] += 1;
				}
			}
		}
	}
	unsigned get_total_num_cliques() { return num_cliques; }
	void printout_motifs(std::vector<UintAccu> &accumulators) {
		std::cout << std::endl;
		if (accumulators.size() == 2) {
			std::cout << "\ttriangles\t" << accumulators[0].reduce() << std::endl;
			std::cout << "\t3-chains\t" << accumulators[1].reduce() << std::endl;
		} else if (accumulators.size() == 6) {
			std::cout << "\t4-paths\t" << accumulators[0].reduce() << std::endl;
			std::cout << "\t3-stars\t" << accumulators[1].reduce() << std::endl;
			std::cout << "\t4-cycles\t" << accumulators[2].reduce() << std::endl;
			std::cout << "\ttailed-triangles\t" << accumulators[3].reduce() << std::endl;
			std::cout << "\tdiamonds\t" << accumulators[4].reduce() << std::endl;
			std::cout << "\t4-cliques\t" << accumulators[5].reduce() << std::endl;
		} else if (accumulators.size() == 21) {
			for (unsigned i = 0; i < 21; i ++)
				std::cout << "\t" << i << "\t" << accumulators[i].reduce() << std::endl;
		} else {
			std::cout << "\ttoo many patterns to show\n";
		}
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
		} else { // count 5-motif and beyond
			std::vector<bool> connected;
			get_connectivity(n, idx, dst, emb, connected);
			StrQPattern qp(n, dst, emb, connected);
			//StrCPattern* cp = turn_canonical_graph<EdgeInducedEmbedding<StructuralElement>, StructuralElement>(qp);
			//pid = cp->get_hash();
			/*
			Matrix A(n+1, std::vector<MatType>(n+1, 0));
			gen_adj_matrix(n+1, connected, A);
			std::vector<MatType> c(n+1, 0);
			char_polynomial(n+1, A, c);
			bliss::UintSeqHash h;
			for (unsigned i = 0; i < n+1; ++i)
				h.update((unsigned)c[i]);
			pid = h.get_value();
			*/
		}
		return pid;
	}
};

#endif // VERTEX_MINER_HPP_
