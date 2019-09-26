#ifndef MINER_HPP_
#define MINER_HPP_
#include "embedding.h"

template <typename T>
inline galois::gstl::Vector<T> parallel_prefix_sum(const galois::gstl::Vector<T> &degrees) {
	galois::gstl::Vector<T> sums(degrees.size() + 1);
	T total = 0;
	for (size_t n = 0; n < degrees.size(); n++) {
		sums[n] = total;
		total += degrees[n];
	}
	sums[degrees.size()] = total;
	return sums;
}

class Miner {
public:
	Miner() {}
	virtual ~Miner() {}
	// insert single-edge embeddings into the embedding queue (worklist)
	inline void init(EmbeddingQueueType &queue) {
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
						EmbeddingType new_emb;
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
	inline unsigned intersect(unsigned a, unsigned b) {
		return intersect_merge(a, b);
	}
	inline unsigned intersect_dag(unsigned a, unsigned b) {
		return intersect_dag_merge(a, b);
	}

protected:
	Graph *graph;
	unsigned max_size;
	std::vector<unsigned> degrees;

	void degree_counting() {
		degrees.resize(graph->size());
		galois::do_all(galois::iterate(graph->begin(), graph->end()),
			[&] (GNode v) {
				degrees[v] = std::distance(graph->edge_begin(v), graph->edge_end(v));
			},
			galois::loopname("DegreeCounting")
		);
	}
	inline unsigned intersect_merge(unsigned src, unsigned dst) {
		unsigned count = 0;
		for (auto e : graph->edges(dst)) {
			GNode dst_dst = graph->getEdgeDst(e);
			for (auto e1 : graph->edges(src)) {
				GNode to = graph->getEdgeDst(e1);
				if (dst_dst == to) {
					count += 1;
					break;
				}
				if (to > dst_dst) break;
			}
		}
		return count;
	}
	inline unsigned intersect_dag_merge(unsigned p, unsigned q) {
		unsigned count = 0;
		auto p_start = graph->edge_begin(p);
		auto p_end = graph->edge_end(p);
		auto q_start = graph->edge_begin(q);
		auto q_end = graph->edge_end(q);
		auto p_it = p_start;
		auto q_it = q_start;
		int a;
		int b;
		while (p_it < p_end && q_it < q_end) {
			a = graph->getEdgeDst(p_it);
			b = graph->getEdgeDst(q_it);
			int d = a - b;
			if (d <= 0) p_it ++;
			if (d >= 0) q_it ++;
			if (d == 0) count ++;
		}
		return count;
	}
	inline unsigned intersect_search(unsigned a, unsigned b) {
		if (degrees[a] == 0 || degrees[b] == 0) return 0;
		unsigned count = 0;
		unsigned lookup = a;
		unsigned search = b;
		if (degrees[a] > degrees[b]) {
			lookup = b;
			search = a;
		} 
		Graph::edge_iterator begin = graph->edge_begin(search, galois::MethodFlag::UNPROTECTED);
		Graph::edge_iterator end = graph->edge_end(search, galois::MethodFlag::UNPROTECTED);
		for (auto e : graph->edges(lookup)) {
			GNode key = graph->getEdgeDst(e);
			if(binary_search(key, begin, end)) count ++;
		}
		return count;
	}
	inline bool is_all_connected_except(unsigned dst, unsigned pos, const BaseEmbedding &emb) {
		unsigned n = emb.size();
		bool all_connected = true;
		for(unsigned i = 0; i < n; ++i) {
			if (i == pos) continue;
			unsigned from = emb.get_vertex(i);
			if (!is_connected(from, dst)) {
				all_connected = false;
				break;
			}
		}
		return all_connected;
	}
	inline bool is_all_connected(unsigned dst, const BaseEmbedding &emb, unsigned end, unsigned start = 0) {
		assert(start >= 0 && end > 0);
		bool all_connected = true;
		for(unsigned i = start; i < end; ++i) {
			unsigned from = emb.get_vertex(i);
			if (!is_connected(from, dst)) {
				all_connected = false;
				break;
			}
		}
		return all_connected;
	}
	inline bool is_all_connected_dag(unsigned dst, const BaseEmbedding &emb, unsigned end, unsigned start = 0) {
		assert(start >= 0 && end > 0);
		bool all_connected = true;
		for(unsigned i = start; i < end; ++i) {
			unsigned from = emb.get_vertex(i);
			if (!is_connected_dag(dst, from)) {
				all_connected = false;
				break;
			}
		}
		return all_connected;
	}
	// check if vertex a is connected to vertex b in a undirected graph
	inline bool is_connected(unsigned a, unsigned b) {
		if (degrees[a] == 0 || degrees[b] == 0) return false;
		unsigned key = a;
		unsigned search = b;
		if (degrees[a] < degrees[b]) {
			key = b;
			search = a;
		} 
		auto begin = graph->edge_begin(search, galois::MethodFlag::UNPROTECTED);
		auto end = graph->edge_end(search, galois::MethodFlag::UNPROTECTED);
		//return serial_search(key, begin, end);
		return binary_search(key, begin, end);
	}
	inline int is_connected_dag(unsigned key, unsigned search) {
		if (degrees[search] == 0) return false;
		auto begin = graph->edge_begin(search, galois::MethodFlag::UNPROTECTED);
		auto end = graph->edge_end(search, galois::MethodFlag::UNPROTECTED);
		return binary_search(key, begin, end);
	}
	inline bool serial_search(unsigned key, Graph::edge_iterator begin, Graph::edge_iterator end) {
		for (auto offset = begin; offset != end; ++ offset) {
			unsigned d = graph->getEdgeDst(offset);
			if (d == key) return true;
			if (d > key) return false;
		}
		return false;
	}
	inline bool binary_search(unsigned key, Graph::edge_iterator begin, Graph::edge_iterator end) {
		Graph::edge_iterator l = begin;
		Graph::edge_iterator r = end-1;
		while (r >= l) { 
			Graph::edge_iterator mid = l + (r - l) / 2; 
			unsigned value = graph->getEdgeDst(mid);
			if (value == key) return true;
			if (value < key) l = mid + 1; 
			else r = mid - 1; 
		} 
		return false;
	}
	inline int binary_search(unsigned key, Graph::edge_iterator begin, int length) {
		if (length < 1) return -1;
		int l = 0;
		int r = length-1;
		while (r >= l) { 
			int mid = l + (r - l) / 2; 
			unsigned value = graph->getEdgeDst(begin+mid);
			if (value == key) return mid;
			if (value < key) l = mid + 1; 
			else r = mid - 1; 
		} 
		return -1;
	}
	inline void gen_adj_matrix(unsigned n, const std::vector<bool> &connected, Matrix &a) {
		unsigned l = 0;
		for (unsigned i = 1; i < n; i++)
			for (unsigned j = 0; j < i; j++)
				if (connected[l++]) a[i][j] = a[j][i] = 1;
	}
	// calculate the trace of a given n*n matrix
	inline MatType trace(unsigned n, Matrix matrix) {
		MatType tr = 0;
		for (unsigned i = 0; i < n; i++) {
			tr += matrix[i][i];
		}
		return tr;
	}
	// matrix mutiplication, both a and b are n*n matrices
	inline Matrix product(unsigned n, const Matrix &a, const Matrix &b) {
		Matrix c(n, std::vector<MatType>(n));
		for (unsigned i = 0; i < n; ++i) { 
			for (unsigned j = 0; j < n; ++j) { 
				c[i][j] = 0; 
				for(unsigned k = 0; k < n; ++k) {
					c[i][j] += a[i][k] * b[k][j];
				}
			} 
		} 
		return c; 
	}
	// calculate the characteristic polynomial of a n*n matrix A
	inline void char_polynomial(unsigned n, Matrix &A, std::vector<MatType> &c) {
		// n is the size (num_vertices) of a graph
		// A is the adjacency matrix (n*n) of the graph
		Matrix C;
		C = A;
		for (unsigned i = 1; i <= n; i++) {
			if (i > 1) {
				for (unsigned j = 0; j < n; j ++)
					C[j][j] += c[n-i+1];
				C = product(n, A, C);
			}
			c[n-i] -= trace(n, C) / i;
		}
	}
	inline void get_connectivity(unsigned n, unsigned idx, VertexId dst, const VertexEmbedding &emb, std::vector<bool> &connected) {
		connected.push_back(true); // 0 and 1 are connected
		for (unsigned i = 2; i < n; i ++)
			for (unsigned j = 0; j < i; j++)
				if (is_connected(emb.get_vertex(i), emb.get_vertex(j)))
					connected.push_back(true);
				else connected.push_back(false);
		for (unsigned j = 0; j < n; j ++) {
			if (j == idx) connected.push_back(true);
			else if (is_connected(emb.get_vertex(j), dst))
				connected.push_back(true);
			else connected.push_back(false);
		}
	}
	// eigenvalue based approach to find the pattern id for a given embedding
	inline unsigned find_motif_pattern_id_eigen(unsigned n, unsigned idx, VertexId dst, const VertexEmbedding& emb) {
		std::vector<bool> connected;
		get_connectivity(n, idx, dst, emb, connected);
		Matrix A(n+1, std::vector<MatType>(n+1, 0));
		gen_adj_matrix(n+1, connected, A);
		std::vector<MatType> c(n+1, 0);
		char_polynomial(n+1, A, c);
		bliss::UintSeqHash h;
		for (unsigned i = 0; i < n+1; ++i)
			h.update((unsigned)c[i]);
		return h.get_value();
	}
};

#endif // MINER_HPP_
