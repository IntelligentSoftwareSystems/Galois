#ifndef EDGELIST_H_
#define EDGELIST_H_
#include "types.h"
#include "core.h"

class EdgeList {
//using iterator = typename galois::gstl::Vector<Edge>::iterator;
using iterator = typename std::vector<Edge>::iterator;
public:
	EdgeList() {}
	EdgeList(Graph& graph, bool directed = false) {
		init(graph, directed);
	}
	~EdgeList() {}
	bool empty() const { return edges.empty(); }
	iterator begin() { return edges.begin(); }
	iterator end() { return edges.end(); }
	size_t size() const { return edges.size(); }
	size_t get_num_vertices() const { return num_vertices; }
	void resize (size_t n) { edges.resize(n); }
	void push_back(Edge e) { edges.push_back(e); }
	Edge& get_edge(size_t i) { return edges[i]; }
	Edge* get_edge_ptr(size_t i) { return &(edges[i]); }
	void init(Graph& graph, bool directed = false, bool symmetrize = false) {
		num_vertices = graph.size();
		num_edges = graph.sizeEdges();
		if (!directed && !symmetrize) num_edges = num_edges / 2;
		edges.resize(num_edges);
		if (directed || symmetrize) {
			galois::do_all(galois::iterate(graph.begin(), graph.end()), [&](const GNode& src) {
				for (auto e : graph.edges(src)) {
					auto dst = graph.getEdgeDst(e);
					add_edge(*e, src, dst);
				}
			}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("Init-edgelist"));
		} else {
			UintList num_init_emb(num_vertices);
			galois::do_all(galois::iterate(graph.begin(), graph.end()), [&](const GNode& src) {
				num_init_emb[src] = 0;
				for (auto e : graph.edges(src)) {
					auto dst = graph.getEdgeDst(e);
					if (src < dst) num_init_emb[src] ++;
				}
			}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("Init-edgelist-alloc"));
			UintList indices(num_vertices + 1);
			unsigned total = 0;
			for (size_t n = 0; n < num_vertices; n++) {
				indices[n] = total;
				total += num_init_emb[n];
			}
			indices[num_vertices] = total;
			galois::do_all(galois::iterate(graph.begin(), graph.end()), [&](const auto& src) {
				auto start = indices[src];
				for (auto e : graph.edges(src)) {
					auto dst = graph.getEdgeDst(e);
					if (src < dst) {
						add_edge(start, src, dst);
						start ++;
					}
				}
			}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("Init-edgelist-insert"));
		}
	}
	//computing degeneracy ordering and core value
	void ord_core() {
		rank.resize(num_vertices);
		std::vector<IndexT> d0(num_vertices, 0);
		std::vector<IndexT> cd0(num_vertices+1);
		std::vector<IndexT> adj0(2*num_edges);
		for (size_t i = 0; i < num_edges; i ++) {
			d0[edges[i].src]++;
			d0[edges[i].dst]++;
		}
		cd0[0] = 0;
		for (size_t i = 1; i < num_vertices+1; i ++) {
			cd0[i] = cd0[i-1] + d0[i-1];
			d0[i-1] = 0;
		}
		for (size_t i = 0; i < num_edges; i ++) {
			adj0[cd0[edges[i].src] + d0[edges[i].src]++] = edges[i].dst;
			adj0[cd0[edges[i].dst] + d0[edges[i].dst]++] = edges[i].src;
		}
		bheap heap;
		heap.mkheap(num_vertices, d0);
		size_t r = 0;
		for (size_t i = 0; i < num_vertices; i ++) {
			keyvalue kv = heap.popmin();
			rank[kv.key] = num_vertices - (++r);
			for (IndexT j = cd0[kv.key]; j < cd0[kv.key + 1]; j ++) {
				heap.update(adj0[j]);
			}
		}
	}
	void RelabelEdges() {
		std::cout << "Relabeling edges\n";
		ord_core();
		for (size_t i = 0; i < num_edges; i ++) {
			int source = rank[edges[i].src];
			int target = rank[edges[i].dst];
			if (source < target) {
				int tmp = source;
				source = target;
				target = tmp;
			}
			edges[i].src = source;
			edges[i].dst = target;
		}
		//std::vector<ValueT> new_labels(num_vertices);
		//for (int i = 0; i < num_vertices; i ++) new_labels[rank[i]] = labels[i];
		//for (int i = 0; i < num_vertices; i ++) labels[i] = new_labels[i];
	}
	unsigned get_core() {
		std::vector<IndexT> d(num_vertices, 0);
		for (size_t i = 0; i < num_edges; i ++) {
			d[edges[i].src]++;
		}
		//std::vector<IndexT> cd(num_vertices+1);
		//cd[0] = 0;
		unsigned max = 0;
		for (size_t i = 1; i < num_vertices+1; i ++) {
			//cd[i] = cd[i-1] + d[i-1];
			max = (max > d[i-1]) ? max : d[i-1];
			d[i-1] = 0;
		}
		printf("core value (max truncated degree) = %u\n",max);
		return max;
	}
private:
	//galois::gstl::Vector<Edge> edges;
	size_t num_vertices;
	size_t num_edges;
	std::vector<Edge> edges;
	void add_edge(size_t pos, VertexId src, VertexId dst) {
		edges[pos] = Edge(src, dst);
	}
	std::vector<int> rank;
};
#endif
