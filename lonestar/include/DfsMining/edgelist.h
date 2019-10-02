#ifndef EDGELIST_H_
#define EDGELIST_H_
#include "types.h"

class EdgeList {
//using iterator = typename galois::gstl::Vector<Edge>::iterator;
using iterator = typename std::vector<Edge>::iterator;
public:
	EdgeList() {}
	EdgeList(Graph& graph, bool is_dag = false) {
		init(graph, is_dag);
	}
	~EdgeList() {}
	bool empty() const { return edges.empty(); }
	iterator begin() { return edges.begin(); }
	iterator end() { return edges.end(); }
	size_t size() const { return edges.size(); }
	void resize (size_t n) { edges.resize(n); }
	void push_back(Edge e) { edges.push_back(e); }
	Edge& get_edge(unsigned i) { return edges[i]; }
	Edge* get_edge_ptr(unsigned i) { return &(edges[i]); }
	void init(Graph& graph, bool is_dag = false) {
		size_t num_edges = graph.sizeEdges();
		if (!is_dag) num_edges = num_edges / 2;
		edges.resize(num_edges);
		if(is_dag) {
			galois::do_all(galois::iterate(graph.begin(), graph.end()),
				[&](const GNode& src) {
					for (auto e : graph.edges(src)) {
						GNode dst = graph.getEdgeDst(e);
						add_edge(*e, src, dst);
					}
				},
				galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::no_conflicts(),
				galois::loopname("Init-edgelist")
			);
		} else {
			size_t num_vertices = graph.size();
			UintList num_init_emb(num_vertices);
			galois::do_all(galois::iterate(graph.begin(), graph.end()),
				[&](const GNode& src) {
					num_init_emb[src] = 0;
					for (auto e : graph.edges(src)) {
						GNode dst = graph.getEdgeDst(e);
						if (src < dst) num_init_emb[src] ++;
					}
				},
				galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::no_conflicts(),
				galois::loopname("Init-edgelist-alloc")
			);
			UintList indices(num_vertices + 1);
			unsigned total = 0;
			for (size_t n = 0; n < num_vertices; n++) {
				indices[n] = total;
				total += num_init_emb[n];
			}
			indices[num_vertices] = total;
			galois::do_all(galois::iterate(graph.begin(), graph.end()),
				[&](const GNode& src) {
					IndexTy start = indices[src];
					for (auto e : graph.edges(src)) {
						GNode dst = graph.getEdgeDst(e);
						if (src < dst) {
							add_edge(start, src, dst);
							start ++;
						}
					}
				},
				galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::no_conflicts(),
				galois::loopname("Init-edgelist-insert")
			);
		}
	}
private:
	//galois::gstl::Vector<Edge> edges;
	std::vector<Edge> edges;
	void add_edge(unsigned pos, IndexTy src, IndexTy dst) {
		edges[pos] = Edge(src, dst);
	}
};
#endif
