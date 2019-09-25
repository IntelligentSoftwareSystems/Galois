#ifndef EDGELIST_H_
#define EDGELIST_H_

#include "types.h"
typedef unsigned IndexTy;
typedef unsigned VertexId;
typedef unsigned char BYTE;
typedef unsigned long Ulong;
typedef galois::gstl::Vector<BYTE> ByteList;
typedef galois::gstl::Vector<unsigned> UintList;
typedef galois::gstl::Vector<Ulong> UlongList;
typedef galois::gstl::Vector<VertexId> VertexList;
typedef galois::gstl::Vector<UintList> IndexLists;
typedef galois::gstl::Vector<ByteList> ByteLists;
typedef galois::gstl::Vector<VertexList> VertexLists;
typedef galois::gstl::Set<VertexId> VertexSet;
typedef galois::substrate::PerThreadStorage<UintList> Lists;

struct Edge {
	IndexTy src;
	IndexTy dst;
	Edge() : src(0), dst(0) {}
	Edge(IndexTy from, IndexTy to) : src(from), dst(to) {}
	std::string to_string() const {
		std::stringstream ss;
		ss << "e(" << src << "," << dst << ")";
		return ss.str();
	}
};

class EdgeList {
using iterator = typename galois::gstl::Vector<Edge>::iterator;
public:
	EdgeList() {}
	EdgeList(Graph& graph, bool is_dag = false) {
		init(graph, is_dag);
	}
	~EdgeList() {}
	bool empty() const { return edgelist.empty(); }
	iterator begin() { return edgelist.begin(); }
	iterator end() { return edgelist.end(); }
	size_t size() const { return edgelist.size(); }
	void resize (size_t n) { edgelist.resize(n); }
	void init(Graph& graph, bool is_dag = false) {
		size_t num_edges = graph.sizeEdges();
		if (!is_dag) num_edges = num_edges / 2;
		edgelist.resize(num_edges);
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
	//std::vector<Edge> edgelist;
	galois::gstl::Vector<Edge> edgelist;
	void add_edge(unsigned pos, IndexTy src, IndexTy dst) {
		edgelist[pos] = Edge(src, dst);
	}
};
#endif
