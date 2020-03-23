#include "DfsMining/edgelist.h"
#include "scan.h"
#include "core.h"

void EdgeList::init(Graph& graph, bool directed, bool symmetrize) {
	num_vertices = graph.size();
	num_edges = graph.sizeEdges();
	if (!directed && !symmetrize) num_edges = num_edges / 2;
	this->resize(num_edges);
	if (directed || symmetrize) {
		//galois::do_all(galois::iterate(graph.begin(), graph.end()), [&](const GNode& src) {
		for (GNode src : graph) {
			for (auto e : graph.edges(src)) {
				auto dst = graph.getEdgeDst(e);
				this->add_edge(*e, src, dst);
			}
		}//, galois::chunk_size<256>(), galois::steal(), galois::loopname("Init-edgelist"));
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
					this->add_edge(start, src, dst);
					start ++;
				}
			}
		}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("Init-edgelist-insert"));
	}
	std::cout << "Done initialize edgelist, size: " << size() << "\n";
}

unsigned EdgeList::get_core() {
	std::vector<IndexT> d(num_vertices, 0);
	for (size_t i = 0; i < num_edges; i ++) {
		//d[all_edges[i].src]++;
		d[(*this)[i].src]++;
	}
	unsigned max = 0;
	for (size_t i = 1; i < num_vertices+1; i ++) {
		max = (max > d[i-1]) ? max : d[i-1];
		d[i-1] = 0;
	}
	printf("core value (max truncated degree) = %u\n",max);
	return max;
}

//computing degeneracy ordering and core value
void EdgeList::ord_core() {
	std::cout << "Computing vertex rank using core value\n";
	rank.resize(num_vertices);
	std::vector<IndexT> d0(num_vertices, 0);
	std::vector<IndexT> cd0(num_vertices+1);
	std::vector<IndexT> adj0(2*num_edges);
	for (size_t i = 0; i < num_edges; i ++) {
		d0[(*this)[i].src]++;
		d0[(*this)[i].dst]++;
	}
	cd0[0] = 0;
	for (size_t i = 1; i < num_vertices+1; i ++) {
		cd0[i] = cd0[i-1] + d0[i-1];
		d0[i-1] = 0;
	}
	for (size_t i = 0; i < num_edges; i ++) {
		adj0[cd0[(*this)[i].src] + d0[(*this)[i].src]++] = (*this)[i].dst;
		adj0[cd0[(*this)[i].dst] + d0[(*this)[i].dst]++] = (*this)[i].src;
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

void EdgeList::relabel() {
	std::cout << "Relabeling edges\n";
	ord_core();
	for (size_t i = 0; i < num_edges; i ++) {
		int source = rank[(*this)[i].src];
		int target = rank[(*this)[i].dst];
		if (source < target) {
			int tmp = source;
			source = target;
			target = tmp;
		}
		(*this)[i].src = source;
		(*this)[i].dst = target;
	}
}

unsigned EdgeList::generate_graph(Graph &g) {
	relabel();
	std::vector<IndexT> degrees(num_vertices, 0);
	for (size_t i = 0; i < num_edges; i ++) {
		degrees[(*this)[i].src]++;
	}
	std::vector<IndexT> offsets(num_vertices+1);
	offsets[0] = 0;
	unsigned max = 0;
	for (size_t i = 1; i < num_vertices+1; i++) {
		offsets[i] = offsets[i-1] + degrees[i-1];
		max = (max > degrees[i-1]) ? max : degrees[i-1];
		//degrees[i-1] = 0;
	}
	printf("core value (max truncated degree) = %u\n", max);
	std::vector<std::vector<VertexId> > vertices(num_vertices);
	for (size_t i = 0; i < num_edges; i++) {
		vertices[(*this)[i].src].push_back((*this)[i].dst);
	}
	g.allocateFrom(num_vertices, num_edges);
	g.constructNodes();
	for (size_t v = 0; v < num_vertices; v++) {
		auto row_begin = offsets[v];
		auto row_end = offsets[v+1];
		g.fixEndEdge(v, row_end);
		for (auto offset = row_begin; offset < row_end; offset ++) {
			g.constructEdge(offset, vertices[v][offset-row_begin], 0);
		}
	}
	return max;
}

