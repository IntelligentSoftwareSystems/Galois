#include "DfsMining/edgelist.h"

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

/*
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
//*/

