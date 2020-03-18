#include "BfsMining/embedding_list.h"

template <typename ElementType,typename EmbeddingType>
void EmbeddingList<ElementType,EmbeddingType>::init(Graph& graph, unsigned max_size, bool is_dag) {
	last_level = 1;
	max_level = max_size;
	vid_lists.resize(max_level);
	idx_lists.resize(max_level);
	size_t num_emb = graph.sizeEdges();
	if (!is_dag) num_emb = num_emb / 2;
	vid_lists[1].resize(num_emb);
	idx_lists[1].resize(num_emb);
	if (std::is_same<ElementType,LabeledElement>::value) {
		his_lists.resize(max_level);
		his_lists[1].resize(num_emb);
		galois::do_all(galois::iterate((size_t)0, num_emb), [&](const size_t& pos) {
			his_lists[1][pos] = 0;
		}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("Init-his"));
	}
	if (is_dag) {
		galois::do_all(galois::iterate(graph.begin(), graph.end()), [&](const GNode& src) {
			for (auto e : graph.edges(src)) {
				auto dst = graph.getEdgeDst(e);
				vid_lists[1][*e] = dst;
				idx_lists[1][*e] = src;
			}
		}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("Init-vid"));
	} else {
		size_t num_vertices = graph.size();
		UintList num_init_emb(num_vertices);
		galois::do_all(galois::iterate(graph.begin(), graph.end()), [&](const GNode& src) {
			num_init_emb[src] = 0;
			for (auto e : graph.edges(src)) {
				auto dst = graph.getEdgeDst(e);
				if (src < dst) num_init_emb[src] ++;
			}
		}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("Init-vid-alloc"));
		UintList indices(num_vertices + 1);
		unsigned total = 0;
		for (size_t n = 0; n < num_vertices; n++) {
			indices[n] = total;
			total += num_init_emb[n];
		}
		indices[num_vertices] = total;
		galois::do_all(galois::iterate(graph.begin(), graph.end()), [&](const GNode& src) {
			auto start = indices[src];
			for (auto e : graph.edges(src)) {
				GNode dst = graph.getEdgeDst(e);
				if (src < dst) { // TODO: this may be incorrect for FSM: may cause the 4-FSM bug
					vid_lists[1][start] = dst;
					idx_lists[1][start] = src;
					start ++;
				}
			}
		}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("Init-vid-insert"));
	}
}

template class EmbeddingList<SimpleElement, BaseEmbedding>; // TC and KCL
template class EmbeddingList<SimpleElement, VertexEmbedding>; // Motif
template class EmbeddingList<LabeledElement, EdgeInducedEmbedding<LabeledElement>>; // FSM
