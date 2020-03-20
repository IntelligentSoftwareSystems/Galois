#include "DfsMining/embedding_list_dfs.h"

template <typename ElementType, typename EmbeddingType,
	bool is_single, bool use_ccode, bool shrink, bool use_formula>
void EmbeddingList<ElementType,EmbeddingType,is_single,use_ccode,shrink,use_formula>::
allocate(Graph *graph, unsigned max_size, unsigned max_degree) {
	global_graph = graph;
	max_level = max_size;
	length = max_degree;

	vid_lists.resize(max_level);
	vid_lists[0].resize(1);
	vid_lists[1].resize(1);
	if (is_single) {
		for (unsigned i = 2; i < max_level; i ++)
			vid_lists[i].resize(length);
	} else {
		if (!use_formula) {
			for (unsigned i = 2; i < max_level; i ++) 
				vid_lists[i].resize(i*length);
			src_indices.resize(max_level);
			for (unsigned i = 2; i < max_level; i ++) 
				src_indices[i].resize(i*length);
		}
	}

	if (use_ccode) {
		pid_lists.resize(max_level);
		if (use_formula) {
			for (unsigned i = 2; i < max_level; i ++)
				pid_lists[i].resize(length);
		} else {
			for (unsigned i = 2; i < max_level; i ++)
				pid_lists[i].resize(i*length);
		}
	}

	sizes.resize(max_level);
	for (unsigned i = 0; i < max_level; i ++) sizes[i] = 0;
	sizes[1] = 1;

	if (shrink) {
		labels.resize(length);
		shrink_graph.allocate(length, max_level);
	} else {
		labels.resize(graph->size());
		std::fill(labels.begin(), labels.end(), 0);
	}
}

template <typename ElementType,typename EmbeddingType,
	bool is_single, bool use_ccode, bool shrink, bool use_formula>
void EmbeddingList<ElementType,EmbeddingType,is_single,use_ccode,shrink,use_formula>::
init_vertex(const VertexId vid) {
	cur_level = 0;
	sizes[0] = 1;
	vid_lists[0][0] = vid;
	vid_lists[1].resize(length);
	history.push_back(vid);
	if (shrink) {
		if (ids.empty()) {
			ids.resize(global_graph->size());
			old_ids.resize(length);
			std::fill(ids.begin(), ids.end(), (unsigned)-1);
		}
	}
	size_t index = 0;
	for (auto e : global_graph->edges(vid)) {
		auto dst = global_graph->getEdgeDst(e);
		labels[dst] = 1; // mark the neighbors of edge.src
		set_vid(1, index, dst);
		if (shrink) {
			ids[dst] = index;
			old_ids[index] = dst;
			set_vid(1, index, index);
			shrink_graph.set_degree(1, index, 0);//new degrees
		}
		index ++;
	}
	sizes[1] = index;
	if (shrink) construct_local_graph_from_vertex(vid);
}

template <typename ElementType,typename EmbeddingType,
	bool is_single, bool use_ccode, bool shrink, bool use_formula>
void EmbeddingList<ElementType,EmbeddingType,is_single,use_ccode,shrink,use_formula>::
construct_local_graph_from_vertex(const VertexId vid) {
	for (size_t i = 0; i < size(1); i ++) {
		auto src = old_ids[i];
		// intersection of two neighbor lists
		for (auto e : global_graph->edges(src)) {
			auto dst = global_graph->getEdgeDst(e); // dst is the neighbor's neighbor
			auto new_id = ids[dst];
			if (new_id != (unsigned)-1) { // if dst is also a neighbor of u
				auto degree = shrink_graph.get_degree(1, i);
				shrink_graph.set_adj(length * i + degree, new_id); // relabel
				shrink_graph.set_degree(1, i, degree+1);
			}
		}
	}
	reset_ids(vid);
}

template <typename ElementType,typename EmbeddingType,
	bool is_single, bool use_ccode, bool shrink, bool use_formula>
void EmbeddingList<ElementType,EmbeddingType,is_single,use_ccode,shrink,use_formula>::
init_edge(const SEdge &edge) {
	//std::cout << "Insert edge: " << edge.to_string() << "\n";
	cur_level = 1;
	wed_count = 0, tri_count = 0;
	clique4_count = 0, cycle4_count = 0;
	history.clear();
	history.push_back(edge.src);
	history.push_back(edge.dst);
	vid_lists[1][0] = edge.dst;
	vid_lists[0][0] = edge.src;
	set_size(1, 1);

	if (shrink) {
		if (ids.empty()) {
			ids.resize(global_graph->size());
			old_ids.resize(length);
			std::fill(ids.begin(), ids.end(), (unsigned)-1);
		}
		for (auto e : global_graph->edges(edge.dst)) {
			auto dst = global_graph->getEdgeDst(e);
			ids[dst] = (unsigned)-2;
		}
		size_t index = 0;
		for (auto e : global_graph->edges(edge.src)) {
			auto dst = global_graph->getEdgeDst(e);
			if (ids[dst] == (unsigned)-2) { // intersection
				ids[dst] = index;
				old_ids[index] = dst;
				labels[index] = 2;
				set_vid(2, index, index);
				shrink_graph.set_degree(2, index, 0);//new degrees
				index ++;
			}
		}
		set_size(2, index); // number of neighbors of src. 
		if (max_level > 3) construct_local_graph_from_edge(edge);
	} else { // non-shrink
		if (use_ccode) {
			if (!is_single) {
				//std::cout << "initializing ccode for multi-pattern solver\n";
				for (auto e : global_graph->edges(edge.dst)) {
					auto dst = global_graph->getEdgeDst(e);
					labels[dst] = 2;
				}
			}
			for (auto e : global_graph->edges(edge.src)) {
				auto dst = global_graph->getEdgeDst(e);
				if (dst == edge.dst) continue;
				if (!is_single) {
					#ifndef MOTIF_ADHOC
					if (labels[dst] == 2) labels[dst] = 3;
					else labels[dst] = 1; // mark the neighbors of edge.src
					#endif
				} else labels[dst] = 1; // mark the neighbors of edge.src
			}
		}
	} // end shrink
	
	#ifdef MOTIF_ADHOC
	if (W_u.empty()) {
		T_vu.resize(length+1); // hold the vertices that form a triangle with u and v
		W_u.resize(length+1); // hold the vertices that form a wedge with u and v
		std::fill(T_vu.begin(), T_vu.end(), 0);
		std::fill(W_u.begin(), W_u.end(), 0);
	}
	#endif
}

// construct the subgraph induced by vertex src's neighbors
template <typename ElementType,typename EmbeddingType,
	bool is_single, bool use_ccode, bool shrink, bool use_formula>
void EmbeddingList<ElementType,EmbeddingType,is_single,use_ccode,shrink,use_formula>::
construct_local_graph_from_edge(const SEdge &edge) {
	for (size_t emb_id = 0; emb_id < size(2); emb_id ++) {
		auto x = old_ids[emb_id];
		for (auto e : global_graph->edges(x)) {
			auto dst = global_graph->getEdgeDst(e); // dst is the neighbor's neighbor
			auto new_id = ids[dst];
			if (new_id < (unsigned)-2) { // if dst is also a neighbor of u
				auto degree = shrink_graph.get_degree(2, emb_id);
				shrink_graph.set_adj(length * emb_id + degree, new_id); // relabel
				shrink_graph.set_degree(2, emb_id, degree+1);
			}
		}
	}
	reset_ids(edge.dst);
}

template <typename ElementType,typename EmbeddingType,
	bool is_single, bool use_ccode, bool shrink, bool use_formula>
void EmbeddingList<ElementType,EmbeddingType,is_single,use_ccode,shrink,use_formula>::
update_egonet(unsigned level) {
	for (size_t new_emb_id = 0; new_emb_id < size(level+1); new_emb_id ++) {
		auto src = get_vertex(level+1, new_emb_id);
		auto begin = shrink_graph.edge_begin(src);
		auto end = begin + shrink_graph.get_degree(level, src);
		for (auto e = begin; e < end; e ++) {
			auto dst = shrink_graph.getEdgeDst(e);
			if (labels[dst] == level+1)
				shrink_graph.inc_degree(level+1, src);
			else {
				shrink_graph.set_adj(e--, shrink_graph.getEdgeDst(--end));
				shrink_graph.set_adj(end, dst);
			}
		}
	}
	shrink_graph.set_cur_level(level+1);
}

template class EmbeddingList<SimpleElement, BaseEmbedding>; // KCL
template class EmbeddingList<SimpleElement, BaseEmbedding, true, true, true, false>; // KCL shrink
template class EmbeddingList<SimpleElement, BaseEmbedding, false>; // Motif
template class EmbeddingList<SimpleElement, VertexEmbedding, false>; // Motif

