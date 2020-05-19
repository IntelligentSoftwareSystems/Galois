#include "pangolin/DfsMining/embedding_list_dfs.h"

template <bool is_single, bool use_ccode, bool use_pcode, bool use_local_graph, bool do_local_counting, bool is_clique>
void EmbeddingList<is_single,use_ccode,use_pcode,use_local_graph,do_local_counting,is_clique>::
allocate(Graph *graph, unsigned max_size, unsigned max_degree, int num_patterns) {
	global_graph = graph;
	max_level = max_size;
	length = max_degree;
	npatterns = num_patterns;
	//std::cout << "max_level=" << max_level << ", length=" << length << "\n";

	vid_lists.resize(max_level);
	vid_lists[0].resize(1);
	vid_lists[1].resize(1);
	if (is_clique || do_local_counting) {
		//std::cout << "allocating vertex list\n";
		for (unsigned i = 2; i < max_level-1; i ++)
			vid_lists[i].resize(length);
	} else {
		for (unsigned i = 2; i < max_level-1; i ++) 
			vid_lists[i].resize(i*length);
		src_indices.resize(max_level);
		for (unsigned i = 2; i < max_level-1; i ++) 
			src_indices[i].resize(i*length);
	}

	if (use_pcode) { // TODO: maybe useful for subgraph listing
		pid_lists.resize(max_level);
		if (do_local_counting) {
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

	if (use_local_graph) {
		//std::cout << "allocating local_graph, length=" << length << ", max_level=" << max_level << "\n";
		labels.resize(length);
		local_graph.allocate(length, max_level);
	} else {
		labels.resize(graph->size());
		std::fill(labels.begin(), labels.end(), 0);
	}
	local_counters.resize(num_patterns); // for local counts of tri, 4-cycle and 4-clique
	allocated = true;
}

template <bool is_single, bool use_ccode, bool use_pcode, bool use_local_graph, bool do_local_counting, bool is_clique>
void EmbeddingList<is_single,use_ccode,use_pcode,use_local_graph,do_local_counting,is_clique>::
init_vertex(const VertexId vid) {
	//std::cout << "Insert vertex: " <<  vid << "\n";
	cur_level = 0;
	sizes[0] = 1;
	vid_lists[0][0] = vid;
	vid_lists[1].resize(length);
	history.clear();
	history.push_back(vid);
	if (use_local_graph) {
		if (ids.empty()) {
			ids.resize(global_graph->size());
			old_ids.resize(length);
			std::fill(ids.begin(), ids.end(), (unsigned)-1);
		}
	}
	size_t index = 0;
	for (auto e : global_graph->edges(vid)) {
		auto dst = global_graph->getEdgeDst(e);
		if (use_local_graph) {
			ids[dst] = index;
			old_ids[index] = dst;
			set_vid(1, index, index); // use local vertex ID
			labels[index] = 1; // mark the neighbors of edge.src
			local_graph.set_degree(1, index, 0);//new degrees
		} else {
			if (!is_clique && dst <= vid) continue;
			set_vid(1, index, dst); // use global vertex ID
			labels[dst] = 1; // mark the neighbors of edge.src
		}
		index ++;
	}
	sizes[1] = index;
	if (use_local_graph) construct_local_graph_from_vertex(vid);
}

template <bool is_single, bool use_ccode, bool use_pcode, bool use_local_graph, bool do_local_counting, bool is_clique>
void EmbeddingList<is_single,use_ccode,use_pcode,use_local_graph,do_local_counting,is_clique>::
construct_local_graph_from_vertex(const VertexId vid) {
	for (size_t i = 0; i < size(1); i ++) {
		auto src = old_ids[i];
		// intersection of two neighbor lists
		for (auto e : global_graph->edges(src)) {
			auto dst = global_graph->getEdgeDst(e); // dst is the neighbor's neighbor
			auto new_id = ids[dst];
			if (new_id != (unsigned)-1) { // if dst is also a neighbor of u
				auto degree = local_graph.get_degree(1, i);
				local_graph.set_adj(length * i + degree, new_id); // relabel
				local_graph.set_degree(1, i, degree+1);
			}
		}
	}
	reset_ids(vid);
}

template <bool is_single, bool use_ccode, bool use_pcode, bool use_local_graph, bool do_local_counting, bool is_clique>
void EmbeddingList<is_single,use_ccode,use_pcode,use_local_graph,do_local_counting,is_clique>::
init_edge(const SEdge &edge) {
	//std::cout << "Insert edge: " << edge.to_string() << "\n";
	cur_level = 1;
	history.clear();
	history.push_back(edge.src);
	history.push_back(edge.dst);
	vid_lists[1][0] = edge.dst;
	vid_lists[0][0] = edge.src;
	set_size(1, 1);

	if (do_local_counting) {
		//wed_count = 0, tri_count = 0;
		//clique4_count = 0, cycle4_count = 0;
		for (int i = 0; i < npatterns; i++) local_counters[i] = 0;
	}

	if (use_local_graph) {
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
				local_graph.set_degree(2, index, 0);//new degrees
				index ++;
			}
		}
		//std::cout << "number of vertices in the list: " << index << "\n";
		set_size(2, index); // number of neighbors of src. 
		if (max_level > 3) construct_local_graph_from_edge(edge);
	} else { // use global graph
		if (use_ccode) {
			if (!is_single || !is_clique) {
				//std::cout << "initializing ccode for multi-pattern solver\n";
				for (auto e : global_graph->edges(edge.dst)) {
					auto dst = global_graph->getEdgeDst(e);
					if (dst == edge.src) continue;
					labels[dst] = 2;
				}
			}
			for (auto e : global_graph->edges(edge.src)) {
				auto dst = global_graph->getEdgeDst(e);
				if (dst == edge.dst) continue;
				if (!is_single || !is_clique) {
					if (labels[dst] == 2) labels[dst] = 3;
					else labels[dst] = 1; // mark the neighbors of edge.src
				} else labels[dst] = 1; // mark the neighbors of edge.src
			}
		}
	} // end if use_local_graph 
}

// construct the subgraph induced by vertex src's neighbors
template <bool is_single, bool use_ccode, bool use_pcode, bool use_local_graph, bool do_local_counting, bool is_clique>
void EmbeddingList<is_single,use_ccode,use_pcode,use_local_graph,do_local_counting,is_clique>::
construct_local_graph_from_edge(const SEdge &edge) {
	for (size_t emb_id = 0; emb_id < size(2); emb_id ++) {
		auto x = old_ids[emb_id];
		for (auto e : global_graph->edges(x)) {
			auto dst = global_graph->getEdgeDst(e); // dst is the neighbor's neighbor
			auto new_id = ids[dst];
			if (new_id < (unsigned)-2) { // if dst is also a neighbor of u
				auto degree = local_graph.get_degree(2, emb_id);
				local_graph.set_adj(length * emb_id + degree, new_id); // relabel
				local_graph.set_degree(2, emb_id, degree+1);
			}
		}
	}
	reset_ids(edge.dst);
}

template <bool is_single, bool use_ccode, bool use_pcode, bool use_local_graph, bool do_local_counting, bool is_clique>
void EmbeddingList<is_single,use_ccode,use_pcode,use_local_graph,do_local_counting,is_clique>::
update_egonet(unsigned level) {
	for (size_t new_emb_id = 0; new_emb_id < size(level+1); new_emb_id ++) {
		auto src = get_vertex(level+1, new_emb_id);
		auto begin = local_graph.edge_begin(src);
		auto end = begin + local_graph.get_degree(level, src);
		for (auto e = begin; e < end; e ++) {
			auto dst = local_graph.getEdgeDst(e);
			if (labels[dst] == level+1) {
				local_graph.inc_degree(level+1, src);
			} else {
				// swap with the last neighbor
				local_graph.set_adj(e--, local_graph.getEdgeDst(--end));
				local_graph.set_adj(end, dst);
			}
		}
	}
	local_graph.set_cur_level(level+1);
}

template class EmbeddingList<true,  true, false, false, false, true>;  // K-cliques
template class EmbeddingList<true,  true, false, true,  false, true>;  // K-cliques using local graph
template class EmbeddingList<true,  true, true,  false, false, false>; // Subgraph Listing
template class EmbeddingList<false, true, true,  false, false, false>; // Motif
template class EmbeddingList<false, true, true,  false, true,  false>; // Motif using formula
//template class EmbeddingList<SimpleElement, VertexEmbedding, false>; // Motif

