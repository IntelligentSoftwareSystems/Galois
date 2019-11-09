#ifndef EMBEDDING_LIST_H
#define EMBEDDING_LIST_H
#include "egonet.h"
#include "embedding.h"

class EmbeddingList {
public:
	EmbeddingList() {}
	~EmbeddingList() {}

	void allocate(Graph *graph, unsigned max_size, unsigned max_degree) {
		//std::cout << "allocating memory for embedding list, max_size = " << max_size << ", max_degree = " << max_degree << "\n";
		global_graph = graph;
		max_level = max_size;
		length = max_degree;

		vid_lists.resize(max_level);
		vid_lists[0].resize(1);
		vid_lists[1].resize(1);
		#ifdef USE_MAP
		#ifndef USE_FORMULA
		for (unsigned i = 2; i < max_level; i ++) vid_lists[i].resize(i*length);
		src_indices.resize(max_level);
		for (unsigned i = 2; i < max_level; i ++) src_indices[i].resize(i*length);
		#endif
		#else
		for (unsigned i = 2; i < max_level; i ++) vid_lists[i].resize(length);
		#endif

		#ifdef USE_PID
		pid_lists.resize(max_level);
		#ifdef USE_FORMULA
		for (unsigned i = 2; i < max_level; i ++) pid_lists[i].resize(length);
		#else
		for (unsigned i = 2; i < max_level; i ++) pid_lists[i].resize(i*length);
		#endif
		#endif

		sizes.resize(max_level);
		for (unsigned i = 0; i < max_level; i ++) sizes[i] = 0;
		sizes[1] = 1;

		#ifdef SHRINK
		//std::cout << "[debug] use shrinking graph\n";
		labels.resize(length);
		shrink_graph.allocate(length, max_level);
		#else
		labels.resize(graph->size());
		std::fill(labels.begin(), labels.end(), 0);
		#endif
	}

	void init_vertex(const VertexId vid) {
		cur_level = 0;
		sizes[0] = 1;
		//idx_lists[0].resize(1);
		vid_lists[0][0] = vid;
		//idx_lists[0][0] = 0;
		vid_lists[1].resize(length);
		history.push_back(vid);
		#ifdef SHRINK
		if (ids.empty()) {
			ids.resize(global_graph->size());
			old_ids.resize(length);
			std::fill(ids.begin(), ids.end(), (unsigned)-1);
		}
		#endif

		size_t index = 0;
		for (auto e : global_graph->edges(vid)) {
			auto dst = global_graph->getEdgeDst(e);
			labels[dst] = 1; // mark the neighbors of edge.src
			set_vid(1, index, dst);
			#ifdef SHRINK
			ids[dst] = index;
			old_ids[index] = dst;
			set_vid(1, index, index);
			shrink_graph.set_degree(1, index, 0);//new degrees
			#endif
			index ++;
		}
		sizes[1] = index;
		#ifdef SHRINK
		construct_local_graph_from_vertex(vid);
		#endif
	}

	void construct_local_graph_from_vertex(const VertexId vid) {
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

	void init_edge(const Edge &edge) {
		//std::cout << "Insert edge: " << edge.to_string() << "\n";
		cur_level = 1;
		wed_count = 0, tri_count = 0;
		clique4_count = 0, cycle4_count = 0;
		history.clear();
		history.push_back(edge.src);
		history.push_back(edge.dst);
		#ifdef SHRINK
		if (ids.empty()) {
			ids.resize(global_graph->size());
			old_ids.resize(length);
			std::fill(ids.begin(), ids.end(), (unsigned)-1);
		}
		for (auto e : global_graph->edges(edge.dst)) {
			auto dst = global_graph->getEdgeDst(e);
			//std::cout << "\tlabeling dst = " << dst << ", src = " << edge.dst << "\n";
			ids[dst] = (unsigned)-2;
		}
		size_t index = 0;
		#else
		vid_lists[1][0] = edge.dst;
		vid_lists[0][0] = edge.src;
		set_size(1, 1);
		#endif
		#ifdef USE_MAP
		for (auto e : global_graph->edges(edge.dst)) {
			auto dst = global_graph->getEdgeDst(e);
			labels[dst] = 2;
		}
		#endif
		for (auto e : global_graph->edges(edge.src)) {
			auto dst = global_graph->getEdgeDst(e);
			#ifdef SHRINK
			if (ids[dst] == (unsigned)-2) { // intersection
				//std::cout << "\tdst = " << dst << ", new_id = " << index << "\n";
				ids[dst] = index;
				old_ids[index] = dst;
				labels[index] = 2;
				set_vid(2, index, index);
				shrink_graph.set_degree(2, index, 0);//new degrees
				index ++;
			}
			#else
			if (dst == edge.dst) continue;
			#ifdef USE_MAP
			#ifndef MOTIF_ADHOC
			if (labels[dst] == 2) labels[dst] = 3;
			else
			#endif
			#endif
			labels[dst] = 1; // mark the neighbors of edge.src
			#endif
		}
		#ifdef SHRINK
		set_size(2, index); // number of neighbors of src. 
		construct_local_graph_from_edge(edge);
		#endif

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
	void construct_local_graph_from_edge(const Edge &edge) {
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

	void init_egonet_degree(unsigned level, VertexId dst) {
		shrink_graph.set_degree(level, dst, 0);
	}

	void update_labels(unsigned level, VertexId src) {
		for (auto e : global_graph->edges(src)) {
			auto dst = global_graph->getEdgeDst(e);
			labels[dst] += 1 << level;
		}
	}

	void clear_labels(const VertexId src) {
		for (auto e : global_graph->edges(src)) {
			auto dst = global_graph->getEdgeDst(e);
			labels[dst] = 0; // mark the neighbors of edge.src
		}
	}

	void reset_labels(unsigned level)	{
		for (size_t emb_id = 0; emb_id < size(level+1); emb_id ++) {//restoring labels
			auto src = get_vertex(level+1, emb_id);
			labels[src] = level;
		}
	}

	void resume_labels(unsigned level, const VertexId src) {
		for (auto e : global_graph->edges(src)) {
			auto dst = global_graph->getEdgeDst(e);
			labels[dst] -= 1 << level;
		}
	}

	void reset_ids(const VertexId vid) {
		for (auto e : global_graph->edges(vid)) {
			auto dst = global_graph->getEdgeDst(e);
			ids[dst] = (unsigned)-1;
		}
	}

	void mark_neighbors() {
		VertexId src = get_vid(0, 0);
		VertexId dst = get_vid(1, 0);
		for (auto e : global_graph->edges(src)) {
			auto w = global_graph->getEdgeDst(e);
			if (dst == w) continue;
			labels[w] = 1;
		}
	}

	void triangles_and_wedges() {
		VertexId src = get_vid(0, 0);
		VertexId dst = get_vid(1, 0);
		for (auto e : global_graph->edges(dst)) {
			auto w = global_graph->getEdgeDst(e);
			if (w == src) continue;
			if (labels[w] == 1) {
				labels[w] = 3;
				T_vu[tri_count] = w;
				tri_count++;
			}
			else {
				W_u[wed_count] = w;
				wed_count++;
				labels[w] = 2;
			}
		}
	}

	void cycle() {
		for (Ulong j = 0; j < wed_count; j++) {
			auto src = W_u[j];
			for (auto e : global_graph->edges(src)) {
				auto dst = global_graph->getEdgeDst(e);
				if (labels[dst] == 1) cycle4_count ++;
			}
			W_u[j] = 0;
		}
	}
	void clique() {
		for (Ulong tr_i = 0; tr_i < tri_count; tr_i++) {
			auto src = T_vu[tr_i];
			for (auto e : global_graph->edges(src)) {
				auto dst = global_graph->getEdgeDst(e);
				if (labels[dst] == 3) clique4_count ++;
			}
			labels[src] = 0;
			T_vu[tr_i] = 0;
		}
	}

	template <typename EmbeddingTy = VertexEmbedding>
	inline void get_embedding(unsigned level, EmbeddingTy &emb) {
		//std::cout << ", get_embedding: level = " << level << ", pos = " << pos;
		for (unsigned l = 0; l < level+1; l ++) {
			ElementType ele(history[l]);
			emb.set_element(l, ele);
		}
	}

	template <typename EmbeddingTy = VertexEmbedding>
	inline void get_embedding(unsigned level, unsigned pos, EmbeddingTy &emb) {
		//std::cout << ", get_embedding: level = " << level << ", pos = " << pos;
		for (unsigned l = 0; l < level+1; l ++) {
			ElementType ele(history[l]);
			emb.set_element(level, ele);
		}
	}

	void update_egonet(unsigned level) {
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

	bool is_connected (unsigned level, VertexId vid, unsigned element_id) const {
		if (level == 1) return (get_label(vid) == element_id + 1) || (get_label(vid) == 3);
		else if (level == 2) {
			return (get_label(vid) & (1 << element_id));
		} else return false;
	}

	size_t size() const { return sizes[cur_level]; }
	size_t size(unsigned level) const { return sizes[level]; }
	VertexId get_vertex(unsigned level, size_t i) const { return vid_lists[level][i]; }
	VertexId get_vid(unsigned level, size_t i) const { return vid_lists[level][i]; }
	VertexId get_history(unsigned level) const { return history[level]; }
	IndexTy get_idx(unsigned level, IndexTy id) const { return idx_lists[level][id]; }
	BYTE get_pid(unsigned level, size_t i) const { return pid_lists[level][i]; }
	BYTE get_src(unsigned level, size_t i) const { return src_indices[level][i]; }
	//BYTE get_label(VertexId vid) const { return labels[vid]; }
	unsigned get_label(VertexId vid) const { return labels[vid]; }
	unsigned get_level() const { return cur_level; }
	void set_size(unsigned level, size_t size) { sizes[level] = size; }
	void set_vid(unsigned level, size_t id, VertexId vid) { vid_lists[level][id] = vid; }
	void set_idx(unsigned level, size_t id, IndexTy idx) { idx_lists[level][id] = idx; }
	void set_pid(unsigned level, size_t id, BYTE pid) { pid_lists[level][id] = pid; }
	void set_src(unsigned level, size_t id, BYTE src) { src_indices[level][id] = src; }
	//void set_label(VertexId vid, BYTE value) { labels[vid] = value; }
	void set_label(VertexId vid, unsigned value) { labels[vid] = value; }
	void set_level(unsigned level) { cur_level = level; }
	VertexId edge_begin(unsigned level, VertexId vid) { return shrink_graph.edge_begin(vid); }
	VertexId edge_end(unsigned level, VertexId vid) { return shrink_graph.edge_begin(vid) + shrink_graph.get_degree(level, vid); }
	VertexId getEdgeDst(VertexId vid) const { return shrink_graph.getEdgeDst(vid); }
	Ulong get_tri_count() { return tri_count; }
	Ulong get_wed_count() { return wed_count; }
	Ulong get_cycle4_count() { return cycle4_count; }
	Ulong get_clique4_count() { return clique4_count; }
	void push_history(VertexId vid) { history.push_back(vid); }
	void pop_history() { history.pop_back(); }
	void inc_tri_count() { tri_count ++; }
	void inc_wed_count() { wed_count ++; }
	void inc_cycle4_count() { cycle4_count ++; }
	void inc_clique4_count() { clique4_count ++; }
	
	Egonet shrink_graph;

protected:
	std::vector<VertexId> history; 
	ByteLists src_indices;
	VertexLists vid_lists;
	IndexLists idx_lists;
	ByteLists pid_lists; //pid[i] is the pattern id of each embedding
	UintList sizes; //sizes[level]: no. of embeddings (i.e. no. of vertices in the the current level)
	//ByteList labels; //labels[i] is the label of each vertex i; it is the perfect hash table for checking in O(1) time if edge (triangle, etc) exists
	UintList labels;
	unsigned max_level;
	unsigned cur_level;
	unsigned length;
	UintList ids;
	UintList old_ids;
	Graph *global_graph; // original input graph
	Graph *local_graph; // shrinking graph 
	UintList T_vu; // T_vu is an array containing all the third vertex of each triangle
	UintList W_u;
	Ulong tri_count; // number of triangles incident to this edge
	Ulong wed_count; // number of wedges incident to this edge
	Ulong clique4_count;
	Ulong cycle4_count;
};
typedef galois::substrate::PerThreadStorage<EmbeddingList> EmbeddingLists;

#endif
