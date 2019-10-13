#define USE_DAG
#define USE_DFS
#define ALGO_EDGE
#define USE_EGONET
#define USE_SIMPLE
#define USE_BASE_TYPES
#define CHUNK_SIZE 256
#include "pangolin.h"

// This is a implementation of the WWW'18 paper:
// Danisch et al., Listing k-cliques in Sparse Real-World Graphs, WWW 2018
const char* name = "Kcl";
const char* desc = "Counts the K-Cliques in a graph using DFS traversal (inputs do NOT need to be symmetrized)";
const char* url  = 0;

class AppMiner : public VertexMiner {
public:
	AppMiner(Graph *g, unsigned size, int np, bool use_dag, unsigned c) : VertexMiner(g, size, np, use_dag, c) {}
	~AppMiner() {}
	// toExtend (only extend the last vertex in the embedding: fast)
	bool toExtend(unsigned n, const BaseEmbedding &emb, VertexId src, unsigned pos) {
		return pos == n-1;
	}
	void print_output() {
		std::cout << "\n\ttotal_num_cliques = " << get_total_count() << "\n";
	}
	#ifndef USE_EGONET
	// only add vertex that is connected to all the vertices in the embedding
	bool toAdd(unsigned n, const BaseEmbedding &emb, VertexId dst, unsigned element_id) {
		#ifdef USE_DAG
		return is_all_connected_dag(dst, emb, n-1);
		#else
		VertexId src = emb.get_vertex(element_id);
		return (src < dst) && is_all_connected(dst, emb, n-1);
		#endif
	}
	#else
	bool toAdd(unsigned level, VertexId dst, const Egonet &egonet) {
		return egonet.get_label(dst) == level;
	}
	void init_egonet_from_edge(const Edge &edge, Egonet &egonet, EmbeddingList &emb_list) {
		UintList *ids = id_lists.getLocal(); // hold the local vertex ID (new ID)
		if (ids->empty()) {
			ids->resize(graph->size());
			std::fill(ids->begin(), ids->end(), (unsigned)-1);
		}
		for (auto e : graph->edges(edge.dst)) {
			auto dst = graph->getEdgeDst(e);
			(*ids)[dst] = (unsigned)-2; // mark the neighbors of edge.dst
		}
		unsigned level = 1;
		unsigned new_id = 0;
		for (auto e : graph->edges(edge.src)) {
			auto dst = graph->getEdgeDst(e);
			// intersection of two neighbor lists: 
			// if dst (a neighbor of edge.src) is also connected to edge.dst
			if ((*ids)[dst] == (unsigned)-2) {
				if (max_size == 3) total_num += 1;
				else {
					(*ids)[dst] = new_id;
					emb_list.set_vertex(level, new_id, dst);
					emb_list.set_vertex(level+1, new_id, new_id);
					//emb_list.set_label(new_id, level+1);
					egonet.set_label(new_id, level+1); // this vertex survives for the next level
					egonet.set_degree(level+1, new_id, 0);//new degrees
				}
				new_id ++;
			}
		}
		if (max_size > 3) {
			size_t new_size = (size_t)new_id;
			emb_list.set_size(level+1, new_size); // number of neighbors of u. Since u is in level k, u's neighbors are in level k-1
			//egonet.set_size(level, new_size);
			for (unsigned i = 0; i < emb_list.size(level+1); i ++) {
				auto src = emb_list.get_vertex(level, i); // get the global vertex ID
				for (auto e : graph->edges(src)) {
					auto dst = graph->getEdgeDst(e); // dst is the neighbor's neighbor
					unsigned local_vid = (*ids)[dst]; // get the local vertex ID
					if (local_vid < (unsigned)-2) {
						auto degree = egonet.get_degree(level+1, i);
						egonet.set_adj(core * i + degree, local_vid);
						egonet.set_degree(level+1, i, degree+1);
					}
				}
			}
		}
		for (auto e : graph->edges(edge.dst)) {
			auto dst = graph->getEdgeDst(e);
			(*ids)[dst] = (unsigned)-1;
		}
	}
	#endif
};

#include "DfsMining/engine.h"
