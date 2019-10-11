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
	#ifdef USE_EGONET
	bool toAdd(unsigned level, VertexId dst, const Egonet &egonet) {
		return egonet.get_label(dst) == level;
	}
	#else
	// only add vertex that is connected to all the vertices in the embedding
	bool toAdd(unsigned n, const BaseEmbedding &emb, VertexId dst, unsigned element_id) {
		#ifdef USE_DAG
		return is_all_connected_dag(dst, emb, n-1);
		#else
		VertexId src = emb.get_vertex(element_id);
		return (src < dst) && is_all_connected(dst, emb, n-1);
		#endif
	}
	#endif
	void print_output() {
		std::cout << "\n\ttotal_num_cliques = " << get_total_count() << "\n";
	}
};

#include "DfsMining/engine.h"
