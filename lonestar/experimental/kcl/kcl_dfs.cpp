#define USE_DAG
#define USE_DFS
//#define NO_LABEL
#define ALGO_EDGE
#define LARGE_SIZE // for large graphs such as soc-Livejournal1 and com-Orkut
#define USE_SIMPLE
#define USE_BASE_TYPES
#define CHUNK_SIZE 256
#include "pangolin.h"

// This is a implementation of the WWW'18 paper:
// Danisch et al., Listing k-cliques in Sparse Real-World Graphs, WWW 2018
const char* name = "Kcl";
const char* desc = "Counts the K-Cliques in a graph using DFS traversal (using DAG)";
const char* url  = 0;

class AppMiner : public VertexMiner {
public:
	AppMiner(Graph *g) : VertexMiner(g) {}
	~AppMiner() {}
	void init(unsigned max_degree, bool use_dag) {
		assert(k > 2);
		set_max_size(k);
		set_max_degree(max_degree);
		set_num_patterns(1);
		set_directed(use_dag);
		#ifdef ALGO_EDGE
		init_edgelist();
		#endif
		init_emb_list();
	}
	void print_output() {
		std::cout << "\n\ttotal_num_cliques = " << get_total_count() << "\n";
		std::cout << "\n\tremoved_edges = " << removed_edges.reduce() << "\n";
	}
	// toExtend (only extend the last vertex in the embedding)
	bool toExtend(unsigned level, unsigned pos) {
		return pos == level;
	}
	// toAdd (only add cliques)
	bool toAdd(unsigned level, VertexId vid, const EmbeddingList &emb_list, unsigned src_idx) { 
		return emb_list.get_label(vid) == level; 
	}
};

#include "DfsMining/engine.h"
