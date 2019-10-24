#define USE_DAG
#define USE_SIMPLE
#define LARGE_SIZE // for large graphs such as soc-Livejournal1 and com-Orkut
#define ENABLE_STEAL
#define USE_EMB_LIST
#define CHUNK_SIZE 256
#define USE_BASE_TYPES
#include "pangolin.h"

const char* name = "Kcl";
const char* desc = "Counts the K-Cliques in a graph using BFS extension";
const char* url  = 0;

class AppMiner : public VertexMiner {
public:
	AppMiner(Graph *g) : VertexMiner(g) {}
	~AppMiner() {}
	void init() {
		assert(k > 2);
		set_max_size(k);
		set_num_patterns(1);
	}
	// toExtend (only extend the last vertex in the embedding: fast)
	bool toExtend(unsigned n, const BaseEmbedding &emb, VertexId src, unsigned pos) {
		return pos == n-1;
	}
	// toAdd (only add vertex that is connected to all the vertices in the embedding)
	bool toAdd(unsigned n, const BaseEmbedding &emb, VertexId dst, unsigned pos) {
		#ifdef USE_DAG
		return is_all_connected_dag(dst, emb, n-1);
		#else
		VertexId src = emb.get_vertex(pos);
		return (src < dst) && is_all_connected(dst, emb, n-1);
		#endif
	}
	void print_output() {
		std::cout << "\n\ttotal_num_cliques = " << get_total_count() << "\n";
	}
};

#include "BfsMining/engine.h"

