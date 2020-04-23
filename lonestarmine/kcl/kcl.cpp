#include "lonestarmine.h"
#include "pangolin/BfsMining/vertex_miner.h"

const char* name = "Kcl";
const char* desc = "Counts the K-Cliques in a graph using BFS extension";
const char* url  = 0;

#include "pangolin/BfsMining/vertex_miner_api.h"
class MyAPI: public VertexMinerAPI<BaseEmbedding> {
public:
	// toExtend (only extend the last vertex in the embedding)
	static bool toExtend(unsigned n, const BaseEmbedding &emb, unsigned pos) {
		return pos == n-1;
	}
	// toAdd (only add vertex connected to all the vertices in the embedding)
	static bool toAdd(unsigned n, Graph &g, const BaseEmbedding &emb, unsigned pos, VertexId dst) {
		return is_all_connected_dag(g, dst, emb, n-1);
	}
};

class AppMiner : public VertexMiner<SimpleElement, BaseEmbedding, MyAPI, true> {
public:
	AppMiner(unsigned ms, int nt) : 
		VertexMiner<SimpleElement, BaseEmbedding, MyAPI, true>(ms, nt, nblocks) {
		assert(ms > 2);
		set_num_patterns(1);
	}
	~AppMiner() {}
	void print_output() {
		std::cout << "\n\ttotal_num_cliques = " << get_total_count() << "\n";
	}
};

#include "pangolin/BfsMining/engine.h"

