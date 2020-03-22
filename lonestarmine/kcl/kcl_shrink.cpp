#include "../lonestarmine.h"
#include "DfsMining/vertex_miner.h"

// This is a implementation of the WWW'18 paper[1] using Sandslash API:
// [1] Danisch et al., Listing k-cliques in Sparse Real-World Graphs, WWW 2018
const char* name = "Kcl";
const char* desc = "Counts the K-Cliques in a graph using DFS traversal with shrinking graph";
const char* url  = 0;

#include "DfsMining/vertex_miner_api.h"
class MyAPI: public VertexMinerAPI<BaseEmbedding> {
public:
	static inline bool toExtend(unsigned level, unsigned pos, BaseEmbedding *emb) {
		return pos == level;
	}
	static inline bool toAdd(unsigned level, unsigned max_level, VertexId vid, 
		//unsigned src_idx, BYTE ccode, BaseEmbedding *emb) {
		unsigned src_idx, BYTE ccode, const std::vector<VertexId> *emb) { 
		return level == max_level-2 || ccode == level; 
	}
};

class AppMiner : public VertexMinerDFS<SimpleElement, BaseEmbedding, MyAPI, true, true, true, true> {
public:
	AppMiner(unsigned ms, int nt) : 
		VertexMinerDFS<SimpleElement, BaseEmbedding, MyAPI, true, true, true, true>(ms, nt, 2) {
		assert(ms > 3);
	}
	~AppMiner() {}
	void print_output() {
		std::cout << "\n\ttotal_num_cliques = " << get_total_count() << "\n";
	}
};

#include "DfsMining/engine.h"
