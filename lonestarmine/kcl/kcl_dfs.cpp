#include "../lonestarmine.h"
#include "DfsMining/vertex_miner.h"

const char* name = "Kcl";
const char* desc = "Counts the K-Cliques in a graph using DFS traversal (using DAG)";
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
		return ccode == level; 
	}
};

class AppMiner : public VertexMinerDFS<SimpleElement, BaseEmbedding, 
	MyAPI, true, true, true, false, false, false, true> {
public:
	AppMiner(unsigned ms, int nt) : 
		VertexMinerDFS<SimpleElement, BaseEmbedding, MyAPI, 
			true, true, true, false, false, false, true>(ms, nt) {
		assert(ms > 2);
	}
	~AppMiner() {}
	void print_output() {
		std::cout << "\n\ttotal_num_cliques = " << get_total_count() << "\n";
		//std::cout << "\n\tremoved_edges = " << removed_edges.reduce() << "\n";
	}
};

#include "DfsMining/engine.h"
