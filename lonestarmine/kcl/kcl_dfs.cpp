#include "../lonestarmine.h"
#include "DfsMining/vertex_miner.h"

const char* name = "Kcl";
const char* desc = "Counts the K-Cliques in a graph using DFS traversal (using DAG)";
const char* url  = 0;

#include "DfsMining/vertex_miner_api.h"
class MyAPI: public VertexMinerAPI {
public:
	static inline bool toExtend(unsigned level, unsigned pos, std::vector<VertexId> *emb) {
		return pos == level;
	}
	static inline bool toAdd(unsigned level, unsigned max_level, VertexId vid, 
		unsigned src_idx, BYTE ccode, const std::vector<VertexId> *emb) { 
		return ccode == level; 
	}
};

class AppMiner : public VertexMinerDFS<MyAPI, true, true, true, false, false, false, false, true> {
public:
	AppMiner(unsigned ms, int nt) : 
		VertexMinerDFS<MyAPI, true, true, true, false, false, false, false, true>(ms, nt) {
		assert(ms > 2);
	}
	~AppMiner() {}
	void print_output() {
		std::cout << "\n\ttotal_num_cliques = " << get_total_count() << "\n";
	}
};

#include "DfsMining/engine.h"
