#include "../lonestarmine.h"
#include "DfsMining/vertex_miner.h"

const char* name = "Sgl";
const char* desc = "Counts a single arbitrary pattern in a graph using DFS traversal";
const char* url  = 0;

#include "DfsMining/vertex_miner_api.h"
class MyAPI: public VertexMinerAPI {
public:
	// customized pattern classification method
	static inline unsigned getPattern(unsigned level, unsigned max_level, VertexId src, 
		VertexId dst, BYTE ccode, unsigned pcode, BYTE src_idx, std::vector<VertexId> *emb) { 
		return get_pattern_id(level, dst, ccode, pcode, src_idx);
	}
};

// we do not consider clique here
// if the given pattern is a clique (which is easy to identify, i.e., n_e = n_v * (n_v - 1) / 2 ), 
// it will be handled by kcl
class AppMiner : public VertexMinerDFS<MyAPI, false, true, true, false, true, false, true, false> {
public:
	AppMiner(unsigned ms, int nt) : 
		VertexMinerDFS<MyAPI, false, true, true, false, true, false, true, false>(ms, nt) {
		assert(k > 2);
	}
	~AppMiner() {}
	void print_output() {
		std::cout << "\n\ttotal_num = " << get_total_count() << "\n";
	}
};

#include "DfsMining/engine.h"

