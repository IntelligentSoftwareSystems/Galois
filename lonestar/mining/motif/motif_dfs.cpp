#include "lonestarmine.h"
#include "pangolin/DfsMining/vertex_miner.h"

const char* name = "Motif";
const char* desc = "Counts motifs in a graph using DFS traversal";
const char* url  = 0;
int num_patterns[3] = {2, 6, 21};

#include "pangolin/DfsMining/vertex_miner_api.h"
class MyAPI: public VertexMinerAPI {
public:
	// customized pattern classification method
	static inline unsigned getPattern(unsigned level, unsigned, VertexId, 
		VertexId dst, BYTE ccode, unsigned pcode, BYTE src_idx, const std::vector<VertexId> *) { 
		if (level < 3) {
			return get_pattern_id(level, dst, ccode, pcode, src_idx);
		} else {
			std::cout << "only support 3 and 4-motif for now\n";
			exit(0);
		}
	}
};

class AppMiner : public VertexMinerDFS<MyAPI, false, false, true, false, true, false, true, false> {
public:
	AppMiner(unsigned ms, int nt) : 
		VertexMinerDFS<MyAPI, false, false, true, false, true, false, true, false>(ms, nt) {
		assert(k > 2);
		set_num_patterns(num_patterns[k-3]);
	}
	~AppMiner() {}
	void print_output() { printout_motifs(); }
};

#include "pangolin/DfsMining/engine.h"

