#include "lonestarmine.h"
#include "pangolin/DfsMining/vertex_miner.h"

// This is a implementation of the WWW'18 paper[1] using Sandslash API:
// [1] Danisch et al., Listing k-cliques in Sparse Real-World Graphs, WWW 2018
const char* name = "Kcl";
const char* desc = "Counts the K-Cliques in a graph using DFS traversal with shrinking graph";
const char* url  = 0;

#include "pangolin/DfsMining/vertex_miner_api.h"
class MyAPI: public VertexMinerAPI {
public:
	static inline bool toExtend(unsigned level, unsigned pos, std::vector<VertexId> *) {
		return pos == level;
	}
	static inline bool toAdd(unsigned level, unsigned max_level, VertexId, 
		unsigned, BYTE ccode, const std::vector<VertexId> *) { 
		return level == max_level-2 || ccode == level; 
	}
};
#define EDGE_PAR 0
#define START_LEVEL (EDGE_PAR+1)
class AppMiner : public VertexMinerDFS<MyAPI, true, true, true, true, false, false, EDGE_PAR, true> {
public:
	AppMiner(unsigned ms, int nt) : 
		VertexMinerDFS<MyAPI, true, true, true, true, false, false, EDGE_PAR, true>(ms, nt, START_LEVEL) {
		if (ms < START_LEVEL+2) {
			std::cout << "k should be at least " << START_LEVEL+2 << "\n";
			exit(1);
		}
	}
	~AppMiner() {}
	void print_output() {
		std::cout << "\n\ttotal_num_cliques = " << get_total_count() << "\n";
	}
};

#include "pangolin/DfsMining/engine.h"
