#include "../lonestarmine.h"
#include "DfsMining/edge_miner.h"

const char* name = "FSM";
const char* desc = "Frequent subgraph mining using DFS exploration";
const char* url  = 0;
 
#include "DfsMining/edge_miner_api.h"
class MyAPI: public EdgeMinerAPI<DFSCode> {
public:
	static inline bool toAdd(BaseEdgeEmbeddingList &emb_list, DFSCode &pattern, unsigned threshold) {
		if (pattern.size() == 1) return true; // frequent single-edge embeddings pre_computed
		if (!is_frequent(emb_list, pattern, threshold)) return false;
		if (pattern.size() == 2) {
			if (pattern[1].from == 1) {
				if (pattern[0].fromlabel <= pattern[1].tolabel) return true;
			} else {
				assert(pattern[1].from == 0);
				if (pattern[0].fromlabel == pattern[0].tolabel) return false;
				if (pattern[0].tolabel == pattern[1].tolabel && pattern[0].fromlabel < pattern[1].tolabel) return true;
				if (pattern[0].tolabel <  pattern[1].tolabel) return true;
			}
			return false;
		}
		return is_canonical(pattern);
	}
};

class AppMiner : public EdgeMinerDFS<MyAPI, true> {
public:
	AppMiner(unsigned ms, int nt) : 
		EdgeMinerDFS<MyAPI, true>(ms, nt) {
		assert(k > 1);
		set_threshold(minsup);
	}
	~AppMiner() {}
	void print_output() {
		std::cout << "\n\ttotal_num_frquent_patterns = " << get_total_count() << "\n";
	}
};

#include "DfsMining/engine.h"

