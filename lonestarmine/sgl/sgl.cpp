#include "../lonestarmine.h"
#include "BfsMining/vertex_miner.h"

const char* name = "Sgl";
const char* desc = "Listing edge-induced subgraphs of a given pattern in a graph using BFS extension";
const char* url  = 0;
#ifdef CYCLE
uint32_t automorph_group_id[4] = {0, 1, 0, 1}; // 4-cycle
const BYTE pt_ccode = 22; // '10110' 4-cycle
#else
uint32_t automorph_group_id[4] = {0, 0, 1, 1}; // diamond
const BYTE pt_ccode = 15; // '01111' diamond
#endif

#include "BfsMining/vertex_miner_api.h"
class MyAPI: public VertexMinerAPI<BaseEmbedding> {
public:
	static inline bool toAddOrdered(unsigned n, Graph &g, const BaseEmbedding &emb, unsigned pos, VertexId dst, Graph &pattern) {
		//std::cout << "\t emb: " << emb << ", dst=" << dst << ", pos=" << pos << "\n";
		if (!fv && dst <= emb.get_vertex(0)) return false;
		if (g.get_degree(dst) < pattern.get_degree(n)) return false;
		for (unsigned i = 0; i < n; ++i) if (dst == emb.get_vertex(i)) return false;
		for (auto e : pattern.edges(n)) {
			VertexId q_dst = pattern.getEdgeDst(e);
			unsigned q_order = q_dst;
			if (q_order < n && q_order != pos) {
				VertexId d_vertex = emb.get_vertex(q_order);
				//if (debug && n == 3 && pos == 1 && emb.get_vertex(pos) == 3 && dst == 5) std:: cout << "\t\t d_vedrtex = " << d_vertex << "\n";
				if (!is_connected(g, dst, d_vertex)) return false;
			}
		}
		for (unsigned i = 0; i < n; ++i) {
			if (automorph_group_id[i] == automorph_group_id[n]) {
				if (dst < emb.get_vertex(i)) return false;
			}
		}
		return true;
	}
};

class AppMiner : public VertexMiner<SimpleElement, BaseEmbedding, MyAPI, false, true, false, true> {
public:
	AppMiner(unsigned ms, int nt) : 
		VertexMiner<SimpleElement, BaseEmbedding, MyAPI, false, true, false, true>(ms, nt, nblocks) {
		assert(ms > 2);
	}
	~AppMiner() {}
	void print_output() {
		std::cout << "\n\ttotal_num_subgraphs = " << get_total_count() << "\n";
	}
};

#include "BfsMining/engine.h"

