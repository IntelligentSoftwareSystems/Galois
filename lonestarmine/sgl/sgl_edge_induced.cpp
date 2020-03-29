#include "../lonestarmine.h"
#include "DfsMining/vertex_miner.h"

const char* name = "Sgl";
const char* desc = "Listing edge-induced subgraphs of a given pattern in a graph using DFS extension";
const char* url  = 0;
#ifdef CYCLE
uint32_t automorph_group_id[4] = {0, 1, 0, 1}; // 4-cycle
const BYTE pt_ccode = 22; // '10110' 4-cycle
#else
uint32_t automorph_group_id[4] = {0, 0, 1, 1}; // diamond
const BYTE pt_ccode = 15; // '01111' diamond
#endif

#include "DfsMining/vertex_miner_api.h"
class MyAPI: public VertexMinerAPI {
public:
	// customized pattern classification method
	//static inline unsigned getPattern(unsigned level, unsigned max_level, VertexId src, 
	//	VertexId dst, BYTE ccode, unsigned pcode, BYTE src_idx, std::vector<VertexId> *emb) { 
	//	return get_pattern_id(level, dst, ccode, pcode, src_idx);
	//}

	// toAdd (only add vertex that is connected to all the vertices in the embedding)
	static inline bool toAdd(unsigned level, unsigned max_level, VertexId dst, 
		unsigned pos, BYTE ccode, const std::vector<VertexId> *emb) { 
		unsigned n = level + 1;
		assert(n > 1);
		// the first vertex should always has the smallest id (if it is not special)
		if (!fv && dst <= (*emb)[0]) return false;
		// if the degree is smaller than that of its corresponding query vertex
		// if this vertex already exists in the embedding
		for (unsigned i = 0; i < n; ++i) if (dst == (*emb)[i]) return false;
		// check the connectivity with previous vertices in the embedding
		// if connected in the pattern, they should be connected in the original graph
		// otherwise it is not a match
		unsigned nbits = n*(n-1)/2-1;
		BYTE code = pt_ccode >> nbits;
		code = code & ((1 << (nbits+n)) - 1);
		//std::cout << "\tcode=" << unsigned(code) << ", ccode&code=" << unsigned(ccode&code) << "\n";
		if ((ccode & code) != code) return false;
		// for any existing vertex that is in the same automorphism group
		// the new vertex should have a larger ID
		// to avoid over-counting
		for (unsigned i = 0; i < n; ++i) {
			if (automorph_group_id[i] == automorph_group_id[n]) {
				//std::cout << "\t comparing dst=" << dst << " and i=" << i << ", src=" << (*emb)[i] << "\n";
				if (dst < (*emb)[i]) return false;
			}
		}
		return true;
	}
};

// we do not consider clique here
// if the given pattern is a clique (just check:  n_e == n_v * (n_v - 1) / 2 ), 
// it will be handled by kcl
class AppMiner : public VertexMinerDFS<MyAPI, false, true, true, false, true, false, true, false> {
//typedef EmbeddingList<true,true,false,false,false,false> EmbeddingListTy;
public:
	AppMiner(unsigned ms, int nt) : 
		VertexMinerDFS<MyAPI, false, true, true, false, true, false, true, false>(ms, nt) {
		assert(k > 2);
		//this->read_presets();
	}
	~AppMiner() {}
	void print_output() {
		std::cout << "\n\ttotal_num_subgraphs = " << get_total_count() << "\n";
	}
};

#include "DfsMining/engine.h"
