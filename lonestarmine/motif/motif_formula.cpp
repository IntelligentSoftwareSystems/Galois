#include "../lonestarmine.h"
#include "DfsMining/vertex_miner.h"

// This is a implementation of the ICDM'15 paper using Sandslash API:
// Nesreen K. Ahmed et al., Efficient Graphlet Counting for Large Networks, ICDM 2015
const char* name = "Motif";
const char* desc = "Counts motifs in a graph using DFS traversal and formula";
const char* url  = 0;
int num_patterns[3] = {2, 6, 21};

#include "DfsMining/vertex_miner_api.h"
class MyAPI: public VertexMinerAPI<BaseEmbedding> {
typedef EmbeddingList<SimpleElement, BaseEmbedding> MyEmbeddingList;
public:
	static inline bool toExtend(unsigned level, unsigned v_idx, BaseEmbedding *emb) { 
		return v_idx == level; // only extend the last vertex
	}
	static inline bool toAdd(unsigned level, unsigned max_level, VertexId vid, 
		unsigned src_idx, BYTE ccode, const std::vector<VertexId> *emb) { 
		//return vid != (*emb)[0]; // non-canonical extension
		if (vid == (*emb)[0] || vid == (*emb)[1]) return false;
		//if (level > 1 && vid == (*emb)[2]) return false;
		return true;
	}
	static inline unsigned getPattern(unsigned level, unsigned max_level, VertexId src, 
		VertexId dst, BYTE ccode, unsigned pcode, BYTE src_idx, BaseEmbedding *emb) { 
		if (level == 1) { // for 3-motif 
			if (ccode == 3) return 0; // triangle
			else return 1; // wedge
		} else if (level == 2) { // for 4-motif
			if (max_level == 4) {
				if (src < dst && pcode == 0 && ccode == 3) { // clique
					return 5;
				} else if (pcode == 1 && ccode == 1) { // 4-cycle: from wedge, and v3 is connected with v0
					return 2;
				}
			} else { 
				exit(0);
			}
		} else { // for 5-motif and beyond
			exit(0);
		}
		return 0;
	}
	static inline void local_reduction(unsigned level, int pid, Ulong &counter) {
		if ((level == 1 && pid == 0) || // count triangles
			(level == 2 && (pid == 2 || pid == 5))) // count 4-cycle and 4-cliques
			counter ++;
	} 
};

class AppMiner : public VertexMinerDFS<SimpleElement, BaseEmbedding, MyAPI, false, false, true, false, true, true> {
public:
	AppMiner(unsigned ms, int nt) : 
		VertexMinerDFS<SimpleElement, BaseEmbedding, MyAPI, false, false, true, false, true, true>(ms, nt) {
		if (ms < 3 || ms > 4) {
			std::cout << "Only 3 and 4-motif are supported for now\n";
		}
		set_num_patterns(num_patterns[k-3]);
	}
	~AppMiner() {}
	void print_output() { printout_motifs(); }
};

#include "DfsMining/engine.h"

