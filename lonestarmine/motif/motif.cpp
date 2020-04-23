#include "lonestarmine.h"
#include "BfsMining/vertex_miner.h"

const char* name = "Motif Counting";
const char* desc = "Counts the vertex-induced motifs in a graph using BFS extension";
const char* url  = 0;
int num_patterns[3] = {2, 6, 21};

#include "BfsMining/vertex_miner_api.h"
class MyAPI: public VertexMinerAPI<VertexEmbedding> {
public:
	// customized pattern classification method
	static unsigned getPattern(unsigned n, Graph &g, unsigned i, VertexId dst, const VertexEmbedding &emb, BYTE *pre_pid, unsigned pos) {
		assert(n < 4);
		return find_motif_pattern_id(n, g, i, dst, emb, pre_pid, pos);
	}
};

class AppMiner : public VertexMiner<SimpleElement, VertexEmbedding, MyAPI, false, false, true> {
public:
	AppMiner(unsigned ms, int nt) : 
		VertexMiner<SimpleElement, VertexEmbedding, MyAPI, false, false, true>(ms, nt, nblocks) {
		assert(ms > 2);
		set_num_patterns(num_patterns[k-3]);
	}
	~AppMiner() {}
	void print_output() { printout_motifs(); }
};

#include "BfsMining/engine.h"

