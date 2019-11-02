#define USE_PID
#define USE_MAP
#define USE_SIMPLE
#define USE_CUSTOM
#define ENABLE_STEAL
#define VERTEX_INDUCED
#define CHUNK_SIZE 256
#include "pangolin.h"

const char* name = "Motif Counting";
const char* desc = "Counts the vertex-induced motifs in a graph using BFS extension";
const char* url  = 0;
int num_patterns[3] = {2, 6, 21};

class AppMiner : public VertexMiner {
public:
	AppMiner(Graph *g) : VertexMiner(g) {}
	~AppMiner() {}
	void init() {
		assert(k > 2);
		set_max_size(k);
		set_num_patterns(num_patterns[k-3]);
	}
	#ifdef USE_CUSTOM
	// customized pattern classification method
	unsigned getPattern(unsigned n, unsigned i, VertexId dst, const VertexEmbedding &emb, unsigned pos) { 
		if (n < 4) return find_motif_pattern_id(n, i, dst, emb, pos);
		return 0;
	}
	#endif
	void print_output() { printout_motifs(); }
};

#include "BfsMining/engine.h"

