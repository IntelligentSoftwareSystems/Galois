//#define USE_DAG
#define USE_DFS
#define USE_MAP
//#define USE_PID
#define ALGO_EDGE
#define USE_ADHOC
#define USE_SIMPLE
//#define USE_EMB_LIST
#define VERTEX_INDUCED
#define CHUNK_SIZE 256
#include "pangolin.h"

// This is a implementation of the ICDM'15 paper:
// Nesreen K. Ahmed et al., Efficient Graphlet Counting for Large Networks, ICDM 2015
const char* name = "Motif";
const char* desc = "Counts motifs in a graph using DFS traversal";
const char* url  = 0;
int num_patterns[3] = {2, 6, 21};

class AppMiner : public VertexMiner {
public:
	AppMiner(Graph *g, unsigned size, int np, bool use_dag, unsigned c) : VertexMiner(g, size, np, use_dag, c) {}
	~AppMiner() {}
	// customized pattern classification method
	unsigned getPattern(unsigned n, unsigned i, VertexId dst, const VertexEmbedding &emb, unsigned previous_pid) { 
		if (n < 4) return find_motif_pattern_id_dfs(n, i, dst, emb, previous_pid);
		return 0;
	}
	void print_output() { printout_motifs(); }
};

#include "DfsMining/engine.h"

