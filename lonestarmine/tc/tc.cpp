#include "../lonestarmine.h"
#include "BfsMining/vertex_miner.h"
#define TRIANGLE

const char* name = "TC";
const char* desc = "Counts the triangles in a graph (inputs do NOT need to be symmetrized)";
const char* url  = 0;

#include "BfsMining/vertex_miner_api.h"
class MyAPI: public VertexMinerAPI<BaseEmbedding> {
public:
	// toExtend (only extend the last vertex in the embedding)
	static bool toExtend(unsigned n, const BaseEmbedding &emb, unsigned pos) {
		return pos == n-1;
	}
	// toAdd (only add vertex connected to all the vertices in the embedding)
	static bool toAdd(unsigned n, Graph &g, const BaseEmbedding &emb, unsigned pos, VertexId dst) {
		return true;
	}
};

class AppMiner : public VertexMiner<SimpleElement, BaseEmbedding, MyAPI, true> {
public:
	AppMiner(unsigned ms, int nt) : 
		VertexMiner<SimpleElement, BaseEmbedding, MyAPI, true>(ms, nt, nblocks) {
		assert(ms == 3);
		set_num_patterns(1);
	}
	~AppMiner() {}
	void print_output() {
		std::cout << "\n\ttotal_num_triangles = " << get_total_count() << "\n";
	}
};
#include "BfsMining/engine.h"

