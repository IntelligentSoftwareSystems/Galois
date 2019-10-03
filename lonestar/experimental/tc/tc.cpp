#define USE_DAG
#define TRIANGLE
#define USE_SIMPLE
#define USE_EMB_LIST
#define USE_BASE_TYPES
#define CHUNK_SIZE 256
#include "pangolin.h"
const char* name = "TC";
const char* desc = "Counts the triangles in a graph (inputs do NOT need to be symmetrized)";
const char* url  = 0;

class AppMiner : public VertexMiner {
public:
	AppMiner(Graph *g) : VertexMiner(g) {}
	~AppMiner() {}
	// toExtend (only extend the last vertex in the embedding: fast)
	bool toExtend(unsigned n, const BaseEmbedding &emb, VertexId src, unsigned pos) {
		return pos == n-1;
	}
	// toAdd (only add vertex that is connected to all the vertices in the embedding)
	bool toAdd(unsigned n, const BaseEmbedding &emb, VertexId dst, unsigned pos) {
		return false;
	}
	void print_output() {
		std::cout << "\n\ttotal_num_triangles = " << get_total_count() << "\n";
	}
};

#include "BfsMining/engine.h"

