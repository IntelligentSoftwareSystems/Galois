#define USE_PID
#define USE_GSTL
#define USE_DOMAIN
#define USE_EMB_LIST
#define ENABLE_LABEL
#define EDGE_INDUCED
#define CHUNK_SIZE 256
#include "pangolin.h"

const char* name = "FSM";
const char* desc = "Frequent subgraph mining in a graph using BFS extension";
const char* url  = 0;

class AppMiner : public EdgeMiner {
public:
	AppMiner(Graph *g, unsigned size) : EdgeMiner(g, size) {
		total_num = 0;
	}
	~AppMiner() {}
	void print_output() {
		std::cout << "\n\tNumber of frequent patterns (minsup=" << minsup << "): " << total_num << "\n";
	}
	void inc_total_num(int value) { total_num += value; }
private:
	int total_num;
};

#include "BfsMining/engine.h"

