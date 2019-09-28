#define USE_DFS
#define ENABLE_LABEL
#define EDGE_INDUCED
#define CHUNK_SIZE 4
#include "pangolin.h"

const char* name = "FSM";
const char* desc = "Frequent subgraph mining using DFS code";
const char* url  = 0;
 
class AppMiner : public Miner {
public:
	AppMiner(Graph *g) : Miner(g) {}
	~AppMiner() {}
};

#include "DfsMining/engine.h"

