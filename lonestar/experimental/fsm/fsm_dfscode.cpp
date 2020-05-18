#define USE_DFS
#define USE_DFSCODE
#define ENABLE_LABEL
#define EDGE_INDUCED
#define CHUNK_SIZE 4
#include "pangolin.h"

const char* name = "FSM";
const char* desc = "Frequent subgraph mining using DFS code";
const char* url  = 0;

class AppMiner : public EdgeMiner {
public:
  AppMiner(Graph* g) : EdgeMiner(g) {}
  ~AppMiner() {}
  void init(unsigned max_degree, bool use_dag) {
    assert(k > 1);
    set_max_size(k);
    set_threshold(minsup);
  }
};

#include "DfsMining/engine.h"
