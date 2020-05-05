#define USE_DFS
#define USE_CUSTOM
#define PRECOMPUTE
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
  bool toAdd(BaseEdgeEmbeddingList& emb_list, DFSCode& pattern) {
    return (is_frequent(emb_list, pattern) && is_canonical(pattern));
  }
  void print_output() {
    std::cout << "\n\ttotal_num_frquent_patterns = " << get_total_count()
              << "\n";
  }
};

#include "DfsMining/engine.h"
