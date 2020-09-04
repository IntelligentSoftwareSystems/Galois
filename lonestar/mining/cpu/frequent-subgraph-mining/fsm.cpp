#include "MiningBench/Start.h"
#include "pangolin/BfsMining/edge_miner.h"

const char* name = "FSM";
const char* desc = "Frequent subgraph mining in a graph using BFS extension";
const char* url  = nullptr;

#include "pangolin/BfsMining/edge_miner_api.h"
class MyAPI : public EdgeMinerAPI<EdgeEmbedding> {
public:
};

class AppMiner : public EdgeMiner<LabeledElement, EdgeEmbedding, MyAPI, true> {
public:
  AppMiner(unsigned ms, int nt)
      : EdgeMiner<LabeledElement, EdgeEmbedding, MyAPI, true>(ms, nt) {
    if (ms <= 1) {
      printf("ERROR: command line argument k must be 2 or greater\n");
      exit(1);
    }
    if (filetype == "gr") {
      printf("ERROR: gr file is not acceptable for FSM. Add -ft=adj and use "
             "adj file instead.\n");
      exit(1);
    }
    set_threshold(minsup);
    total_num = 0;
  }
  ~AppMiner() {}
  void print_output() {
    std::cout << "\n\ttotal_num_frequent_patterns = " << this->total_num
              << "\n";
  }
};

#include "pangolin/BfsMining/engine.h"
