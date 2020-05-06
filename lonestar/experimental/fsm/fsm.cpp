#define USE_PID      // record pattern ID for the pattern-support map
#define USE_GSTL     // use Galois memory allocator for domain support
#define USE_DOMAIN   // use domain support
#define LARGE_SIZE   // for large number of embeddings
#define USE_EMB_LIST // use embedding list (SoA)
#define ENABLE_LABEL // enable vertex label
#define EDGE_INDUCED
#define CHUNK_SIZE 256
#include "pangolin.h"

const char* name = "FSM";
const char* desc = "Frequent subgraph mining in a graph using BFS extension";
const char* url  = 0;

class AppMiner : public EdgeMiner {
public:
  AppMiner(Graph* g) : EdgeMiner(g) {}
  ~AppMiner() {}
  void init() {
    assert(k > 1);
    set_max_size(k);
    set_threshold(minsup);
    construct_edgemap();
    total_num = 0;
  }
  void print_output() {
    std::cout << "\n\tNumber of frequent patterns (minsup=" << minsup
              << "): " << total_num << "\n";
  }
  void inc_total_num(int value) { total_num += value; }

private:
  int total_num;
};

#include "BfsMining/engine.h"
