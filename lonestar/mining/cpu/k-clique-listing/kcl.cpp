#include "MiningBench/Start.h"
#include "pangolin/BfsMining/vertex_miner.h"

const char* name = "Kcl";
const char* desc = "Listing cliques of size k in a graph using BFS extension";
const char* url  = nullptr;

#include "pangolin/BfsMining/vertex_miner_api.h"
class MyAPI : public VertexMinerAPI<BaseEmbedding> {
public:
  // toExtend (only extend the last vertex in the embedding)
  static bool toExtend(unsigned n, const BaseEmbedding&, unsigned pos) {
    return pos == n - 1;
  }
  // toAdd (only add vertex connected to all the vertices in the embedding)
  static bool toAdd(unsigned n, PangolinGraph& g, const BaseEmbedding& emb,
                    unsigned, VertexId dst) {
    return is_all_connected_dag(g, dst, emb, n - 1);
  }
};

class AppMiner : public VertexMiner<SimpleElement, BaseEmbedding, MyAPI, true> {
public:
  AppMiner(unsigned ms, int nt)
      : VertexMiner<SimpleElement, BaseEmbedding, MyAPI, true>(ms, nt,
                                                               nblocks) {
    if (ms <= 2) {
      printf("ERROR: command line argument k must be 3 or greater\n");
      exit(1);
    }
    set_num_patterns(1);
  }
  ~AppMiner() {}
  void print_output() {
    std::cout << "\n\ttotal_num_cliques = " << get_total_count() << "\n";
  }
};

#include "pangolin/BfsMining/engine.h"
