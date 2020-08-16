#include "MiningBench/Start.h"
#include "pangolin/BfsMining/vertex_miner.h"

const char* name = "sgl";
const char* desc = "listing edge-induced subgraphs of a given pattern in a "
                   "graph using bfs extension";
const char* url = nullptr;

#include "pangolin/BfsMining/vertex_miner_api.h"
class MyAPI : public VertexMinerAPI<BaseEmbedding> {
public:
  // matching order of the pattern
  static inline unsigned getExtendableVertex(unsigned n) {
    if (n == 3)
      return 1;   // u3 extended from u1
    return n - 1; // u[i] extended from u[i-1]
  }

  static inline bool toAdd(unsigned n, PangolinGraph& g,
                           const BaseEmbedding& emb, unsigned, VertexId dst) {
    // std::cout << "\t emb: " << emb << ", dst=" << dst << ", pos=" << pos <<
    // "\n";
    // u3 > u2
    if (n == 3) {
      if (dst <= emb.get_vertex(2))
        return false;
    }
    // both u2 and u3 (extended from u1) connected to u0
    if (!is_connected(g, dst, emb.get_vertex(0)))
      return false;
    return true;
  }
};

class AppMiner
    : public VertexMiner<SimpleElement, BaseEmbedding, MyAPI, 0, 1, 0, 1> {
public:
  AppMiner(unsigned ms, int nt)
      : VertexMiner<SimpleElement, BaseEmbedding, MyAPI, 0, 1, 0, 1>(ms, nt,
                                                                     nblocks) {}
  ~AppMiner() {}
  void print_output() {
    std::cout << "\n\ttotal_num_subgraphs = " << get_total_count() << "\n";
  }
};

#include "pangolin/BfsMining/engine.h"
