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
    if (n == 2)
      return 0;   // u2 extended from u0
    return n - 1; // u[i] extended from u[i-1]
  }

  static inline bool toAdd(unsigned n, PangolinGraph& g,
                           const BaseEmbedding& emb, unsigned, VertexId dst) {
    // std::cout << "\t emb: " << emb << ", dst=" << dst << "\n";
    if (n == 3) {
      if (dst <= emb.get_vertex(0))
        return false;
      if (!is_connected(g, dst, emb.get_vertex(1)))
        return false;
    } else {
      if (dst <= emb.get_vertex(n - 1))
        return false;
    }
    // if (g.get_degree(dst) < pattern.get_degree(n)) return false;
    // for (unsigned i = 1; i < n; ++i)
    //  if (dst == emb.get_vertex(i)) return false;

    // u3 (extended from u2) connected to u1
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
