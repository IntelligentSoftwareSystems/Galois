// naive k-clique
//#define USE_DAG
#define USE_SIMPLE
#define LARGE_SIZE // for large graphs such as soc-Livejournal1 and com-Orkut
#define ENABLE_STEAL
#define USE_BASE_TYPES
#define CHUNK_SIZE 256
#include "pangolin.h"

const char* name = "Kcl";
const char* desc =
    "Counts the K-Cliques in an undirected graph using BFS extension";
const char* url = 0;

class AppMiner : public VertexMiner {
public:
  AppMiner(Graph* g) : VertexMiner(g) {}
  ~AppMiner() {}
  void init() {
    assert(k > 2);
    set_max_size(k);
    set_num_patterns(1);
  }
  // toExtend (extend every vertex in the embedding: slow)
  bool toExtend(unsigned n, const BaseEmbedding& emb, VertexId dst,
                unsigned pos) {
    return true;
  }
  bool toAdd(unsigned n, const BaseEmbedding& emb, VertexId dst, unsigned pos) {
#ifdef USE_DAG
    if (is_automorphism_dag<BaseEmbedding>(n, emb, pos, dst))
      return false;
    return is_all_connected_except_dag(dst, pos, emb);
#else
    VertexId src = emb.get_vertex(pos);
    if (dst <= src)
      return false;
    if (is_vertexInduced_automorphism<BaseEmbedding>(n, emb, pos, dst))
      return false;
    return is_all_connected_except(dst, pos, emb);
#endif
  }
  void print_output() {
    std::cout << "\n\ttotal_num_cliques = " << get_total_count() << "\n";
  }
};

#include "BfsMining/engine.h"
