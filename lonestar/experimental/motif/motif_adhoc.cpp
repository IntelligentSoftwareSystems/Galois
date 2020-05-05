#define USE_DFS
#define USE_MAP
#define USE_PID
#define USE_OPT
#define ALGO_EDGE
#define USE_SIMPLE
#define MOTIF_ADHOC
#define VERTEX_INDUCED
#define CHUNK_SIZE 256
#include "pangolin.h"

// This is a implementation of the ICDM'15 paper:
// Nesreen K. Ahmed et al., Efficient Graphlet Counting for Large Networks, ICDM
// 2015
const char* name    = "Motif";
const char* desc    = "Counts motifs in a graph using DFS traversal";
const char* url     = 0;
int num_patterns[3] = {2, 6, 21};

class AppMiner : public VertexMiner {
public:
  AppMiner(Graph* g) : VertexMiner(g) {}
  ~AppMiner() {}
  void init(unsigned max_degree, bool use_dag) {
    std::cout << "Starting Ad-Hoc Motif Solver\n";
    assert(k > 2 && k < 5);
    set_max_size(k);
    set_max_degree(max_degree);
    set_num_patterns(num_patterns[k - 3]);
    set_directed(use_dag);
#ifdef ALGO_EDGE
    init_edgelist();
#endif
    init_emb_list();
  }
  void print_output() { printout_motifs(); }
  void edge_process_opt() {
    galois::do_all(
        galois::iterate(edge_list),
        [&](const auto& edge) {
          EmbeddingList* emb_list = emb_lists.getLocal();
          // std::cout << "debug: edge = " << edge.to_string() << "\n";
          emb_list->init_edge(edge);
          dfs_extend_adhoc(1, *emb_list);
          solve_motif_equations(*emb_list);
          emb_list->clear_labels(emb_list->get_vid(0, 0));
        },
        galois::chunk_size<CHUNK_SIZE>(), galois::steal(),
        galois::loopname("DfsAdhocSolver"));
    motif_count();
  }
  // construct the subgraph induced by edge (u, v)'s neighbors
  void dfs_extend_adhoc(unsigned level, EmbeddingList& emb_list) {
    if (level == max_size - 2) {
      for (auto e : graph->edges(emb_list.get_vid(1, 0))) {
        auto w = graph->getEdgeDst(e);
        if (emb_list.get_label(w) == 1)
          emb_list.inc_tri_count();
      }
      return;
    }
    assert(max_size == 4);
    emb_list.triangles_and_wedges();
    emb_list.cycle();
    emb_list.clique();
    accumulators[5] += emb_list.get_clique4_count();
    accumulators[2] += emb_list.get_cycle4_count();
  }
};

#include "DfsMining/engine.h"
