#define USE_DFS
#define USE_MAP
#define USE_PID
//#define DIAMOND
#define ALGO_EDGE
#define LARGE_SIZE // for large graphs such as soc-Livejournal1 and com-Orkut
#define USE_SIMPLE
#define NON_CANONICAL
#define VERTEX_INDUCED
#define CHUNK_SIZE 256
#include "pangolin.h"

const char* name = "Sgl";
const char* desc =
    "Counts a single arbitrary pattern in a graph using DFS traversal";
const char* url     = 0;
int num_patterns[3] = {2, 6, 21};

class AppMiner : public VertexMiner {
public:
  AppMiner(Graph* g) : VertexMiner(g) {}
  ~AppMiner() {}
  void init(unsigned max_degree, bool use_dag) {
    assert(k > 2);
    set_max_size(k);
    set_max_degree(max_degree);
    set_num_patterns(num_patterns[k - 3]);
    set_directed(use_dag);
#ifdef ALGO_EDGE
    init_edgelist();
#endif
    init_emb_list();
  }
  void print_output() {
    std::cout << "\n\ttotal_num = " << get_total_count() << "\n";
  }
#ifdef NON_CANONICAL
  unsigned getPattern(unsigned level, VertexId vid, EmbeddingList& emb_list,
                      unsigned previous_pid, BYTE src_idx) {
    if (level < 3)
      return find_pattern_id_dfs(level, vid, emb_list, previous_pid, src_idx);
    return 0;
  }
  void reduction(unsigned level, EmbeddingList& emb_list, VertexId src_idx,
                 VertexId vid, unsigned previous_pid) {
    unsigned qcode = emb_list.get_label(vid);
    if (previous_pid == 0 && qcode < 5 && qcode != 3)
      total_num += 1;
    if (previous_pid == 1) {
      if (src_idx == 0) {
        if (qcode == 3 || qcode == 5)
          total_num += 1;
      } else {
        if (qcode == 3 || qcode == 6)
          total_num += 1;
        ;
      }
    }
  }
/*
// diamond
void reduction(unsigned level, EmbeddingList &emb_list, VertexId src_idx,
VertexId vid, unsigned previous_pid) { if (previous_pid == 0) { // extending a
triangle if (emb_list.get_label(vid) == 3 || emb_list.get_label(vid) == 5 ||
emb_list.get_label(vid) == 6) { total_num += 1; // diamond
        }
    } else if (emb_list.get_label(vid) == 7) total_num += 1; // diamond
}
//*/
/*
// 4-cycle
void reduction(unsigned level, EmbeddingList &emb_list, VertexId src_idx,
VertexId vid, unsigned previous_pid) { if (previous_pid == 1) { // extending a
wedge if (src_idx == 0) { if (emb_list.get_label(vid) == 6) total_num += 1; //
4-cycle } else { if (emb_list.get_label(vid) == 5) total_num += 1; // 4-cycle
        }
    }
}
//*/
/*
//3-star
void reduction(unsigned level, EmbeddingList &emb_list, VertexId src_idx,
VertexId vid, unsigned previous_pid) { if (previous_pid == 1) { // extending a
wedge if (src_idx == 0) { if (emb_list.get_label(vid) == 1) total_num += 1; //
3-star } else { if (emb_list.get_label(vid) == 2) total_num += 1; // 3-star
        }
    }
}
            if (emb_list.get_label(vid) == 3 || emb_list.get_label(vid) == 5)
pid = 3; // tailed-triangle if (emb_list.get_label(vid) == 3 ||
emb_list.get_label(vid) == 6) pid = 3; // tailed-triangle

//*/
#endif
};

#include "DfsMining/engine.h"
