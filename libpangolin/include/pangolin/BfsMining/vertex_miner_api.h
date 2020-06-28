#pragma once
#include "pangolin/gtypes.h"

template <typename EmbeddingTy, bool use_wedge = true>
class VertexMinerAPI {
public:
  VertexMinerAPI() {}
  ~VertexMinerAPI() {}
  // toExtend
  static inline bool toExtend(unsigned, const EmbeddingTy&, unsigned) {
    return true;
  }

  // toAdd (only add non-automorphisms)
  static inline bool toAdd(unsigned n, PangolinGraph& g, const EmbeddingTy& emb,
                           unsigned pos, VertexId dst) {
    return !is_vertex_automorphism(n, g, emb, pos, dst);
  }

  static inline bool toAddOrdered(unsigned, PangolinGraph&, const EmbeddingTy&,
                                  unsigned, VertexId, PangolinGraph&) {
    return true;
  }

  // specify which vertex to extend when using matching order
  static inline unsigned getExtendableVertex(unsigned n) { return n - 1; }

  // given an embedding, return its pattern id (hash value)
  static inline unsigned getPattern(unsigned, PangolinGraph&, unsigned,
                                    VertexId, const EmbeddingTy&, unsigned) {
    return 0;
  }

protected:
  static inline bool is_vertex_automorphism(unsigned n, PangolinGraph& g,
                                            const EmbeddingTy& emb,
                                            unsigned idx, VertexId dst) {
    // unsigned n = emb.size();
    // the new vertex id should be larger than the first vertex id
    if (dst <= emb.get_vertex(0))
      return true;
    // the new vertex should not already exist in the embedding
    for (unsigned i = 1; i < n; ++i)
      if (dst == emb.get_vertex(i))
        return true;
    // the new vertex should not already be extended by any previous vertex in
    // the embedding
    for (unsigned i = 0; i < idx; ++i)
      if (is_connected(g, emb.get_vertex(i), dst))
        return true;
    // the new vertex id must be larger than any vertex id after its source
    // vertex in the embedding
    for (unsigned i = idx + 1; i < n; ++i)
      if (dst < emb.get_vertex(i))
        return true;
    return false;
  }
  static inline bool is_all_connected_dag(PangolinGraph& g, unsigned dst,
                                          const EmbeddingTy& emb, unsigned end,
                                          unsigned start = 0) {
    assert(end > 0);
    bool all_connected = true;
    for (unsigned i = start; i < end; ++i) {
      unsigned from = emb.get_vertex(i);
      if (!is_connected_dag(g, dst, from)) {
        all_connected = false;
        break;
      }
    }
    return all_connected;
  }
  static inline bool is_connected(PangolinGraph& g, unsigned a, unsigned b) {
    if (g.get_degree(a) == 0 || g.get_degree(b) == 0)
      return false;
    unsigned key    = a;
    unsigned search = b;
    if (g.get_degree(a) < g.get_degree(b)) {
      key    = b;
      search = a;
    }
    auto begin = g.edge_begin(search);
    auto end   = g.edge_end(search);
    return binary_search(g, key, begin, end);
  }
  static inline int is_connected_dag(PangolinGraph& g, unsigned key,
                                     unsigned search) {
    if (g.get_degree(search) == 0)
      return false;
    auto begin = g.edge_begin(search);
    auto end   = g.edge_end(search);
    return binary_search(g, key, begin, end);
  }
  static inline bool binary_search(PangolinGraph& g, unsigned key,
                                   PangolinGraph::edge_iterator begin,
                                   PangolinGraph::edge_iterator end) {
    auto l = begin;
    auto r = end - 1;
    while (r >= l) {
      auto mid       = l + (r - l) / 2;
      unsigned value = g.getEdgeDst(mid);
      if (value == key)
        return true;
      if (value < key)
        l = mid + 1;
      else
        r = mid - 1;
    }
    return false;
  }
  static inline unsigned find_motif_pattern_id(unsigned n, PangolinGraph& g,
                                               unsigned idx, VertexId dst,
                                               const EmbeddingTy& emb,
                                               BYTE* pre_pid,
                                               unsigned pos = 0) {
    unsigned pid = 0;
    if (n == 2) { // count 3-motifs
      pid = 1;    // 3-chain
      if (idx == 0) {
        if (is_connected(g, emb.get_vertex(1), dst))
          pid = 0; // triangle
        else if (use_wedge)
          pre_pid[pos] = 1; // wedge; used for 4-motif
      }
    } else if (n == 3) { // count 4-motifs
      unsigned num_edges = 1;
      pid                = emb.get_pid();
      if (pid == 0) { // extending a triangle
        for (unsigned j = idx + 1; j < n; j++)
          if (is_connected(g, emb.get_vertex(j), dst))
            num_edges++;
        pid = num_edges + 2; // p3: tailed-triangle; p4: diamond; p5: 4-clique
      } else {               // extending a 3-chain
        assert(pid == 1);
        std::vector<bool> connected(3, false);
        connected[idx] = true;
        for (unsigned j = idx + 1; j < n; j++) {
          if (is_connected(g, emb.get_vertex(j), dst)) {
            num_edges++;
            connected[j] = true;
          }
        }
        if (num_edges == 1) {
          pid             = 0; // p0: 3-path
          unsigned center = 1;
          if (use_wedge) {
            if (pre_pid[pos])
              center = 0;
          } else
            center =
                is_connected(g, emb.get_vertex(1), emb.get_vertex(2)) ? 1 : 0;
          if (idx == center)
            pid = 1; // p1: 3-star
        } else if (num_edges == 2) {
          pid             = 2; // p2: 4-cycle
          unsigned center = 1;
          if (use_wedge) {
            if (pre_pid[pos])
              center = 0;
          } else
            center =
                is_connected(g, emb.get_vertex(1), emb.get_vertex(2)) ? 1 : 0;
          if (connected[center])
            pid = 3; // p3: tailed-triangle
        } else {
          pid = 4; // p4: diamond
        }
      }
    } else { // count 5-motif and beyond
             // pid = find_motif_pattern_id_eigen(n, idx, dst, emb);
    }
    return pid;
  }
};
