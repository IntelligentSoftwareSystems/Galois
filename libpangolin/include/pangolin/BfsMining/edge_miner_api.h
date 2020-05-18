#pragma once
#include "pangolin/gtypes.h"

template <typename EmbeddingTy>
class EdgeMinerAPI {
public:
  EdgeMinerAPI() {}
  ~EdgeMinerAPI() {}

  // toExtend
  static inline bool toExtend(unsigned, const EmbeddingTy&, unsigned) {
    return true;
  }
  // toAdd (only add non-automorphisms)
  static inline bool toAdd(unsigned n, const EmbeddingTy& emb, unsigned pos,
                           VertexId src, VertexId dst, BYTE& existed,
                           const VertexSet& vertex_set) {
    return !is_edge_automorphism(n, emb, pos, src, dst, existed, vertex_set);
  }
  // customized pattern classification method
  static inline unsigned getPattern(unsigned GALOIS_USED_ONLY_IN_DEBUG(n),
                                    const EmbeddingTy&, unsigned, VertexId,
                                    VertexId) {
    assert(n < 4);
    return 0;
  }

protected:
  static inline bool is_quick_automorphism(unsigned size,
                                           const EmbeddingTy& emb,
                                           unsigned history, VertexId dst,
                                           BYTE& existed) {
    if (dst <= emb.get_vertex(0))
      return true;
    if (dst == emb.get_vertex(1))
      return true;
    if (history == 0 && dst < emb.get_vertex(1))
      return true;
    if (size == 2) {
    } else if (size == 3) {
      if (history == 0 && emb.get_history(2) == 0 && dst <= emb.get_vertex(2))
        return true;
      if (history == 0 && emb.get_history(2) == 1 && dst == emb.get_vertex(2))
        return true;
      if (history == 1 && emb.get_history(2) == 1 && dst <= emb.get_vertex(2))
        return true;
      if (dst == emb.get_vertex(2))
        existed = 1;
    } else {
      std::cout << "Error: should go to detailed check\n";
    }
    return false;
  }

  static inline bool is_edge_automorphism(unsigned size, const EmbeddingTy& emb,
                                          unsigned history, VertexId src,
                                          VertexId dst, BYTE& existed,
                                          const VertexSet& vertex_set) {
    if (size < 3)
      return is_quick_automorphism(size, emb, history, dst, existed);
    // check with the first element
    if (dst <= emb.get_vertex(0))
      return true;
    if (history == 0 && dst <= emb.get_vertex(1))
      return true;
    // check loop edge
    if (dst == emb.get_vertex(emb.get_history(history)))
      return true;
    if (vertex_set.find(dst) != vertex_set.end())
      existed = 1;
    // check to see if there already exists the vertex added;
    // if so, just allow to add edge which is (smaller id -> bigger id)
    if (existed && src > dst)
      return true;
    std::pair<VertexId, VertexId> added_edge(src, dst);
    for (unsigned index = history + 1; index < emb.size(); ++index) {
      std::pair<VertexId, VertexId> edge;
      edge.first  = emb.get_vertex(emb.get_history(index));
      edge.second = emb.get_vertex(index);
      // assert(edge.first != edge.second);
      int cmp = compare(added_edge, edge);
      if (cmp <= 0)
        return true;
    }
    return false;
  }
  static inline void swap(std::pair<VertexId, VertexId>& pair) {
    if (pair.first > pair.second) {
      auto tmp    = pair.first;
      pair.first  = pair.second;
      pair.second = tmp;
    }
  }
  static inline int compare(std::pair<VertexId, VertexId>& oneEdge,
                            std::pair<VertexId, VertexId>& otherEdge) {
    swap(oneEdge);
    swap(otherEdge);
    if (oneEdge.first == otherEdge.first)
      return oneEdge.second - otherEdge.second;
    else
      return oneEdge.first - otherEdge.first;
  }
};
