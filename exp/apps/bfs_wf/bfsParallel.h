#ifndef BFS_PARALLEL_H
#define BFS_PARALLEL_H

#include "bfs.h"

using Level_ty = unsigned;

struct Update {
  GNode node;
  Level_ty level;

  Update (const GNode& node, const Level_ty& level)
    : node (node), level (level) 
  {}

  friend std::ostream& operator << (std::ostream& out, const Update& up) {
    out << "(node:" << up.node << ",level:" << up.level << ")";
    return out;
  }
};

struct GetLevel {

  Level_ty operator () (const Update& up) const {
    return up.level;
  }
};

struct Comparator {
  bool operator () (const Update& left, const Update& right) const {
    int d = left.level - right.level;

    if (d == 0) {
      // FIXME: assuming nodes are actually integer like
      d = left.node - right.node;
    }

    return (d < 0);
  }
};

struct VisitNhood {

  static const unsigned CHUNK_SIZE = DEFAULT_CHUNK_SIZE;

  Graph& graph;

  explicit VisitNhood (Graph& graph): graph (graph) {}

  template <typename C>
  void operator () (const Update& up, C& ctx) {

    // just like DES, we only lock the node being updated, but not its
    // outgoing neighbors
    graph.getData (up.node, Galois::MethodFlag::WRITE);
  }
};

struct OpFunc {

  static const unsigned CHUNK_SIZE = DEFAULT_CHUNK_SIZE;

  typedef int tt_does_not_need_aborts; // used by LevelExecBFS

  Graph& graph;
  ParCounter& numIter;

  OpFunc (Graph& graph, ParCounter& numIter): graph (graph), numIter (numIter) {}

  template <typename C>
    void operator () (const Update& up, C& ctx) const {

      if (graph.getData (up.node, Galois::MethodFlag::UNPROTECTED) == BFS_LEVEL_INFINITY) {

        graph.getData (up.node, Galois::MethodFlag::UNPROTECTED) = up.level;


        for (auto ni = graph.edge_begin (up.node, Galois::MethodFlag::UNPROTECTED)
            , eni = graph.edge_end (up.node, Galois::MethodFlag::UNPROTECTED); ni != eni; ++ni) {

          GNode dst = graph.getEdgeDst (ni);

          if (graph.getData (dst, Galois::MethodFlag::UNPROTECTED) == BFS_LEVEL_INFINITY) {
            ctx.push (Update (dst, up.level + 1));
          }
        }

      }

      numIter += 1;
    }

};

#endif //  BFS_PARALLEL_H
