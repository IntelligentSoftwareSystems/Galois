#include "Reachability.h"
#include "Filter.h"
#include "Galois/Statistic.h"
#include "Galois/Bag.h"

#include <limits>

static const size_t DIST_INFINITY = std::numeric_limits<size_t>::max() - 1;

template<bool isBackward>
struct Reach {
  Graph& g;
  int hop;

  Reach(Graph& g, int h): g(g), hop(h) {}

  void operator()(const GNode n, Galois::UserContext<GNode>& ctx) {
    auto& data = g.getData(n);
    auto& dist = (isBackward) ? data.II.vInt2 : data.II.vInt1;
    auto newDist = dist + 1;

    if (dist >= hop) {
      return;
    }

    if (isBackward) {
      for (auto e: g.in_edges(n)) {
        auto ngh = g.getEdgeDst(e);
        auto& distB = g.getData(ngh).II.vInt2;
        if (distB <= newDist) {
          continue;
        }
        distB = newDist;
        if (newDist < hop) {
          ctx.push(ngh);
        }
      }
    } else {
      for (auto e: g.edges(n)) {
        auto ngh = g.getEdgeDst(e);
        auto& distF = g.getData(ngh).II.vInt1;
        if (distF <= newDist) {
          continue;
        }
        distF = newDist;
        if (newDist < hop) {
          ctx.push(ngh);
        }
      }
    } // end else
  }
};

// report nodes touched by shortest paths from src to dst within hop steps
//   1. go forward from src by hop steps 
//   2. go backward from dst by hop steps
//   3. intersect nodes touched by 1. and 2.
NodeList findReachable(Graph *g, NodeList src, NodeList dst, int hop) {
  Galois::StatManager statManager;
  Galois::InsertBag<GNode> from, to, intersect;

  // set all distance to infinity
  Galois::do_all_local(*g, 
    [=] (GNode n)
      {
        auto& data = (*g).getData(n);
        data.II.vInt1 = DIST_INFINITY;
        data.II.vInt2 = DIST_INFINITY;
      },
    Galois::do_all_steal<true>()
    );

  // set forward distance of src to 0 
  for (auto i = 0; i < src.num; ++i) {
    auto n = src.nodes[i];
    g->getData(n).II.vInt1 = 0;
    from.push_back(n);
  }

  // set backward distance of dst to 0
  for (auto i = 0; i < dst.num; ++i) {
    auto n = dst.nodes[i];
    g->getData(n).II.vInt2 = 0;
    to.push_back(n);
  }

  Galois::StatTimer T;
  T.start();

  // move from src
  Galois::for_each_local(from, Reach<false>{*g, hop});

  // move from dst
  Galois::for_each_local(to, Reach<true>{*g, hop});

  // intersect the above two movements
  Galois::do_all_local(
    *g, 
    [g, &intersect, hop] (GNode n) 
      {
        auto& data = (*g).getData(n);
        auto distF = data.II.vInt1, distB = data.II.vInt2;

        // untouched
        if (DIST_INFINITY == distF || DIST_INFINITY == distB) {
          return;
        }

        // only count nodes on shortest paths
        if (distF + distB <= hop) {
          intersect.push_back(n);
        }
      },
    Galois::do_all_steal<true>()
    );

  T.stop();

  auto num = std::distance(intersect.begin(), intersect.end());
  NodeList l = createNodeList(num);
  auto i = 0;
  for (auto n: intersect) {
    l.nodes[i++] = n;
  }
  return l;
}

