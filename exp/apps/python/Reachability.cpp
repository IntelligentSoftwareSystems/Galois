#include "Reachability.h"
#include "Galois/Statistic.h"
#include "Galois/Bag.h"

typedef Galois::InsertBag<GNode> NodeSet;

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

static void initialize(Graph *g) {
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
}

template<bool isBackward>
static void findOutward(Graph *g, NodeList l, int hop) {
  NodeSet w;

  // set distance of l.nodes to 0 
  for (auto i = 0; i < l.num; ++i) {
    auto n = l.nodes[i];
    auto& dist = (!isBackward) ? g->getData(n).II.vInt1 : g->getData(n).II.vInt2;
    dist = 0;
    w.push_back(n);
  }

  // move from w up to hop steps
  Galois::for_each_local(w, Reach<isBackward>{*g, hop});
}

// collect nodes marked within hop steps
template<bool isBackward>
static NodeSet collectOutward(Graph *g, int hop) {
  NodeSet w;

  Galois::do_all_local(*g, 
    [g, &w, hop] (GNode n)
      {
        auto dist = (!isBackward) ? (*g).getData(n).II.vInt1 : (*g).getData(n).II.vInt2;
        if (dist <= hop) {
          w.push_back(n);
        }
      },
    Galois::do_all_steal<true>()
    );

  return w;
}

// collect nodes marked within hop steps from both src and dst
static NodeSet collectBetween(Graph *g, int hop) {
  NodeSet w;

  Galois::do_all_local(
    *g, 
    [g, &w, hop] (GNode n) 
      {
        auto& data = (*g).getData(n);
        auto distF = data.II.vInt1, distB = data.II.vInt2;

        // only count nodes on shortest paths
        if (distF + distB <= hop) {
          w.push_back(n);
        }
      },
    Galois::do_all_steal<true>()
    );

  return w;
}

static NodeList allocateNodeList(NodeSet& w) {
  auto num = std::distance(w.begin(), w.end());
  NodeList l = createNodeList(num);
  auto i = 0;
  for (auto n: w) {
    l.nodes[i++] = n;
  }
  return l;
}

template<bool isBackward>
static NodeList findReachableOutward(Graph *g, NodeList l, int hop) {
//  Galois::StatManager statManager;

//  Galois::StatTimer T("findReachableOutward");
//  T.start();

  initialize(g);
  findOutward<isBackward>(g, l, hop);
  NodeSet w = collectOutward<isBackward>(g, hop);

//  T.stop();

  return allocateNodeList(w); 
}

// find forward, e.g. where src goes to
NodeList findReachableTo(Graph *g, NodeList src, int hop) {
  return findReachableOutward<false>(g, src, hop);
}

// find backward, e.g. where dst comes from
NodeList findReachableFrom(Graph *g, NodeList dst, int hop) {
  return findReachableOutward<true>(g, dst, hop);
}

NodeList findReachableBetween(Graph *g, NodeList src, NodeList dst, int hop) {
//  Galois::StatManager statManager;

//  Galois::StatTimer T("findReachableBetween");
//  T.start();

  initialize(g);

  // move forward from src
  findOutward<false>(g, src, hop);

  // move backward from dst
  findOutward<true>(g, dst, hop);

  // intersect the two movements
  NodeSet intersect = collectBetween(g, hop);

//  T.stop();

  return allocateNodeList(intersect);
}

