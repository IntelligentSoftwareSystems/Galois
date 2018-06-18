#include "Coarsen.h"
#include "galois/Reduction.h"

#include <string>
#include <unordered_map>
#include <list>

typedef std::list<std::string> List;
typedef std::unordered_map<std::string, List> EdgeMap;
typedef std::unordered_map<std::string, GNode> NodeMap;

struct Updater {
  void operator()(EdgeMap& lhs, std::pair<std::string, std::string> rhs) {
    lhs[rhs.first].push_back(rhs.second);
  }
};

struct Reducer {
  void operator()(EdgeMap& lhs, EdgeMap& rhs) {
    // merge rhs to lhs elementwise
    for (auto& rn : rhs) {
      std::copy(rn.second.begin(), rn.second.end(),
                std::back_inserter(lhs[rn.first]));
    }
    // remove duplicates
    for (auto& ln : lhs) {
      ln.second.sort();
      ln.second.unique();
    }
  }
};

struct LocateCoarsenEdges {
  galois::GBigReducible<Updater, EdgeMap>* pg;
  Graph& fg;
  KeyAltTy key;

  LocateCoarsenEdges(Graph& g, KeyAltTy k) : fg(g), key(k) {}

  void operator()(GNode n) {
    auto& attr = fg.getData(n).attr;

    // no such key for n
    auto it = attr.find(key);
    if (it == attr.end()) {
      return;
    }

    auto& v = it->second;
    for (auto e : fg.edges(n)) {
      auto dst      = fg.getEdgeDst(e);
      auto& dstAttr = fg.getData(dst).attr;

      // no such key for dst
      auto dstIt = dstAttr.find(key);
      if (dstIt == dstAttr.end()) {
        continue;
      }

      auto& dstV = dstIt->second;
      if (dstV != v) {
        pg->update(std::make_pair(v, dstV));
      }
    }
  }

  EdgeMap collect() {
    galois::GReducible<EdgeMap, Updater> E;
    pg = &E;
    galois::do_all(fg, *this, galois::steal());
    return E.reduce(Reducer());
  }
};

/*
 * Coarsen fg to cg by key
 * Nodes with no such key will have no representative in the coarsened graph
 */
void coarsen(Graph* fg, Graph* cg, const KeyAltTy key) {
  // collect nodes and edges in the coarsened graph
  LocateCoarsenEdges lce{*fg, key};
  EdgeMap edges = lce.collect();

  // the following may be paralleized

  // create coarsened nodes
  NodeMap nodes;
  for (auto& n : edges) {
    GNode cn = createNode(cg);
    addNode(cg, cn);
    setNodeAttr(cg, cn, key, const_cast<ValAltTy>(n.first.c_str()));
    nodes[n.first] = cn;
  }

  // create coarsened edges
  for (auto& n : edges) {
    for (auto& e : n.second) {
      addEdge(cg, nodes[n.first], nodes[e]);
    }
  }
}
