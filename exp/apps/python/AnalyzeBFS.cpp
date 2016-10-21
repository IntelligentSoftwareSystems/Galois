#include "AnalyzeBFS.h"
#include "Galois/Statistic.h"

#include <limits>
#include <iostream>

static const size_t DIST_INFINITY = std::numeric_limits<size_t>::max() - 1;

struct BFS {
  Graph& g;
  BFS(Graph& g): g(g) {}

  // use vInt for distance
  void operator()(GNode n, Galois::UserContext<GNode>& ctx) {
    auto newDist = g.getData(n).vInt + 1;
    for(auto e: g.edges(n)) {
      auto dst = g.getEdgeDst(e);
      auto& dstDist = g.getData(dst).vInt;
      if(dstDist > newDist) {
        dstDist = newDist;
        ctx.push(dst);
      }
    }
  }
};

void analyzeBFS(Graph *g, GNode src, GNode report) {
  Galois::StatManager statManager;

  Galois::StatTimer T;
  T.start();

  Galois::do_all_local(
    *g, 
    [=] (GNode n) 
      {
        auto& data = (*g).getData(n); 
        data.mode = 0; 
        data.vInt = DIST_INFINITY;
      }
    );

  g->getData(src).vInt = 0;
  Galois::for_each(src, BFS{*g});

  Galois::do_all_local(
    *g, 
    [=] (GNode n) 
      {
        auto& data = (*g).getData(n); 
        data.attr["dist"] = std::to_string(data.vInt);
      }
    );

  T.stop();

  std::cout << "distance of reported node: " << g->getData(report).attr["dist"] << std::endl;
}

