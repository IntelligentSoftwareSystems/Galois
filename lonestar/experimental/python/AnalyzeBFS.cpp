#include "AnalyzeBFS.h"
#include "Auxiliary.h"
#include "Galois/Statistic.h"

#include <limits>
#include <iostream>

struct BFS {
  Graph& g;
  BFS(Graph& g): g(g) {}

  // use vInt for distance
  void operator()(GNode n, Galois::UserContext<GNode>& ctx) {
    auto newDist = g.getData(n).ID.vInt + 1;
    for(auto e: g.edges(n)) {
      auto dst = g.getEdgeDst(e);
      auto& dstDist = g.getData(dst).ID.vInt;
      if(dstDist > newDist) {
        dstDist = newDist;
        ctx.push(dst);
      }
    }
  }
};

void analyzeBFS(Graph *g, GNode src, const ValAltTy result) {
//  Galois::StatManager statManager;

//  Galois::StatTimer T;
//  T.start();

  Galois::do_all_local(
    *g, 
    [=] (GNode n) 
      {
        auto& data = (*g).getData(n); 
        data.ID.vInt = DIST_INFINITY;
      }
    );

  g->getData(src).ID.vInt = 0;
  Galois::for_each(src, BFS{*g});

//  T.stop();

  Galois::do_all_local(
    *g, 
    [=] (GNode n) 
      {
        auto& data = (*g).getData(n); 
        data.attr[result] = (DIST_INFINITY == data.ID.vInt) ? "INFINITY" : std::to_string(data.ID.vInt);
      }
    );
}
