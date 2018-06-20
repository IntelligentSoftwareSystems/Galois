/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 */

#include "AnalyzeBFS.h"
#include "Auxiliary.h"
#include "galois/Timer.h"

#include <limits>
#include <iostream>

struct BFS {
  Graph& g;
  BFS(Graph& g) : g(g) {}

  // use vInt for distance
  void operator()(GNode n, galois::UserContext<GNode>& ctx) {
    auto newDist = g.getData(n).ID.vInt + 1;
    for (auto e : g.edges(n)) {
      auto dst      = g.getEdgeDst(e);
      auto& dstDist = g.getData(dst).ID.vInt;
      if (dstDist > newDist) {
        dstDist = newDist;
        ctx.push(dst);
      }
    }
  }
};

void analyzeBFS(Graph* g, GNode src, const ValAltTy result) {
  //  galois::StatManager statManager;

  //  galois::StatTimer T;
  //  T.start();

  galois::do_all(*g, [=](GNode n) {
    auto& data   = (*g).getData(n);
    data.ID.vInt = DIST_INFINITY;
  });

  g->getData(src).ID.vInt = 0;
  galois::for_each(src, BFS{*g});

  //  T.stop();

  galois::do_all(*g, [=](GNode n) {
    auto& data        = (*g).getData(n);
    data.attr[result] = (DIST_INFINITY == data.ID.vInt)
                            ? "INFINITY"
                            : std::to_string(data.ID.vInt);
  });
}
