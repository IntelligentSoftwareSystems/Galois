/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
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

#ifndef WAVEFRONT_BULK_SYNC_EDGE_H
#define WAVEFRONT_BULK_SYNC_EDGE_H

#include "bfs.h"

#include "galois/PerThreadContainer.h"

#include "galois/runtime/ParallelWorkInline_Exp.h"
#include "galois/runtime/CoupledExecutor.h"

typedef uint32_t ND_ty;
class WavefrontBulkSyncEdge : public BFS<ND_ty> {

protected:
  virtual const std::string getVersion() const {
    return "Wavefront using Bulk Synchronous Executor, Edge-based operator";
  }

  struct Update {
    GNode node;
    ND_ty level;

    Update(GNode node, ND_ty level) : node(node), level(level) {}
  };

  typedef galois::PerThreadVector<Update> WL_ty;

  struct PftchFunc {
    typedef int tt_does_not_need_aborts;
    typedef char tt_does_not_need_push;

    Graph& graph;
    WL_ty& nextWL;

    explicit PftchFunc(Graph& graph, WL_ty& nextWL)
        : graph(graph), nextWL(nextWL) {}

    template <typename K>
    GALOIS_ATTRIBUTE_PROF_NOINLINE void operator()(const Update& up,
                                                   K pftch_kind) {}
  };

  struct OpFunc {
    typedef int tt_does_not_need_aborts;
    // typedef char tt_does_not_need_push;

    Graph& graph;

#if 1
    explicit OpFunc(Graph& graph) : graph(graph) {}
#else
    WL_ty& nextWL;
    explicit OpFunc(Graph& graph, WL_ty& nextWL)
        : graph(graph), nextWL(nextWL) {}

    GALOIS_ATTRIBUTE_PROF_NOINLINE void operator()(const Update& up) {
      (*this)(up, nextWL.get());
    }
#endif

    template <typename C>
    GALOIS_ATTRIBUTE_PROF_NOINLINE void operator()(const Update& up, C& wl) {
      bool updated = false;
      if (graph.getData(up.node, galois::MethodFlag::UNPROTECTED) ==
          BFS_LEVEL_INFINITY) {
        graph.getData(up.node, galois::MethodFlag::UNPROTECTED) = up.level;
        updated                                                 = true;
      }

      if (updated) {
        ND_ty dstLevel = up.level + 1;

        graph.mapOutEdges(
            up.node,
            [dstLevel, &wl](GNode dst) {
              wl.push(dst, dstLevel);
              // wl.emplace(dst, dstLevel);
            },
            galois::MethodFlag::UNPROTECTED);
      }
    }
  };

  virtual size_t runBFS(Graph& graph, GNode& startNode) {

    ParCounter numAdds;

#if 1

    Update init[1] = {Update(startNode, 0)};
    // galois::for_each ( &init[0], &init[1], OpFunc (graph),
    // galois::wl<galois::worklists::BulkSynchronousInline <> > ());
    galois::runtime::do_all_coupled_bs(
        galois::runtime::makeStandardRange(&init[0], &init[1]), OpFunc(graph),
        "WavefrontBulkSyncEdge");
    // galois::runtime::for_each_coupled_bs (galois::runtime::makeStandardRange
    // (&init[0], &init[1]), OpFunc (graph), "WavefrontBulkSyncEdge");

#else

    WL_ty* currWL = new WL_ty();
    WL_ty* nextWL = new WL_ty();

    nextWL->get().emplace_back(startNode, 0);

    while (!nextWL->empty_all()) {

      std::swap(currWL, nextWL);
      nextWL->clear_all();

      galois::do_all_choice(*currWL, OpFunc(graph, *nextWL));
    }

    delete currWL;
    delete nextWL;
  }
#endif

    return numAdds.reduce();
  }
};

#endif // WAVEFRONT_BULK_SYNC_EDGE_H
