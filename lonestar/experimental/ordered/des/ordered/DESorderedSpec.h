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

#ifndef DES_ORDERED_H
#define DES_ORDERED_H

#include "galois/Reduction.h"
#include "galois/Timer.h"
#include "galois/Atomic.h"
#include "galois/Galois.h"

#include "galois/substrate/PaddedLock.h"
#include "galois/substrate/CompilerSpecific.h"
#include "galois/runtime/ROBexecutor.h"
#include "galois/runtime/OrderedSpeculation.h"

#include "abstractMain.h"
#include "SimInit.h"
#include "TypeHelper.h"

#include <deque>
#include <functional>
#include <queue>

#include <cassert>

namespace des_ord {

typedef galois::GAccumulator<size_t> Accumulator_ty;

typedef des::EventRecvTimeLocalTieBrkCmp<TypeHelper<>::Event_ty> Cmp_ty;

typedef galois::PerThreadVector<TypeHelper<>::Event_ty> AddList_ty;

class DESorderedSpec : public des::AbstractMain<TypeHelper<>::SimInit_ty>,
                       public TypeHelper<> {

  using VecGNode = std::vector<GNode>;

  VecGNode nodes;

  struct NhoodVisitor {
    typedef int tt_has_fixed_neighborhood;

    static const unsigned CHUNK_SIZE = DEFAULT_CHUNK_SIZE;

    Graph& graph;
    VecGNode& nodes;

    NhoodVisitor(Graph& graph, VecGNode& nodes) : graph(graph), nodes(nodes) {}

    template <typename C>
    void operator()(const Event_ty& event, C& ctx) const {
      GNode n = nodes[event.getRecvObj()->getID()];
      graph.getData(n, galois::MethodFlag::WRITE);
    }
  };

  struct OpFunc {

    static const unsigned CHUNK_SIZE = DEFAULT_CHUNK_SIZE;

    Graph& graph;
    VecGNode& nodes;
    Accumulator_ty& nevents;

    OpFunc(Graph& graph, VecGNode& nodes, Accumulator_ty& nevents)
        : graph(graph), nodes(nodes), nevents(nevents) {}

    template <typename C>
    void operator()(const Event_ty& event, C& ctx) {

      // std::cout << ">>> Processing: " << event.detailedString () <<
      // std::endl;

      // TODO: needs a PQ with remove operation to work correctly
      // assert (ReadyTest (sobjInfoVec) (event));

      SimObj_ty* recvObj = static_cast<SimObj_ty*>(event.getRecvObj());
      GNode n            = nodes[recvObj->getID()];

      size_t stateSize = recvObj->getStateSize();

      galois::runtime::FixedSizeHeap heap(stateSize);
      void* const p = heap.allocate(stateSize);

      recvObj->copyState(p, stateSize);

      auto f = [recvObj, p, stateSize, heap](void) mutable {
        recvObj->restoreState(p, stateSize);
        heap.deallocate(p);
      };

      ctx.addUndoAction(f);

      auto addNewFunc = [&ctx](const Event_ty& e) { ctx.push(e); };

      // FIXME: newEvents needs to be iteration local
      recvObj->execEvent(event, graph, n, addNewFunc);

      // for (auto a = newEvents.begin ()
      // , enda = newEvents.end (); a != enda; ++a) {
      // ctx.push (*a);
      // // std::cout << "### Adding: " << a->detailedString () << std::endl;
      // }

      auto inc = [this, p, heap](void) mutable {
        nevents += 1;
        heap.deallocate(p);
      };

      ctx.addCommitAction(inc);
    }
  };

protected:
  virtual std::string getVersion() const {
    return "Handwritten Ordered ODG, no barrier";
  }

  virtual void initRemaining(const SimInit_ty& simInit, Graph& graph) {
    nodes.clear();
    nodes.resize(graph.size());

    for (Graph::iterator n = graph.begin(), endn = graph.end(); n != endn;
         ++n) {

      BaseSimObj_ty* so  = graph.getData(*n, galois::MethodFlag::UNPROTECTED);
      nodes[so->getID()] = *n;
    }
  }

  virtual void runLoop(const SimInit_ty& simInit, Graph& graph) {

    Accumulator_ty nevents;

    // galois::for_each_ordered (
    galois::runtime::for_each_ordered_spec(
        galois::runtime::makeStandardRange(simInit.getInitEvents().begin(),
                                           simInit.getInitEvents().end()),
        Cmp_ty(), NhoodVisitor(graph, nodes), OpFunc(graph, nodes, nevents),
        std::make_tuple(galois::loopname("des_ordered_spec")));

    std::cout << "Number of events processed= " << nevents.reduce()
              << std::endl;
  }
};

} // end namespace des_ord

#endif // DES_ORDERED_H
