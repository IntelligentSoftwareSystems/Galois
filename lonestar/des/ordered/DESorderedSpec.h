/** DES ordered version -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * Galois is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Galois.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @section Copyright
 *
 * Copyright (C) 2015, The University of Texas at Austin. All rights
 * reserved.
 *
 * @author M. Amber Hassaan <ahassaan@ices.utexas.edu>
 */

#ifndef DES_ORDERED_H
#define DES_ORDERED_H


#include "galois/Accumulator.h"
#include "galois/Timer.h"
#include "galois/Atomic.h"
#include "galois/Galois.h"

#include "galois/Substrate/PaddedLock.h"
#include "galois/Substrate/CompilerSpecific.h"
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



class DESorderedSpec: 
  public des::AbstractMain<TypeHelper<>::SimInit_ty>, public TypeHelper<> {

  using VecGNode = std::vector<GNode>;

  VecGNode nodes;

  struct NhoodVisitor {
    typedef int tt_has_fixed_neighborhood;

    static const unsigned CHUNK_SIZE = DEFAULT_CHUNK_SIZE;

    Graph& graph;
    VecGNode& nodes;

    NhoodVisitor (Graph& graph, VecGNode& nodes) 
      : graph (graph), nodes (nodes) 
    {}
    
    template <typename C>
    void operator () (const Event_ty& event, C& ctx) const {
      GNode n = nodes[event.getRecvObj ()->getID ()];
      graph.getData (n, galois::MethodFlag::WRITE);
    }
  };


  struct OpFunc {

    static const unsigned CHUNK_SIZE = DEFAULT_CHUNK_SIZE;

    Graph& graph;
    VecGNode& nodes;
    Accumulator_ty& nevents;

    OpFunc (
        Graph& graph,
        VecGNode& nodes,
        Accumulator_ty& nevents)
      :
        graph (graph),
        nodes (nodes),
        nevents (nevents)
    {}

    template <typename C>
    void operator () (const Event_ty& event, C& ctx) {

      // std::cout << ">>> Processing: " << event.detailedString () << std::endl;

      // TODO: needs a PQ with remove operation to work correctly
      // assert (ReadyTest (sobjInfoVec) (event));

      SimObj_ty* recvObj = static_cast<SimObj_ty*> (event.getRecvObj ());
      GNode n = nodes[recvObj->getID ()];

      size_t stateSize = recvObj->getStateSize ();

      galois::runtime::FixedSizeHeap heap (stateSize);
      void* const p = heap.allocate (stateSize);

      recvObj->copyState (p, stateSize);

      auto f = [recvObj, p, stateSize, heap] (void) mutable {
        recvObj->restoreState (p, stateSize);
        heap.deallocate (p);
      };

      ctx.addUndoAction (f);

      auto addNewFunc = [&ctx] (const Event_ty& e) {
        ctx.push (e);
      };

      // FIXME: newEvents needs to be iteration local
      recvObj->execEvent (event, graph, n, addNewFunc);

      // for (auto a = newEvents.begin ()
          // , enda = newEvents.end (); a != enda; ++a) {
        // ctx.push (*a);
        // // std::cout << "### Adding: " << a->detailedString () << std::endl;
      // }

      auto inc = [this, p, heap] (void) mutable {
        nevents += 1;
        heap.deallocate (p);
      };

      ctx.addCommitAction (inc);
    }

  };

protected:
  virtual std::string getVersion () const { return "Handwritten Ordered ODG, no barrier"; }

  virtual void initRemaining (const SimInit_ty& simInit, Graph& graph) {
    nodes.clear ();
    nodes.resize (graph.size ());

    for (Graph::iterator n = graph.begin ()
        , endn = graph.end (); n != endn; ++n) {

      BaseSimObj_ty* so = graph.getData (*n, galois::MethodFlag::UNPROTECTED);
      nodes[so->getID ()] = *n;
    }
  }

  virtual void runLoop (const SimInit_ty& simInit, Graph& graph) {

    Accumulator_ty nevents;

    // galois::for_each_ordered (
    galois::runtime::for_each_ordered_spec (
        galois::runtime::makeStandardRange(
          simInit.getInitEvents ().begin (), simInit.getInitEvents ().end ()),
        Cmp_ty (), 
        NhoodVisitor (graph, nodes),
        OpFunc (graph, nodes, nevents),
        std::make_tuple (
          galois::loopname("des_ordered_spec")));

    std::cout << "Number of events processed= " << 
      nevents.reduce () << std::endl;
  }
};


} // end namespace des_ord

#endif // DES_ORDERED_H
