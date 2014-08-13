/** DES ordered version -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
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
 *
 * @author M. Amber Hassaan <ahassaan@ices.utexas.edu>
 */

#ifndef DES_ORDERED_H
#define DES_ORDERED_H


#include "Galois/Accumulator.h"
#include "Galois/Timer.h"
#include "Galois/Atomic.h"
#include "Galois/Galois.h"

#include "Galois/Runtime/ll/PaddedLock.h"
#include "Galois/Runtime/ll/CompilerSpecific.h"
#include "Galois/Runtime/ROBexecutor.h"

#include "abstractMain.h"
#include "SimInit.h"
#include "TypeHelper.h"

#include <deque>
#include <functional>
#include <queue>

#include <cassert>


namespace des_ord {

typedef Galois::GAccumulator<size_t> Accumulator_ty;

typedef des::EventRecvTimeLocalTieBrkCmp<TypeHelper::Event_ty> Cmp_ty;

typedef Galois::Runtime::PerThreadVector<TypeHelper::Event_ty> AddList_ty;



class DESorderedSpec: 
  public des::AbstractMain<TypeHelper::SimInit_ty>, public TypeHelper {

  using VecGNode = std::vector<GNode>;

  VecGNode nodes;

  struct NhoodVisitor {
    typedef int tt_has_fixed_neighborhood;

    Graph& graph;
    VecGNode& nodes;

    NhoodVisitor (Graph& graph, VecGNode& nodes) 
      : graph (graph), nodes (nodes) 
    {}
    
    template <typename C>
    void operator () (const Event_ty& event, C& ctx) const {
      GNode n = nodes[event.getRecvObj ()->getID ()];
      graph.getData (n, Galois::MethodFlag::CHECK_CONFLICT);
    }
  };


  struct OpFunc {
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
      auto alloc = ctx.getPerIterAlloc ();

      char* const p = alloc.allocate (stateSize);

      recvObj->copyState (p, stateSize);

      auto f = [recvObj, p, stateSize] (void) {
        recvObj->restoreState (p, stateSize);
      };

      ctx.addUndoAction (f);

      using AddList_ty = std::vector<Event_ty, Galois::PerIterAllocTy::rebind<Event_ty>::other>;

      AddList_ty newEvents (alloc);

      // FIXME: newEvents needs to be iteration local
      recvObj->execEvent (event, graph, n, newEvents);

      for (auto a = newEvents.begin ()
          , enda = newEvents.end (); a != enda; ++a) {
        ctx.push (*a);
        // std::cout << "### Adding: " << a->detailedString () << std::endl;
      }

      auto inc = [this] (void) {
        nevents += 1;
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

      BaseSimObj_ty* so = graph.getData (*n, Galois::MethodFlag::NONE);
      nodes[so->getID ()] = *n;
    }
  }

  virtual void runLoop (const SimInit_ty& simInit, Graph& graph) {

    Accumulator_ty nevents;

    // Galois::for_each_ordered (
    Galois::Runtime::for_each_ordered_rob (
        Galois::Runtime::makeStandardRange(
          simInit.getInitEvents ().begin (), simInit.getInitEvents ().end ()),
        Cmp_ty (), 
        NhoodVisitor (graph, nodes),
        OpFunc (graph, nodes, nevents));

    std::cout << "Number of events processed= " << 
      nevents.reduce () << std::endl;
  }
};


} // end namespace des_ord

#endif // DES_ORDERED_H
