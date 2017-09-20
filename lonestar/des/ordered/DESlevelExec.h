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

#ifndef DES_ORDERED_LEVEL_EXEC_H
#define DES_ORDERED_LEVEL_EXEC_H


#include "galois/Accumulator.h"
#include "galois/Timer.h"
#include "galois/Atomic.h"
#include "galois/Galois.h"
#include "galois/PerThreadContainer.h"

#include "galois/Substrate/PaddedLock.h"
#include "galois/Substrate/CompilerSpecific.h"
#include "galois/runtime/LevelExecutor.h"
#include "galois/runtime/Range.h"

#include "abstractMain.h"
#include "SimInit.h"
#include "TypeHelper.h"

#include <deque>
#include <functional>
#include <queue>

#include <cassert>


namespace des_ord {

typedef galois::GAccumulator<size_t> Accumulator_ty;

typedef des::EventRecvTimeLocalTieBrkCmp<TypeHelper::Event_ty> Cmp_ty;

typedef galois::PerThreadVector<TypeHelper::Event_ty> AddList_ty;



class DESlevelExec: 
  public des::AbstractMain<TypeHelper::SimInit_ty>, public TypeHelper {

  using VecGNode = std::vector<GNode>;
  using AddList_ty =  galois::PerThreadVector<TypeHelper::Event_ty>;

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
      graph.getData (n, galois::MethodFlag::WRITE);
    }
  };


  struct OpFunc {

    static const unsigned CHUNK_SIZE = 16;

    Graph& graph;
    VecGNode& nodes;
    AddList_ty& newEvents;
    Accumulator_ty& nevents;

    OpFunc (
        Graph& graph,
        VecGNode& nodes,
        AddList_ty& newEvents,
        Accumulator_ty& nevents)
      :
        graph (graph),
        nodes (nodes),
        newEvents (newEvents),
        nevents (nevents)
    {}

    template <typename C>
    void operator () (const Event_ty& event, C& ctx) const {

      // std::cout << ">>> Processing: " << event.detailedString () << std::endl;

      SimObj_ty* recvObj = static_cast<SimObj_ty*> (event.getRecvObj ());
      GNode n = nodes[recvObj->getID ()];

      newEvents.get ().clear ();

      recvObj->execEvent (event, graph, n, newEvents.get ());

      for (auto a = newEvents.get ().begin ()
          , enda = newEvents.get ().end (); a != enda; ++a) {
        ctx.push (*a);
        // std::cout << "### Adding: " << a->detailedString () << std::endl;
      }

      nevents += 1;
    }

  };


  struct GetRecvTime {
    des::SimTime operator () (const Event_ty& e) const {
      return e.getRecvTime ();
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

    AddList_ty newEvents;
    Accumulator_ty nevents;

    // galois::for_each_ordered (
    galois::runtime::for_each_ordered_level (
        galois::runtime::makeStandardRange (simInit.getInitEvents ().begin (), simInit.getInitEvents ().end ()),
        GetRecvTime (), std::less<des::SimTime> (), 
        NhoodVisitor (graph, nodes),
        OpFunc (graph, nodes, newEvents, nevents));

    std::cout << "Number of events processed= " << 
      nevents.reduce () << std::endl;
  }
};


} // end namespace des_ord

#endif // DES_ORDERED_LEVEL_EXEC_H
