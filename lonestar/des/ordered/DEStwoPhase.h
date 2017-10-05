/** DES two phase ordered version -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galois is a framework to exploit
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


#include "galois/Reduction.h"
#include "galois/Timer.h"
#include "galois/Atomic.h"
#include "galois/Galois.h"

#include "galois/runtime/KDGtwoPhase.h"
#include "galois/substrate/PaddedLock.h"
#include "galois/substrate/CompilerSpecific.h"

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

struct SimObjInfo;
typedef std::vector<SimObjInfo> VecSobjInfo;


struct SimObjInfo: public TypeHelper<> {

  typedef des::AbstractMain<SimInit_ty>::GNode GNode;
  GNode node;
  size_t numInputs;
  size_t numOutputs;
  std::vector<Event_ty> lastInputEvents;
  mutable volatile des::SimTime clock;

  SimObjInfo () {}

  SimObjInfo (GNode node, SimObj_ty* sobj): node (node) {
    SimGate_ty* sg = static_cast<SimGate_ty*> (sobj);
    assert (sg != NULL);

    numInputs = sg->getImpl ().getNumInputs ();
    numOutputs = sg->getImpl ().getNumOutputs ();

    lastInputEvents.resize (numInputs, sobj->makeZeroEvent ());

    clock = 0;
  }


  void recv (const Event_ty& event) {

    SimGate_ty* dstGate = static_cast<SimGate_ty*> (event.getRecvObj ());
    assert (dstGate != NULL);

    const std::string& outNet = event.getAction ().getNetName ();
    size_t dstIn = dstGate->getImpl ().getInputIndex (outNet); // get the input index of the net to which my output is connected

    assert (dstIn < lastInputEvents.size ());
    lastInputEvents[dstIn] = event;
  }

  bool isReady (const Event_ty& event) const {
    // not ready if event has a timestamp greater than the latest event received
    // on any input.
    // an input with INFINITY_SIM_TIME is dead and will not receive more non-null events
    // in the future
    bool notReady = false;

    if (event.getRecvTime () < clock) {
      return true;

    } else {

      des::SimTime new_clk = 2 * des::INFINITY_SIM_TIME;
      for (std::vector<Event_ty>::const_iterator e = lastInputEvents.begin ()
          , ende = lastInputEvents.end (); e != ende; ++e) {

        if ((e->getRecvTime () < des::INFINITY_SIM_TIME) &&
            (Cmp_ty::compare (event, *e) > 0)) {
          notReady = true;
          // break;
        }

        if (e->getRecvTime () < des::INFINITY_SIM_TIME) {
          new_clk = std::min (new_clk, e->getRecvTime ());
        }
      }

      this->clock = new_clk;
    }

    return !notReady;
  }

};


class DEStwoPhase:
  public des::AbstractMain<TypeHelper<>::SimInit_ty>, public TypeHelper<> {

  struct NhoodVisitor {
    typedef int tt_has_fixed_neighborhood;

    Graph& graph;
    VecSobjInfo& sobjInfoVec;

    static const int CHUNK_SIZE = 16;

    NhoodVisitor (Graph& graph, VecSobjInfo& sobjInfoVec)
      : graph (graph), sobjInfoVec (sobjInfoVec)
    {}

    template <typename C>
    void operator () (const Event_ty& event, C&) const {
      SimObjInfo& recvInfo = sobjInfoVec[event.getRecvObj ()->getID ()];
      if (recvInfo.isReady (event)) {
        graph.getData (recvInfo.node, galois::MethodFlag::WRITE);

      } else {
        galois::runtime::signalConflict ();
      }
    }
  };

  // struct ReadyTest {
    // VecSobjInfo& sobjInfoVec;
//
    // explicit ReadyTest (VecSobjInfo& sobjInfoVec): sobjInfoVec (sobjInfoVec) {}
//
    // bool operator () (const Event_ty& event) const {
      // size_t id = event.getRecvObj ()->getID ();
      // SimObjInfo& sinfo = sobjInfoVec[id];
      // return sinfo.isReady (event);
    // }
  // };


  struct OpFunc {

    static const unsigned CHUNK_SIZE = 16;


    Graph& graph;
    std::vector<SimObjInfo>& sobjInfoVec;
    AddList_ty& newEvents;
    Accumulator_ty& nevents;

    OpFunc (
        Graph& graph,
        std::vector<SimObjInfo>& sobjInfoVec,
        AddList_ty& newEvents,
        Accumulator_ty& nevents)
      :
        graph (graph),
        sobjInfoVec (sobjInfoVec),
        newEvents (newEvents),
        nevents (nevents)
    {}

    template <typename C>
    void operator () (const Event_ty& event, C& lwl) {

      // std::cout << ">>> Processing: " << event.detailedString () << std::endl;

      // TODO: needs a PQ with remove operation to work correctly
      // assert (ReadyTest (sobjInfoVec) (event));

      SimObj_ty* recvObj = static_cast<SimObj_ty*> (event.getRecvObj ());
      SimObjInfo& recvInfo = sobjInfoVec[recvObj->getID ()];

      assert (recvInfo.isReady (event));

      nevents += 1;
      // newEvents.get ().clear ();


      auto addNewFunc = [this, &lwl] (const Event_ty& e) {

        SimObjInfo& sinfo = sobjInfoVec[e.getRecvObj()->getID ()];
        sinfo.recv (e);
        lwl.push (e);
      };

      recvObj->execEvent (event, graph, recvInfo.node, addNewFunc);

      // for (AddList_ty::local_iterator a = newEvents.get ().begin ()
          // , enda = newEvents.get ().end (); a != enda; ++a) {
//
        // SimObjInfo& sinfo = sobjInfoVec[a->getRecvObj()->getID ()];
        // sinfo.recv (*a);
        // lwl.push (*a);
//
        // // std::cout << "### Adding: " << a->detailedString () << std::endl;
      // }

    }

  };

  std::vector<SimObjInfo> sobjInfoVec;

protected:
  virtual std::string getVersion () const { return "Handwritten Ordered ODG, no barrier"; }

  virtual void initRemaining (const SimInit_ty& simInit, Graph& graph) {
    sobjInfoVec.clear ();
    sobjInfoVec.resize (graph.size ());

    for (Graph::iterator n = graph.begin ()
        , endn = graph.end (); n != endn; ++n) {

      SimObj_ty* so = static_cast<SimObj_ty*> (graph.getData (*n, galois::MethodFlag::UNPROTECTED));
      sobjInfoVec[so->getID ()] = SimObjInfo (*n, so);
    }
  }


  virtual void runLoop (const SimInit_ty& simInit, Graph& graph) {

    for (std::vector<Event_ty>::const_iterator e = simInit.getInitEvents ().begin ()
        , ende = simInit.getInitEvents ().end (); e != ende; ++e) {

      SimObjInfo& sinfo = sobjInfoVec[e->getRecvObj ()->getID ()];
      sinfo.recv (*e);
    }

    AddList_ty newEvents;
    Accumulator_ty nevents;

    galois::runtime::for_each_ordered_ikdg (
        galois::runtime::makeStandardRange(
          simInit.getInitEvents ().begin (), simInit.getInitEvents ().end ()),
        Cmp_ty (),
        NhoodVisitor (graph, sobjInfoVec),
        OpFunc (graph, sobjInfoVec, newEvents, nevents),
        std::make_tuple (galois::loopname("des-ikdg")));
        // ReadyTest (sobjInfoVec));

    std::cout << "Number of events processed= " <<
      nevents.reduce () << std::endl;
  }
};


}

#endif // DES_ORDERED_H
