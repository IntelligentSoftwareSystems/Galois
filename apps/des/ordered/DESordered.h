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

typedef GaloisRuntime::PerThreadVector<TypeHelper::Event_ty> AddList_ty;

struct SimObjInfo;
typedef std::vector<SimObjInfo> VecSobjInfo;


struct SimObjInfo: public TypeHelper {

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


class DESordered: 
  public des::AbstractMain<TypeHelper::SimInit_ty>, public TypeHelper {

  struct NhoodVisitor {
    typedef int tt_has_fixed_neighborhood;

    Graph& graph;
    VecSobjInfo& sobjInfoVec;

    NhoodVisitor (Graph& graph, VecSobjInfo& sobjInfoVec)
      : graph (graph), sobjInfoVec (sobjInfoVec) 
    {}
    
    template <typename C>
    void operator () (const Event_ty& event, C&) const {
      SimObjInfo& recvInfo = sobjInfoVec[event.getRecvObj ()->getID ()];
      graph.getData (recvInfo.node, Galois::CHECK_CONFLICT);
    }
  };

  struct ReadyTest {
    VecSobjInfo& sobjInfoVec;

    ReadyTest (VecSobjInfo& sobjInfoVec): sobjInfoVec (sobjInfoVec) {}

    bool operator () (const Event_ty& event) const {
      SimObjInfo& sinfo = sobjInfoVec[event.getRecvObj ()->getID ()];
      return sinfo.isReady (event);
    }
  };


  struct OpFunc {
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
      assert (ReadyTest (sobjInfoVec) (event));

      SimObj_ty* recvObj = static_cast<SimObj_ty*> (event.getRecvObj ());
      SimObjInfo& recvInfo = sobjInfoVec[recvObj->getID ()];

      nevents += 1;
      newEvents.get ().clear ();

      recvObj->execEvent (event, graph, recvInfo.node, newEvents.get ());

      for (AddList_ty::local_iterator a = newEvents.get ().begin ()
          , enda = newEvents.get ().end (); a != enda; ++a) {

        SimObjInfo& sinfo = sobjInfoVec[a->getRecvObj()->getID ()];
        sinfo.recv (*a);
        lwl.push (*a);

        // std::cout << "### Adding: " << a->detailedString () << std::endl;
      }

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

      SimObj_ty* so = static_cast<SimObj_ty*> (graph.getData (*n, Galois::NONE));
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

    Galois::for_each_ordered (
        simInit.getInitEvents ().begin (), simInit.getInitEvents ().end (),
        Cmp_ty (), 
        NhoodVisitor (graph, sobjInfoVec),
        OpFunc (graph, sobjInfoVec, newEvents, nevents),
        ReadyTest (sobjInfoVec));

    std::cout << "Number of events processed= " << 
      nevents.reduce () << std::endl;
  }
};


}

#endif // DES_ORDERED_H
