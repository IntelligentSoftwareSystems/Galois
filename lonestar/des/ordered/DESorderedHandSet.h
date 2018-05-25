/**
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of XYZ License (a copy is located in
 * LICENSE.txt at the top-level directory).
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

#ifndef DES_ORDERED_HAND_SET_H
#define DES_ORDERED_HAND_SET_H

#include "galois/Reduction.h"
#include "galois/Timer.h"
#include "galois/Atomic.h"
#include "galois/PerThreadContainer.h"

#include "galois/substrate/PaddedLock.h"
#include "galois/substrate/CompilerSpecific.h"

#include <deque>
#include <functional>
#include <queue>

#include <cassert>

#include "abstractMain.h"
#include "SimInit.h"
#include "TypeHelper.h"


namespace des_ord {

typedef galois::GAccumulator<size_t> Accumulator_ty;

typedef des::EventRecvTimeLocalTieBrkCmp<TypeHelper<>::Event_ty> Cmp_ty;

typedef galois::PerThreadVector<TypeHelper<>::Event_ty> AddList_ty;

typedef galois::GAtomicPadded<bool> AtomicBool_ty;

static const bool DEBUG = false;

struct SimObjInfo: public TypeHelper<> {


  struct MarkedEvent {
    Event_ty event;
    // mutable AtomicBool_ty flag;
    mutable bool flag;

    explicit MarkedEvent (const Event_ty& _event)
      : event (_event), flag (false)
    {}

    bool isMarked () const { return flag; }

    bool mark () const {
      // return flag.cas (false, true);
      if (flag == false) {
        flag = true;
        return true;

      } else {
        return false;
      }
    }

    void unmark () {
      flag = false;
    }

    operator const Event_ty& () const { return event; }
  };


  typedef galois::substrate::PaddedLock<true> Lock_ty;
  typedef des::AbstractMain<SimInit_ty>::GNode GNode;
  typedef std::set<MarkedEvent, Cmp_ty
    , galois::FixedSizeAllocator<MarkedEvent> > PQ;
  // typedef std::priority_queue<MarkedEvent, std::vector<MarkedEvent>, Cmp_ty::RevCmp> PQ;

  Lock_ty mutex;
  PQ pendingEvents;

  GNode node;
  size_t numInputs;
  size_t numOutputs;
  std::vector<Event_ty> lastInputEvents;

  volatile mutable des::SimTime clock;

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
    assert (Cmp_ty::compare (event, lastInputEvents[dstIn]) >= 0); // event >= last[dstIn]
    lastInputEvents[dstIn] = event;

    mutex.lock ();
      pendingEvents.insert (MarkedEvent(event));
    mutex.unlock ();
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


  bool hasPending () const {
    mutex.lock ();
      bool ret = !pendingEvents.empty ();
    mutex.unlock ();
    return ret;
  }

  MarkedEvent getMin () const {
    mutex.lock ();
       MarkedEvent ret = *pendingEvents.begin ();
    mutex.unlock ();

    return ret;
  }

  bool isMin (const Event_ty& event) const {
    mutex.lock ();
      bool ret = false;
      if (!pendingEvents.empty ()) {
        ret = (event == *pendingEvents.begin ());
      }
    mutex.unlock ();

    return ret;
  }


  Event_ty removeMin () {
    mutex.lock ();
      assert (!pendingEvents.empty ());
      Event_ty event = *pendingEvents.begin ();
      pendingEvents.erase (pendingEvents.begin ());
    mutex.unlock ();

    return event;
  }



  bool isSrc (const Event_ty& event) const {
    return isReady(event)
      && (event.getRecvTime() < des::INFINITY_SIM_TIME ? isMin (event) : true);
  }

  bool canAddMin () const {
    mutex.lock ();
      bool ret = !pendingEvents.empty ()
        && !(pendingEvents.begin ()->isMarked ())
        && isReady (*pendingEvents.begin ())
        && pendingEvents.begin ()->mark ();
    mutex.unlock ();

    return ret;
  }

};


std::vector<SimObjInfo>::iterator
getGlobalMin (std::vector<SimObjInfo>& sobjInfoVec) {

  std::vector<SimObjInfo>::iterator minPos = sobjInfoVec.end ();

  for (std::vector<SimObjInfo>::iterator i = sobjInfoVec.begin ()
      , endi = sobjInfoVec.end (); i != endi; ++i) {

    if (i->hasPending ()) {

      if (minPos == endi) {
        minPos = i;

      } else if (Cmp_ty::compare (i->getMin (), minPos->getMin ()) < 0) {
        minPos = i;
      }
    }
  }

  return minPos;

}

class DESorderedHandSet:
  public des::AbstractMain<TypeHelper<>::SimInit_ty>, public TypeHelper<> {

  struct OpFuncSet {

    typedef int tt_does_not_need_aborts;

    Graph& graph;
    std::vector<SimObjInfo>& sobjInfoVec;
    AddList_ty& newEvents;
    Accumulator_ty& nevents;

    OpFuncSet (
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
    void operator () (Event_ty event, C& lwl) {

      // std::cout << ">>> Processing: " << event.detailedString () << std::endl;

      unsigned epi = 0;
      while (epi < DEFAULT_EPI) {
        ++epi;

        SimObj_ty* recvObj = static_cast<SimObj_ty*> (event.getRecvObj ());
        SimObjInfo& srcInfo = sobjInfoVec[recvObj->getID ()];

        if (DEBUG && !srcInfo.isSrc (event)) {
          abort ();
        }

        nevents += 1;
        newEvents.get ().clear ();

        auto addNewEvents = [this] (const Event_ty& e) {
          newEvents.get().push_back(e);
        };

        recvObj->execEvent (event, graph, srcInfo.node, addNewEvents);

        for (AddList_ty::local_iterator a = newEvents.get ().begin ()
            , enda = newEvents.get ().end (); a != enda; ++a) {

          SimObjInfo& sinfo = sobjInfoVec[a->getRecvObj ()->getID ()];

          sinfo.recv (*a);


          if (sinfo.canAddMin ()) {

            assert (sinfo.getMin ().isMarked ());
            lwl.push (sinfo.getMin ());

            // std::cout << "### Adding: " << static_cast<const Event_ty&> (sinfo.getMin ()).detailedString () << std::endl;
          }

        }


        if (DEBUG && !srcInfo.isSrc (event)) { abort (); }
        srcInfo.removeMin ();

        if (srcInfo.canAddMin ()) {

          assert (srcInfo.isSrc (srcInfo.getMin ()));
          assert (srcInfo.getMin ().isMarked ());

          event = srcInfo.getMin ();
          assert (srcInfo.isSrc (event));

          if (epi == DEFAULT_EPI) { lwl.push (event); }
          // lwl.push (srcInfo.getMin ());
          // std::cout << "%%% Adding: " << static_cast<const Event_ty&> (srcInfo.getMin ()).detailedString () << std::endl;
        } else {
          break;
        }

      } // end while

      SimObjInfo& srcInfo = sobjInfoVec[event.getRecvObj ()->getID ()];
      if (srcInfo.canAddMin ()) {
        assert (srcInfo.getMin ().isMarked ());
        lwl.push (srcInfo.getMin ());
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

      SimObj_ty* so = static_cast<SimObj_ty*> (graph.getData (*n, galois::MethodFlag::UNPROTECTED));
      sobjInfoVec[so->getID ()] = SimObjInfo (*n, so);
    }
  }


  virtual void runLoop (const SimInit_ty& simInit, Graph& graph) {

    std::vector<Event_ty> initWL;

    for (std::vector<Event_ty>::const_iterator i = simInit.getInitEvents ().begin ()
        , endi = simInit.getInitEvents ().end (); i != endi; ++i) {

      SimObj_ty* recvObj = static_cast<SimObj_ty*> (i->getRecvObj ());
      SimObjInfo& sinfo = sobjInfoVec[recvObj->getID ()];
      sinfo.recv (*i);

    }


    for (std::vector<Event_ty>::const_iterator i = simInit.getInitEvents ().begin (), endi =
          simInit.getInitEvents ().end (); i != endi; ++i) {

        BaseSimObj_ty* recvObj = i->getRecvObj ();
        SimObjInfo& sinfo = sobjInfoVec[recvObj->getID ()];

        if (sinfo.canAddMin ()) {

          initWL.push_back (sinfo.getMin ());
          // std::cout << "Initial source found: " << Event_ty (sinfo.getMin ()).detailedString () << std::endl;
        }

      }

    std::cout << "Number of initial sources: " << initWL.size () << std::endl;

    AddList_ty newEvents;
    Accumulator_ty nevents;
    size_t round = 0;

    while (true) {
      ++round;

      typedef galois::worklists::dChunkedFIFO<DEFAULT_CHUNK_SIZE> WL_ty;

      galois::for_each(initWL.begin (), initWL.end (),
                       OpFuncSet (graph, sobjInfoVec, newEvents,  nevents),
                       galois::wl<WL_ty>());

      initWL.clear ();

      std::vector<SimObjInfo>::iterator p = getGlobalMin (sobjInfoVec);

      if (p == sobjInfoVec.end ()) {
        break;

      } else {
        initWL.push_back (p->getMin ());
      }

    }

    std::cout << "Number of rounds = " << round << std::endl;
    std::cout << "Number of events processed= " <<
      nevents.reduce () << std::endl;
  }

};


} // namespace des_ord
#endif // DES_ORDERED_HAND_SET_H
