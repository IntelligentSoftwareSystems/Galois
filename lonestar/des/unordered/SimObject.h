/** SimObject: the abstract interface to be implemented by any simulation object -*- C++ -*-
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


#ifndef SIMOBJECT_H_
#define SIMOBJECT_H_

#include <vector>
#include <queue>

#include <cassert>


#include "comDefs.h"

#include "BaseSimObject.h"
#include "Event.h"

#include "Galois/PriorityQueue.h"
#include "Galois/gIO.h"

//TODO: modeling one output for now. Need to extend for multiple outputs
/**
 * @section Description
 *
 * The Class SimObject represents an abstract simulation object (processing station). A simulation application
 * would inherit from this class.
 */

namespace des_unord {

template <typename Event_tp>
class SimObject: public des::BaseSimObject<Event_tp> {

  typedef typename des::BaseSimObject<Event_tp> Base;
  typedef Event_tp Event_ty;


protected:

  struct SendWrapperImpl: public Base::SendWrapper {
    virtual void send (Base* rs, const Event_ty& e) {
      SimObject* recvObj = static_cast<SimObject*> (rs);
      recvObj->recv (e);
    }
  };

  typedef des::EventRecvTimeLocalTieBrkCmp<Event_ty> Cmp;
  typedef typename Galois::ThreadSafeOrderedSet<Event_ty, Cmp> PQ;
  // typedef typename Galois::ThreadSafeMinHeap<Event_ty, Cmp> PQ;


  static const bool DEBUG = false;

  unsigned numOutputs;
  unsigned numInputs;

  std::vector<des::SimTime> inputTimes;
  PQ pendingEvents;

public:
  static size_t NEVENTS_PER_ITER;

  SimObject (size_t id, unsigned numOutputs, unsigned numInputs)
    :
      Base (id),
      numOutputs (numOutputs),
      numInputs (numInputs)
  {
    assert (numOutputs == 1);
    inputTimes.resize (numInputs, 0);
  }


  virtual ~SimObject () {}


  void recv (const Event_ty& e) {
    size_t inIdx = this->getInputIndex (e);
    assert (inIdx < numInputs);

    // GALOIS_DEBUG ("%s, Received : %s\n", this->str ().c_str (), e.str ().c_str ());

    if (inputTimes[inIdx] > e.getRecvTime () 
        && e.getRecvTime () < des::INFINITY_SIM_TIME ) {

      Galois::gDebug ("Non-FIFO order on input[",inIdx,"], last msg time=",inputTimes[inIdx],", current message =", e.str ().c_str ());

      assert (inputTimes[inIdx] <= e.getRecvTime ());

    }


    // assert (inputTimes[inIdx] <= e.getRecvTime ());
    inputTimes[inIdx] = e.getRecvTime ();

    pendingEvents.push (e);
  }

  /**
   * Simulate.
   *
   * @param graph: the graph composed of simulation objects/stations and communication links 
   * @param myNode the node in the graph that has this SimObject as its node data
   * @return number of events that were processed during the call
   */
  template <typename G>
  size_t simulate(G& graph, typename G::GraphNode& myNode) {
    assert (isActive ());
    // if (!isActive ()) { return 0; }


    size_t nevents = 0;

    if (isActive ()) {

      des::SimTime clock = this->getClock ();
      while ((!pendingEvents.empty())
          && (pendingEvents.top ().getRecvTime () <= clock)
          && (nevents < NEVENTS_PER_ITER)) {

        Event_ty event = pendingEvents.pop ();

        // GALOIS_DEBUG ("%s, Processing: %s\n", this->str ().c_str (), event.str ().c_str ());


        //DEBUG
        if (DEBUG && !pendingEvents.empty ()) {
          Event_ty curr = event;
          Event_ty next = pendingEvents.top ();

          if (curr.getRecvTime () > next.getRecvTime ()) {
            std::cerr << "ERROR: curr > next" << std::endl;
            std::cerr << "curr = " << curr.str () << std::endl << "next = " << next.str () << std::endl;
          }
        }


        assert (graph.getData(myNode, Galois::MethodFlag::UNPROTECTED) == this); // should already own a lock
        assert (event.getRecvObj () == this);

        typename Base::template OutDegIterator<G> beg = Base::make_begin (graph, myNode);
        typename Base::template OutDegIterator<G> end = Base::make_end (graph, myNode);

        SendWrapperImpl sendWrap;

        this->execEventIntern(event, sendWrap, beg, end);

        ++nevents;
      }
    }

    return nevents;
  }



  /**
   * Checks if is active.
   * i.e. can process some of its pending events
   *
   *
   * @return true, if is active
   */
  bool isActive() const {
    // not active if pendingEvents is empty
    // not active if earliest pending event has a time stamp less than 
    // the latest time on an input i.e. possibly waiting for an earlier 
    // event on some input
    bool notActive = true;

    if (!pendingEvents.empty ()) {
      notActive = false;

      const des::SimTime& min_time = pendingEvents.top ().getRecvTime ();

      for (std::vector<des::SimTime>::const_iterator t = inputTimes.begin ()
          , endt = inputTimes.end (); t != endt; ++t) {

        if ((*t < des::INFINITY_SIM_TIME) && (*t < min_time)) {
          // not active if waiting for an earlier message on an input
          // input considered dead if last message on the input had a time stamp
          // of INFINITY_SIM_TIME or greater
          notActive = true;
          break;
        }
      }

    }

    return !notActive;
  }

  size_t numPendingEvents () const {
    return pendingEvents.size ();
  }

  /**
   * string representation for printing
   */
  virtual std::string str() const {

    std::ostringstream ss;
    ss << Base::str ();

    for (size_t i = 0; i < numInputs; ++i) {
      ss << ", inputTimes[" << i << "] = " << inputTimes[i];
    }

    if (DEBUG) {
      for (size_t i = 0; i < numInputs; ++i) {
        ss << ", inputTimes[" << i << "] = " << inputTimes[i];
      }
      ss << std::endl;

      ss << ", active = " << isActive () << ", pendingEvents.size() = " << pendingEvents.size ()
          << ", pendingEvent.top () = " << pendingEvents.top ().str () << std::endl;


    }

    return ss.str ();
  }


protected:
  /**
   * @return the min of the time stamps of the latest message recieved on each
   * input
   * An input becomes dead when a message with time INFINITY_SIM_TIME is received
   * on it,
   * such dead inputs are not included in clock computation
   */
  des::SimTime getClock () const {
    assert (inputTimes.size () == numInputs);

    des::SimTime min_t = 2 * des::INFINITY_SIM_TIME; // to ensure a value of INFINITY_SIM_TIME + any small delay

    for (std::vector<des::SimTime>::const_iterator i = inputTimes.begin ()
        , endi = inputTimes.end (); i != endi; ++i) {

      if (*i < des::INFINITY_SIM_TIME) { // 
        min_t = std::min (*i, min_t);
      }
    }

    return min_t;

    // std::vector<des::SimTime>::const_iterator min_pos = std::min_element (inputTimes.begin (), inputTimes.end ());
    // return *min_pos;
  }

  


}; // end class


} // end namespace des_unord

template <typename Event_tp>
size_t des_unord::SimObject<Event_tp>::NEVENTS_PER_ITER = 1;


#endif /* SIMOBJECT_H_ */
