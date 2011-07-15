/** Abstract Simulation Object -*- C++ -*-
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

#ifndef _ABS_SIMOBJECT_H_
#define _ABS_SIMOBJECT_H_



#include <vector>
#include <queue>

#include <cassert>

#include "Galois/Graphs/Graph.h"

#include "comDefs.h"
#include "BaseEvent.h"

#include "SimObject.h"



/**
 * @section Description
 *
 * The Class AbsSimObject represents an abstract simulation object (processing station). A simulation application
 * would inherit from this class.
 */

template <typename GraphTy, typename GNodeTy, typename EventTy>
class AbsSimObject: public SimObject {
public:
  static size_t NEVENTS_PER_ITER;

private:

  static const bool DEBUG = false;

  typedef std::priority_queue<EventTy, std::vector<EventTy>, EventRecvTimeTieBrkCmp<EventTy> > PriorityQueue;

  /** The id counter.
   * each object has an id, which is assigned at the 
   * time of creation by incrementing a counter
   */
  static size_t idCntr;

  /** The id. */
  size_t id;

  /** The number of inputs. */
  size_t numInputs;

  // std::vector<PriorityQueue> inputEvents;

  /** store the timestamp of latest event received on an input line */
  std::vector<SimTime> inputTimes;

  /** local clock value, is the minimum of events received on all input lines */
  SimTime clock;

  /** 
   * Events received on any input go into pendingEvents and are stored here until processed.
   * readyEvents set, is a set of events that can safely be processed, if this AbsSimObject is
   * active.  If minRecv is the min. of the latest timestamp received on any input i.e. min of inputTimes[i] for all i
   * then events with timestamp <= minRecv go into readyEvents set. readyEvents set is a 
   * subset of pendingEvents.
   */
  PriorityQueue pendingEvents;

  /** time stamp of the  last message sent on output
  protected SimTime lastSent = 0; */


  /** whether it can process some events received on the input. i.e. if pendingEvents
    is computed, it'll be non-empty
   */
  bool active;



  /**
   * Inits the object.
   *
   * @param numOutputs the number of outputs
   * @param numInputs the number of  inputs
   */
  void init(size_t numOutputs, size_t numInputs) {
    this->id = idCntr++;

    this->numInputs = numInputs;
    inputTimes.resize (numInputs);

    for (size_t i = 0; i < numInputs; ++i) {
      inputTimes[i] = SimTime(0);
    }

    pendingEvents = PriorityQueue();

    // NEVENTS_PER_ITER * numInputs because each input can get fed with NEVENTS_PER_ITER on average
    // pendingEvents.reserve (numInputs * NEVENTS_PER_ITER);

    clock = 0;

  }

  /**
   * performs a deep copy.
   *
   * @param that the that
   */
  void deepCopy(const AbsSimObject<GraphTy, GNodeTy, EventTy>& that) {
    init(that.numInputs, 1);
    for (size_t i = 0; i < numInputs; ++i) {
      this->inputTimes[i] = that.inputTimes[i];
    }

    PriorityQueue cpyQ(that.pendingEvents);
    while (!cpyQ.empty ()) {
      this->pendingEvents.push (cpyQ.top ());
      cpyQ.pop ();
    }

    // defensive
    this->clock = that.clock;
  }

  /**
   * compute the min. of the timestamps of latest message received so far
   * for every input
   */
  void updateClock() {
    SimTime min = INFINITY_SIM_TIME;
    for (size_t i = 0; i < numInputs; ++i) {
      if (this->inputTimes[i] < min) {
        min = this->inputTimes[i];
      }
    }

    this->clock = min;
  }


  // void sendEvent(size_t outIndex, SimObject target, Event<?> e) {
    // //TODO: not implemented yet
  // }

public:
  /**
   * Instantiates a new simulation object.
   *
   * @param numOutputs the number of outputs
   * @param numInputs the number of  inputs
   */
  AbsSimObject(size_t numOutputs, size_t numInputs): SimObject() {
    init(numOutputs, numInputs);
  }

  AbsSimObject (const AbsSimObject<GraphTy, GNodeTy, EventTy>& that): SimObject (that) {
    deepCopy (that);
  }

  virtual ~AbsSimObject () {}
  /**
   * a way to construct different subtypes
   * @return a copy of this
   */
  virtual AbsSimObject<GraphTy, GNodeTy, EventTy>* clone() const = 0;


  /**
   * Recv event.
   * put in the input indexed by inputIndex
   *
   * @param inputIndex
   * @param e the event
   */
  void recvEvent(size_t inputIndex, const EventTy& e) {

    assert (inputIndex >= 0 && inputIndex < numInputs);

    this->pendingEvents.push (e);

    if (this->inputTimes[inputIndex] < e.getRecvTime()) {
      this->inputTimes[inputIndex] = e.getRecvTime();
    }
  }

  /**
   * Exec event.
   *
   * The user code should override this method inorder to
   * define the semantics of executing and event on a sub-type
   *
   * @param graph: the graph containg simulation objects as nodes and communication links as edges
   * @param myNode: the node in the graph that has this SimObject as its node data
   * @param e: the input event
   */
  virtual void execEvent(GraphTy& graph, GNodeTy& myNode, const EventTy& e) = 0;

  /**
   * Simulate.
   *
   * pre-condition: The object should be active
   *
   * Computes the set of readyEvents, then executes the events
   * in time-stamp order
   *
   * Here the set of readyEvents is the subset of pendingEvents that have a 
   * recvTime <= this->clock
   *
   * The parameter NEVENTS_PER_ITER defines an upper limit on the number of ready events
   * that can be processed in one call to simulate
   *
   * @param graph: the graph containg simulation objects as nodes and communication links as edges
   * @param myNode the node in the graph that has this SimObject as its node data
   * @return number of events ready to be executed.
   */
  size_t simulate(GraphTy& graph, GNodeTy& myNode) {
    assert (isActive ());

    updateClock();// update the clock, 

    size_t retVal = 0;

    while ((!pendingEvents.empty())
        && (pendingEvents.top ().getRecvTime () <= this->clock)
        && (retVal < NEVENTS_PER_ITER)) {

      EventTy e = pendingEvents.top ();
      pendingEvents.pop ();

      //DEBUG
      if (!pendingEvents.empty ()) {
        const EventTy& curr = e;
        const EventTy& next = (pendingEvents.top ());

        // assert (EventRecvTimeTieBrkCmp<EventTy> ().compare (prev, curr) < 0);
        if (EventRecvTimeTieBrkCmp<EventTy> ().compare (curr, next) >= 0) {
          std::cerr << "EventRecvTimeTieBrkCmp ().compare (curr, next) >= 0" << std::endl;
          std::cerr << "curr = " << curr.detailedString () << std::endl << "next = " << next.detailedString () << std::endl;
        }
      }


      assert (graph.getData(myNode, Galois::NONE) == this); // should already own a lock

      execEvent(graph, myNode, e);

      ++retVal;
    }

    return (retVal);
  }




  /**
   * Checks if is active.
   *
   * @return true, if is active
   */
  bool isActive() const {
    return active;
  }

  /**
   * active is set to true if there exists a pending event(s) on each input pq
   * or if one input pq (not all) is empty and an event with INFINITY_SIM_TIME has already been received
   * telling that no more events on this input will be received.
   *
   * We can tell if there are pending events on input i if inputTimes[i] >= pendingEvents.top ()
   * We can tell if there are no pending events by checking pendingEvents.empty ();
   *
   */
  void updateActive() {

    bool allEmpty = pendingEvents.empty ();

    bool isWaiting = false; // is Waiting on an input event

    if (allEmpty) {
      // pendingEvents is empty, but how do we know whether this is the beginning of simulation
      // or the end. In the beginning we are waiting on input events with empty queue in the
      // end we're not.
      // we are not waiting only if the time stamp of latest event is >= INFINITY_SIM_TIME
      for (size_t i = 0; i < numInputs; ++i) {
        if (inputTimes[i] <  INFINITY_SIM_TIME) {
          isWaiting = true;
          break;
        }
      }

    } else {
      // still some pending events to process
      // what we need to determine now is whether we can process the pending events safely
      // which can be determined by checking pendingEvents.top () against inputTimes[i] for input i
      // because if inputTimes[i] < pendingEvents.top () then we cannot safely process any
      // pendingEvents because there's a possibility of receiving more events on input i which have
      // a time stamp less that pendingEvents.top ()


      const SimTime& top = pendingEvents.top ().getRecvTime ();
      for (size_t i = 0; i < numInputs; ++i) {
        if (inputTimes[i] < top) {
          isWaiting = true;
          break;
        }
      }
    }


    active = !allEmpty && !isWaiting;
  }


  /** 
   * @return a string representation for printing purposes
   */
  virtual const std::string toString() const {
    std::ostringstream ss;
    ss << "SimObject-" << id << " ";
    if (DEBUG) {
      for (size_t i = 0; i < numInputs; ++i) {
        ss << ", inputTimes[" << i << "] = " << inputTimes[i];
      }
      ss << std::endl;

      ss << ", active = " << active << ", clock = " << clock << ", pendingEvents.size() = " << pendingEvents.size ()
          << ", pendingEvent.top () = " << pendingEvents.top ().toString () << std::endl;


    }

    return ss.str ();
  }

  /**
   * Gets the id.
   *
   * @return the id
   */
  size_t getId() const { return id; }

  /**
   * resets the id counter to 0
   */
  static void resetIdCounter () {
    idCntr = 0;
  }

}; // end class

template <typename GraphTy, typename GNodeTy, typename EventTy>
size_t AbsSimObject<GraphTy, GNodeTy, EventTy>::idCntr = 0;

template <typename GraphTy, typename GNodeTy, typename EventTy>
size_t AbsSimObject<GraphTy, GNodeTy, EventTy>::NEVENTS_PER_ITER = 1;
#endif 
