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

private:

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

  /** store the incoming events on each input */
  std::vector<PriorityQueue> inputEvents;

  /** store the timestamp of latest event received on an input line */
  std::vector<SimTime> inputTimes;

  /** local clock value, is the minimum of events received on all input lines */
  SimTime clock;

  /** readyEvents set, is a pq of events that can safely be processed, if this AbsSimObject is
   * active.  If minRecv is the min. of the latest timestamp received on any event i.e. min of inputTimes
    then events with timestamp <= minRecv go into readyEvents
   */
  PriorityQueue readyEvents;

  /** time stamp of the  last message sent on output
  protected SimTime lastSent = 0; */


  /** whether it can process some events received on the input. i.e. if readyEvents
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
    inputEvents.resize (numInputs);
    inputTimes.resize (numInputs);

    for (size_t i = 0; i < numInputs; ++i) {
      inputTimes[i] = 0;
      inputEvents[i] = PriorityQueue();
    }

    readyEvents = PriorityQueue();
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

      PriorityQueue cpyQ(that.inputEvents[i]);

      while (!cpyQ.empty ()) {
        this->inputEvents[i].push (cpyQ.top ());
        cpyQ.pop ();
      }

    }

    // defensive
    this->clock = that.clock;
  }

  /**
   * compute the minimum time of a message received so far
   * for every input
   * if pq is not empty then take the time of min event in the pq
   * else take the time of the last event received on the input.
   */
  /*
  protected void computeClock() {
    SimTime min = INFINITY_SIM_TIME;
    for(int i = 0; i < numInputs; ++i) {
      PriorityQueue<Event> pq = this->inputEvents[i];
      if(!pq.isEmpty()) {
        if(pq.peek().recvTime < min ) {
          min = pq.peek().recvTime;
        }
      }
      else {
        min = this->inputTimes[i];
      }
    }

    this->clock = min;

  }
   */

  /**
   * compute the min. of the timestamps of latest message received so far
   * for every input
   * if pq is not empty then take the time of min event in the pq
   * else take the time of the last event received on the input.
   */
  void computeClock() {
    SimTime min = INFINITY_SIM_TIME;
    for (size_t i = 0; i < numInputs; ++i) {
      if (min < this->inputTimes[i]) {
        min = this->inputTimes[i];
      }
    }

    this->clock = min;
  }


  // void sendEvent(size_t outIndex, SimObject target, Event<?> e) {
    // //TODO: not implemented yet
  // }

  /**
   * Populate ready events.
   *
   * computes a PriorityQueue of events that have recvTime <= this->clock
   * called after @see computeClock () has been called
   */
  void populateReadyEvents() {
    assert (readyEvents.empty ());
    for (typename std::vector<PriorityQueue>::iterator i = inputEvents.begin (), e = inputEvents.end (); i != e; ++i) {
      PriorityQueue& pq = *i;
      while (!pq.empty () && pq.top ().getRecvTime () <= this->clock) {
        this->readyEvents.push (pq.top ());
        pq.pop ();
      }

      // In while(!pq.isEmpty() && pq.peek().recvTime <= this.clock) {
      // changing 'while' to 'if' results into processing one event per input per iteration
      // this increases parallelism, 'while' reduces parallelism but increases the 
      // work performed per iteration (total work remains the same).
    }

  }


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

    assert (inputIndex >= 0 && inputIndex < inputEvents.size ());

    this->inputEvents[inputIndex].push (e);
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
   * @param graph: the graph containg simulation objects as nodes and communication links as edges
   * @param myNode the node in the graph that has this SimObject as its node data
   * @return number of events ready to be executed.
   */
  size_t simulate(GraphTy& graph, GNodeTy& myNode) {
    computeClock();// update the clock, can do at the end if null msgs are propagated initially
    populateReadyEvents(); // fill up readyEvents,
    size_t retVal = this->readyEvents.size();

    while (!readyEvents.empty()) {
      EventTy e = readyEvents.top ();
      readyEvents.pop ();

      //DEBUG
      if (!readyEvents.empty ()) {
        EventTy curr (e);
        EventTy next (readyEvents.top ());

        // assert (EventRecvTimeTieBrkCmp<EventTy> ().compare (prev, curr) < 0);
        if (EventRecvTimeTieBrkCmp<EventTy> ().compare (curr, next) >= 0) {
          std::cerr << "EventRecvTimeTieBrkCmp ().compare (curr, next) >= 0" << std::endl;
          std::cerr << "curr = " << curr.detailedString () << std::endl << "next = " << next.detailedString () << std::endl;
        }
      }


      assert (graph.getData(myNode, Galois::Graph::NONE) == this); // should already own a lock

      execEvent(graph, myNode, e);
    }

    return retVal;
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
   * active is set to true if there exists an event on each input pq
   * or if an input pq is empty and an event with INFINITY_SIM_TIME has been received
   * telling that no more events on this input will be received.
   */
  void updateActive() {
    bool isWaiting = false; // is Waiting on an input event
    // i.e. pq of the input is empty and the time of last evetn is not INFINITY_SIM_TIME
    bool allEmpty = true;
    for (size_t i = 0; i < numInputs; ++i) {
      if (inputEvents[i].empty ()) {
        if (inputTimes[i] < INFINITY_SIM_TIME) {
          isWaiting = true;
        }
      } else {
        allEmpty = false;
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

#endif 
