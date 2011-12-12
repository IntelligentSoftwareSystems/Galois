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
#include <algorithm>

#include <cassert>

#include "Galois/Graphs/Graph.h"

#include "comDefs.h"
#include "Event.h"

#include "SimObject.h"



/**
 * @section Description
 *
 * The Class AbstractSimObject represents an abstract simulation object (processing station). A simulation application
 * would inherit from this class.
 */

class AbstractSimObject: public SimObject {
private:
  
  /**
   * This implementation of PriorityQueue imitates std::prioirty_queue but
   * allows reserving space in the underlying std::vector
   */
  template <typename T, typename Cmp> 
  class CustomPriorityQueue {

    Cmp cmp;
    std::vector<T> vec;
    
  public:
    typedef std::vector<T> container_type;

    typedef typename container_type::value_type value_type;
    typedef typename container_type::reference reference;
    typedef typename container_type::const_reference const_reference;
    typedef typename container_type::size_type size_type;


    CustomPriorityQueue (const Cmp& cmp = Cmp(), const std::vector<T>& vec = std::vector<T>()) 
      : cmp(cmp), vec(vec) {
      std::make_heap ( this->vec.begin (), this->vec.end (), cmp);
    }

    template <typename Iter> 
    CustomPriorityQueue (Iter b, Iter e, const Cmp& cmp = Cmp ()) : cmp (cmp) {
      vec.insert (vec.end (), b, e);

      std::make_heap (vec.begin (), vec.end (), cmp);
    }

    CustomPriorityQueue (const CustomPriorityQueue& that): cmp(that.cmp), vec (that.vec) {
    }


    bool empty () const {
      return vec.empty ();
    }

    size_type size () const {
      return vec.size ();
    }

    const_reference top () const {
      return vec.front ();
    }

    void push (const value_type& x) {
      vec.push_back (x);
      std::push_heap (vec.begin (), vec.end (), cmp);
    }

    void pop () {
      std::pop_heap (vec.begin (), vec.end (), cmp);

      vec.pop_back ();
    }

    void reserve (size_type s) {
      assert (s > 0);
      vec.reserve (s);
    }

  };

public:
  /**
   * Upper limit on the number of events processed by a SimObject
   * during a call to @see SimObject::simulate
   */
  static size_t NEVENTS_PER_ITER;


private:

  static const bool DEBUG = false;

  /** multiplication factor for space reserved in the prioirty queue */
  static const size_t PQ_OVER_RESERVE = 1024;

  // typedef std::priority_queue<EventTy, std::vector<EventTy>, EventRecvTimeLocalTieBrkCmp<EventTy> > PriorityQueue;
  typedef CustomPriorityQueue<EventTy, EventRecvTimeLocalTieBrkCmp<EventTy> > PriorityQueue;

  /** The id. */
  size_t id;

  /** The number of inputs. */
  size_t numInputs;

  /** number of outputs */
  size_t numOutputs;

  // std::vector<PriorityQueue> inputEvents;

  /** store the timestamp of latest event received on an input line */
  std::vector<SimTime> inputTimes;

  /** local clock value, is the minimum of events received on all input lines */
  SimTime clock;

  /** 
   * Events received on any input go into pendingEvents and are stored here until processed.
   * readyEvents set, is a set of events that can safely be processed, if this AbstractSimObject is
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
   * Counter to assign id's to events sent by this SimObject
   */
  size_t eventIdCntr;

  /**
   * Inits the object.
   *
   * @param id unique identifier for this object 
   * @param numOutputs the number of outputs
   * @param numInputs the number of  inputs
   */
  void init(size_t id, size_t numOutputs, size_t numInputs) {
    eventIdCntr = 0; 
    this->id = id;

    assert (numOutputs == 1);
    this->numOutputs = numOutputs;

    this->numInputs = numInputs;


    inputTimes.resize (numInputs);

    for (size_t i = 0; i < numInputs; ++i) {
      inputTimes[i] = SimTime(0);
    }

    pendingEvents = PriorityQueue();

    // reserving space upfront to avoid memory allocation due to doubling of the PriorityQueue etc
    pendingEvents.reserve (numInputs * NEVENTS_PER_ITER * PQ_OVER_RESERVE);

    clock = 0;

  }

  /**
   * performs a deep copy.
   *
   * @param that the that
   */
  void deepCopy(const AbstractSimObject& that) {

    init(that.id, that.numOutputs, that.numInputs);

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
    this->eventIdCntr = that.eventIdCntr;
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
   * @param id Must be unique 
   * @param numOutputs the number of outputs
   * @param numInputs the number of  inputs
   */
  AbstractSimObject(size_t id, size_t numOutputs, size_t numInputs): SimObject() {
    assert (numOutputs == 1);
    init(id, numOutputs, numInputs);
  }

  AbstractSimObject (const AbstractSimObject& that): SimObject (that) {
    deepCopy (that);
  }

  virtual ~AbstractSimObject () {}
  /**
   * a way to construct different subtypes
   * @return a copy of this
   */
  virtual AbstractSimObject* clone() const = 0;

  /**
   * Make event.
   *
   * @param sendObj the send obj
   * @param recvObj the recv obj
   * @param type the type
   * @param act the action to be performed
   * @param sendTime the send time
   * @param delay the delay
   * @return the event
   */
   EventTy makeEvent(SimObject* sendObj, SimObject* recvObj, const EventTy::Type& type, const LogicUpdate&  act
      , const SimTime& sendTime, SimTime delay = MIN_DELAY) {

    assert (sendObj == this);

    if (delay <= 0) {
      delay = MIN_DELAY;
    }

    SimTime recvTime;
    if (sendTime >= INFINITY_SIM_TIME) {
      recvTime = INFINITY_SIM_TIME;
    } else {
      recvTime = sendTime + delay;
    }
    return  EventTy((eventIdCntr++), sendObj, recvObj, act, type, sendTime, recvTime);
  }


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
  virtual void execEvent(Graph& graph, GNode& myNode, const EventTy& e) = 0;

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
  size_t simulate(Graph& graph, GNode& myNode) {
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

        // assert (EventRecvTimeLocalTieBrkCmp<EventTy> ().compare (prev, curr) < 0);
        if (EventRecvTimeLocalTieBrkCmp<EventTy> ().compare (curr, next) >= 0) {
          std::cerr << "EventRecvTimeLocalTieBrkCmp ().compare (curr, next) >= 0" << std::endl;
          std::cerr << "curr = " << curr.detailedString () << std::endl << "next = " << next.detailedString () << std::endl;
        }
      }


      assert (graph.getData(myNode, Galois::NONE) == this); // should already own a lock
      assert (e.getRecvObj () == this);

      execEvent(graph, myNode, e);

      ++retVal;
    }

    return (retVal);
  }




  /**
   * Checks if is active after recomputing and updating active state
   *
   * @return true, if is active
   */
  bool isActive() {
    updateActive ();
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


  virtual size_t numPendingEvents () const { 
    return pendingEvents.size ();
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

}; // end class


size_t AbstractSimObject::NEVENTS_PER_ITER = 16; // a good value for many inputs
#endif 
