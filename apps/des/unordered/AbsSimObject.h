#ifndef _ABS_SIMOBJECT_H_
#define _ABS_SIMOBJECT_H_

#include <vector>
#include <queue>

#include <cassert>

#include "Galois/Graphs/Graph.h"

#include "comDefs.h"
#include "BaseEvent.h"

#include "SimObject.h"



//TODO: modeling one output for now. Need to extend for multiple outputs
/**
 * The Class AbsSimObject represents an abstract simulation object (processing station). A simulation application
 * would inherit from this class.
 */

template <typename GraphTy, typename GNodeTy, typename EventTy>
class AbsSimObject: public SimObject {

private:

  typedef std::priority_queue<EventTy, std::vector<EventTy>, EventRecvTimeTieBrkCmp<EventTy> > PriorityQueue;

  /** The id counter. */
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

  /** readyEvents set, is a pq of events that can safely be processed now
    if minRecv = min latest timestamp received on any event i.e. min of inputTimes
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

  void computeClock() {
    SimTime min = INFINITY_SIM_TIME;
    for (size_t i = 0; i < numInputs; ++i) {
      if (min < this->inputTimes[i]) {
        min = this->inputTimes[i];
      }
    }

    this->clock = min;
  }


  /**
   * Send event.
   *
   * @param outIndex the out index
   * @param target the target
   * @param e the e
   */
  // void sendEvent(size_t outIndex, SimObject target, Event<?> e) {
    // //TODO: not implemented yet
  // }

  /**
   * Populate ready events.
   *
   * @return PriorityQueue of events that have recvTime <= this->clock
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
   * @return a new instance of subtype
   */
  virtual AbsSimObject<GraphTy, GNodeTy, EventTy>* clone() const = 0;


  /**
   * Recv event.
   *
   * @param in the in
   * @param e the e
   */
  void recvEvent(size_t inputIndex, const EventTy& e) {

    assert (inputIndex >= 0 && inputIndex < inputEvents.size ());

    this->inputEvents[inputIndex].push (e);
    if (this->inputTimes[inputIndex] < e.getRecvTime()) {
      this->inputTimes[inputIndex] = e.getRecvTime();
    }
  }
  // The user code should override this method inorder to
  // define the semantics of executing and event on
  /**
   * Exec event.
   *
   * @param myNode the node in the graph that has this SimObject as its node data
   * @param e the input event
   */
  virtual void execEvent(GraphTy&, GNodeTy& myNode, const EventTy& e) = 0;

  /**
   * Simulate.
   *
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


  /* (non-Javadoc)
   * @see java.lang.Object#toString()
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

  static void resetIdCounter () {
    idCntr = 0;
  }

}; // end class

template <typename GraphTy, typename GNodeTy, typename EventTy>
size_t AbsSimObject<GraphTy, GNodeTy, EventTy>::idCntr = 0;

#endif 
