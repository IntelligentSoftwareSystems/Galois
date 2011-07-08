/** BaseEvent: is the basic structure of an event in the simulation -*- C++ -*-
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

#ifndef _BASE_EVENT_H_
#define _BASE_EVENT_H_

#include <string>
#include <sstream>

#include "comDefs.h"

#include "Galois/util/Atomic.h"

/**
 * The Class BaseEvent.
 *
 * @param <S> the type representing the simulation object 
 * @param <A> the type representing the action to be performed on receipt of this event
 */
template <typename S, typename A>
class BaseEvent {

private:
  /** The id counter for assigning ids to events.
   * Events need to be globally ordered for correctness. The
   * ties between events having same receive time are broken using
   * this id
   * The counter is atomic so that events created dynamically don't have duplicate ids
   */
  static Galois::GAtomic<int> idCntr;

  /** The id. */
  size_t id;
  
  /** The send obj. */
  S sendObj;
  
  /** The recv obj. */
  S recvObj;
  
  /** The action to be performed on receipt of this event. */
  A action;

  /** The send time. */
  SimTime sendTime;
  
  /** The recv time. */
  SimTime recvTime;
  



public:
  /**
   * Instantiates a new base event.
   *
   * @param sendObj the sending simulation obj
   * @param recvObj the receiving simulatio obj
   * @param action the action
   * @param sendTime the send time
   * @param recvTime the recv time
   */
  BaseEvent (const S& sendObj, const S& recvObj, const A& action, const SimTime& sendTime, const SimTime& recvTime):
    id (idCntr++), sendObj (sendObj), recvObj (recvObj), action (action), sendTime (sendTime), recvTime (recvTime) {}


  /**
   * copy constructor copies over id as well
   */
  BaseEvent (const BaseEvent<S,A>& that): id (that.id), sendObj (that.sendObj), recvObj (that.recvObj), action (that.action)
  , sendTime (that.sendTime), recvTime (that.recvTime) {}

  /**
   * assignment operator assigns id as well
   */
  BaseEvent<S, A>& operator = (const BaseEvent<S, A>& that) {
    if (this != &that) {
      id = that.id;
      sendObj = that.sendObj;
      recvObj = that.recvObj;
      action = that.action;
      sendTime = that.sendTime;
      recvTime = that.recvTime;
    }
    return (*this);
  }

  /**
   * Detailed string.
   *
   * @return the string
   */
  const std::string detailedString() const {
    std::ostringstream ss;
    ss << "Event-" << id << ":" << "sendTime = " << sendTime << "sendObj = " << &sendObj << std::endl 
      << "recvTime = " << recvTime << "recvObj = " << &recvObj << std::endl 
      << "action = " << action.toString () << std::endl;
    return ss.str ();
  }

  /**
   * a simpler string representation for debugging
   */
  const std::string toString () const {
    std::ostringstream ss;
    ss << getRecvTime;
    return ss.str ();
  }

  /**
   * Gets the send obj.
   *
   * @return the send obj
   */
  const S& getSendObj() const {
    return sendObj;
  }


  /**
   * Gets the recv obj.
   *
   * @return the recv obj
   */
  const S& getRecvObj() const {
    return recvObj;
  }


  /**
   * Gets the send time.
   *
   * @return the send time
   */
  const SimTime& getSendTime() const {
    return sendTime;
  }

  /**
   * Gets the recv time.
   *
   * @return the recv time
   */
  const SimTime& getRecvTime() const {
    return recvTime;
  }
  /**
   * Gets the action.
   *
   * @return the action
   */
  const A& getAction() const {
    return action;
  }


  /**
   * Gets the id.
   *
   * @return the id
   */
  size_t getId() const {
    return id;
  }


  /**
   * Reset id counter
   */
  static void resetIdCounter(); 
};

template <typename S, typename A>
Galois::GAtomic<int> BaseEvent<S, A>::idCntr(0);

template <typename S, typename A>
void BaseEvent<S, A>::resetIdCounter () {
  BaseEvent<S, A>::idCntr = 0;
}


/**
 * EventRecvTimeTieBrkCmp is used to compare two events and 
 * break ties when the receiving time of two events is the same
 * Ties are resolved based on the ids of the events. Such global
 * ordering is necessary to ensure the events are added and removed
 * from an min-heap (used to implement priority queue) in a consistend order
 */

template <typename EventTy> 
struct EventRecvTimeTieBrkCmp {

  int compare (const EventTy& left, const EventTy& right) const {
    int res;
    if ( left.getRecvTime () < right.getRecvTime ()) {
      res = -1;

    } else if (left.getRecvTime () > right.getRecvTime ()) {
      res = 1;

    } else {
      res = left.getId () - right.getId ();
    }

    return res;

  }

  /**
   * returns true if left > right
   * Since std::priority_queue is a max heap, we use > semantics instead of <
   * in order to get a min heap and thus process events in increasing order of recvTime.
   */
  bool operator () (const EventTy& left, const EventTy& right) const {
    return compare (left, right) > 0;
  }

};



#endif
