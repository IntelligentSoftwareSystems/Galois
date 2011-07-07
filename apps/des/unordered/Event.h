/** An element -*- C++ -*-
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


#ifndef _EVENT_H__
#define _EVENT_H__


#include "comDefs.h"
#include "BaseEvent.h"

/**
 * @section Description
 *
 * Represents the basic structure an unordered Event
 */

template <typename S, typename A>
class Event: public BaseEvent<S, A> {
public:
  /** 
   * An event can be a NULL_EVENT @see Chandy-Misra's paper
   * or a normal REGULAR_EVENT
   */

  enum Type {
    REGULAR_EVENT, NULL_EVENT
  };

private:
  /** The MIN_DELAY allowed for new events */
  static const SimTime MIN_DELAY = 1l;


  /** type of the event */
  Type type;

protected:

  /**
   * Constructor
   *
   * @param sendObj: the sending node in the graph
   * @param recvObj: the receiving node in the graph
   * @param action: a user defined object describing the action to be performed on the receipt of
   * this event
   * @param type: the type of the event @see EventType
   * @param sendTime: sending time
   * @param recvTime: receiving time
   */

  Event (const S& sendObj, const S& recvObj, const A& action, const Type type, const SimTime& sendTime, const SimTime& recvTime)
    : BaseEvent <S, A> (sendObj, recvObj, action, sendTime, recvTime), type (type) {}

public:

  /**
   * @return the type of this Event
   */
  const Type& getType () const {
    return type;
  }

  /**
   * @return detailed string representaion
   */
  const std::string detailedString () const {
    switch (type) {
      case REGULAR_EVENT:
        return BaseEvent<S, A>::detailedString () + "type: REGULAR_EVENT";
      case NULL_EVENT:
        return BaseEvent<S, A>::detailedString () + "type: NULL_EVENT";

    }
  }

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
  static Event<S, A> makeEvent(const S& sendObj, const S& recvObj, const Type& type, const A&  act
      , const SimTime& sendTime, SimTime delay = MIN_DELAY) {

    if (delay <= 0) {
      delay = MIN_DELAY;
    }

    SimTime recvTime;
    if (sendTime == INFINITY_SIM_TIME) {
      recvTime = INFINITY_SIM_TIME;
    } else {
      recvTime = sendTime + delay;
    }
    return  Event<S, A>(sendObj, recvObj, act, type, sendTime, recvTime);
  }

};



#endif
