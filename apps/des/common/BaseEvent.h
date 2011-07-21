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

/**
 * The Class BaseEvent.
 *
 * @param <S> the type representing the simulation object 
 * @param <A> the type representing the action to be performed on receipt of this event
 */
template <typename S, typename A>
class BaseEvent {

private:
  /** The id: not guaranteed to be unique. */
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
  



protected:
  /**
   * Instantiates a new base event.
   *
   * @param id: not guaranteed to be unique
   * @param sendObj the sending simulation obj
   * @param recvObj the receiving simulatio obj
   * @param action the action
   * @param sendTime the send time
   * @param recvTime the recv time
   */
  BaseEvent (size_t id, const S& sendObj, const S& recvObj, const A& action, const SimTime& sendTime, const SimTime& recvTime):
    id (id), sendObj (sendObj), recvObj (recvObj), action (action), sendTime (sendTime), recvTime (recvTime) {}


public:

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
    ss << getRecvTime ();
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

};





#endif
