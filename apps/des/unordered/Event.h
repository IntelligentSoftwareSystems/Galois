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


  /** type of the event */
  Type type;

  /**
   * Constructor
   *
   * @param id: not guaranteed to be unique. Only events from same sender must have unique id's
   * @param sendObj: the sending node in the graph
   * @param recvObj: the receiving node in the graph
   * @param action: a user defined object describing the action to be performed on the receipt of
   * this event
   * @param type: the type of the event @see EventType
   * @param sendTime: sending time
   * @param recvTime: receiving time
   */

  Event (size_t id, const S& sendObj, const S& recvObj, const A& action, const Type type, const SimTime& sendTime, const SimTime& recvTime)
    : BaseEvent <S, A> (id, sendObj, recvObj, action, sendTime, recvTime), type (type) {}


  // only this class can use the constructor
  friend class AbstractSimObject;
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

      default:
        return "";
    }
  }

};


/**
 * EventRecvTimeLocalTieBrkCmp is used to compare two events and 
 * break ties when the receiving time of two events is the same
 *
 * Ties between events with same recvTime need to be borken consistently,
 * i.e. compare(m,n) and compare (n,m) are consistent with each other during 
 * the life of events 'm' and 'n'. 
 *
 * There are at least two reasons for breaking ties between events of same time stamps:
 *
 * - Chandy-Misra's algorithm requires FIFO communication channels on edges between the 
 *   stations. i.e. On a given input edge, two events with the same time stamp should not be
 *   reordered. Therefore ties must be resolved for events received on the same input i.e. when
 *   the sender is the same for two events.
 *
 * - PriorityQueue's are usually implemented using heaps and trees, which rebalance when an item is
 *   removed/added. This means if we add two items 'a' and 'b' with the same priority in the time
 *   order (a,b), then depending on what other itmes are added and removed, we may end up removing 'a' and
 *   'b' in the order (b,a), i.e. PriorityQueue may reorder elements of same priority. Therefor,
 *   If we break ties between events on same input and not break ties between events
 *   on different inputs, this may result in reordering events on the same input.
 *
 */

template <typename EventTy> 
struct EventRecvTimeLocalTieBrkCmp {

  /**
   * 
   * Compares two events 'left' and 'right' based on getRecvTime().
   * if recvTime is same, then we compare the sender (using id), because two events from the same
   * sender should not be reordered. 
   * If the sender is the same then we use the id on the event to 
   * break the tie, since, sender is guaranteed to assign a unique
   * id to events
   *
   *
   * @param left
   * @param right
   * @return -1 if left < right. 1 if left > right. Should not return 0 unless left and right are
   * aliases
   */

  int compare (const EventTy& left, const EventTy& right) const {
    int res;
    if ( left.getRecvTime () < right.getRecvTime ()) {
      res = -1;

    } else if (left.getRecvTime () > right.getRecvTime ()) {
      res = 1;

    } else {

      res = left.getSendObj ()->getId () - right.getSendObj ()->getId ();

      if (res == 0) { // same sender
        res = left.getId () - right.getId ();
      }
        
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
