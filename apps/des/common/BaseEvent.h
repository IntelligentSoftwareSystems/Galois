#ifndef _BASE_EVENT_H_
#define _BASE_EVENT_H_

#include <string>
#include <sstream>

#include "comDefs.h"

#include "Galois/util/Atomic.h"

/**
 * The Class BaseEvent.
 *
 * @param <S> the generic type representing the simulation object 
 * @param <A> the generic type representing the action type
 */
template <typename S, typename A>
class BaseEvent {

private:
  /** The id counter for assigning ids to events. */
  static Galois::AtomicInteger idCntr;

  /** The id. */
  size_t id;
  
  /** The send obj. */
  S sendObj;
  
  /** The recv obj. */
  S recvObj;
  
  /** The action. */
  A action;

  /** The send time. */
  SimTime sendTime;
  
  /** The recv time. */
  SimTime recvTime;
  



public:
  /**
   * Instantiates a new base event.
   *
   * @param sendObj the send obj
   * @param recvObj the recv obj
   * @param action the action
   * @param sendTime the send time
   * @param recvTime the recv time
   */
  BaseEvent (const S& sendObj, const S& recvObj, const A& action, const SimTime& sendTime, const SimTime& recvTime):
    id (idCntr.getAndIncrement ()), sendObj (sendObj), recvObj (recvObj), action (action), sendTime (sendTime), recvTime (recvTime) {}

  BaseEvent (const BaseEvent<S,A>& that): id (that.id), sendObj (that.sendObj), recvObj (that.recvObj), action (that.action)
  , sendTime (that.sendTime), recvTime (that.recvTime) {}

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

  /*
   * compares equality two messages ignoring their types
   * for now the equals criteria include:
   * - id, sendObj, recvObj, sendTime, recvTime and action references equal
   *   but not the action types
   *
   */

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

  /* (non-Javadoc)
   * @see java.lang.Object#toString()
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
Galois::AtomicInteger BaseEvent<S, A>::idCntr(0);

template <typename S, typename A>
void BaseEvent<S, A>::resetIdCounter () {
  BaseEvent<S, A>::idCntr.set(0);
}



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
