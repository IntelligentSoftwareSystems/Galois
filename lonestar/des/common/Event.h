#ifndef DES_EVENT_H
#define DES_EVENT_H

#include <string>
#include <sstream>

#include "comDefs.h"
#include "BaseSimObject.h"

/**
 * The Class Event.
 *
 * @param <A> the type representing the action to be performed on receipt of this event
 */
namespace des {

template <typename A>
class Event {

  // template <typename E>
  friend class des::BaseSimObject<Event>;

public:

  enum Type { REGULAR_EVENT, NULL_EVENT };

  typedef A Action_ty;
  typedef BaseSimObject<Event> BaseSimObj_ty;

private:
  /** The id: not guaranteed to be unique. */
  size_t id;
  
  /** The send obj. */
  BaseSimObj_ty* sendObj;
  
  /** The recv obj. */
  BaseSimObj_ty* recvObj;
  
  /** The action to be performed on receipt of this event. */
  A action;

  /** type of event null or non-null */
  Type type;

  /** The send time. */
  SimTime sendTime;
  
  /** The recv time. */
  SimTime recvTime;
  



protected:
  /**
   * Instantiates a new base event.
   *
   * @param id not guaranteed to be unique
   * @param sendObj the sending simulation obj
   * @param recvObj the receiving simulatio obj
   * @param action the action
   * @param type the type
   * @param sendTime the send time
   * @param recvTime the recv time
   */
  Event (size_t id, BaseSimObj_ty* sendObj, BaseSimObj_ty* recvObj, const A& action, const Type& type, const SimTime& sendTime, const SimTime& recvTime):
    id (id), sendObj (sendObj), recvObj (recvObj), action (action), type (type), sendTime (sendTime), recvTime (recvTime) {}

public:
  friend bool operator == (const Event& left, const Event& right) {
    return (left.id == right.id)
        && (left.sendObj == right.sendObj)
        && (left.recvObj == right.recvObj)
        && (left.action == right.action)
        && (left.type == right.type)
        && (left.sendTime == right.sendTime)
        && (left.recvTime == right.recvTime);
  }

  friend bool operator != (const Event& left, const Event& right) {
    return !(left == right);
  }

  friend std::ostream& operator << (std::ostream& out, const Event& event) {
    return (out << event.str ());
  }


  /**
   * Detailed string.
   *
   * @return the string
   */
  std::string detailStr() const {
    std::ostringstream ss;
    ss << "Event-" << id << ", " << (type == NULL_EVENT ? "NULL_EVENT" : "REGULAR_EVENT")
      << ", sendTime = " << sendTime << ", recvTime = " << recvTime 
      << ", sendObj = " << sendObj->getID () << ", recvObj = " << recvObj->getID () 
      << std::endl;
      // << "action = " << action.str () << std::endl << std::endl;
    return ss.str ();
  }

  /**
   * a simpler string representation for debugging
   */
  std::string shortStr () const {
    std::ostringstream ss;
    ss << getRecvTime ();
    return ss.str ();
  }

  inline std::string str () const {
    return detailStr ();
  }

  /**
   * Gets the send obj.
   *
   * @return the send obj
   */
  BaseSimObj_ty* getSendObj() const {
    return sendObj;
  }


  /**
   * Gets the recv obj.
   *
   * @return the recv obj
   */
  BaseSimObj_ty* getRecvObj() const {
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


  const Type& getType () const {
    return type;
  }


  /**
   * Gets the id.
   *
   * @return the id
   */
  size_t getID() const {
    return id;
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

template <typename Event_tp> 
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

  static int compare (const Event_tp& left, const Event_tp& right) {
    int res;
    if ( left.getRecvTime () < right.getRecvTime ()) {
      res = -1;

    } else if (left.getRecvTime () > right.getRecvTime ()) {
      res = 1;

    } else {

      res = left.getSendObj ()->getID () - right.getSendObj ()->getID ();

      if (res == 0) { // same sender
        res = left.getID () - right.getID ();
      }
        
    }

    return res;

  }

  bool operator () (const Event_tp& left, const Event_tp& right) const {
    return compare (left, right) < 0;
  }

  /**
   * returns true if left > right
   * Since std::priority_queue is a max heap, we use > semantics instead of <
   * in order to get a min heap and thus process events in increasing order of recvTime.
   */
  struct RevCmp {
    bool operator () (const Event_tp& left, const Event_tp& right) const { 
      return EventRecvTimeLocalTieBrkCmp::compare (left, right) > 0;
    }
  };
};


} // namespace des



#endif // DES_EVENT_H 
