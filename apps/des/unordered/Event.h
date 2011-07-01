#ifndef _EVENT_H__
#define _EVENT_H__


#include "comDefs.h"
#include "BaseEvent.h"

template <typename S, typename A>
class Event: public BaseEvent<S, A> {
public:
  enum Type {
    REGULAR_EVENT, NULL_EVENT
  };

private:
  /** The MI n_ delay. */
  static const SimTime MIN_DELAY = 1l;


  Type type;

protected:
  Event (const S& sendObj, const S& recvObj, const A& action, const Type type, const SimTime& sendTime, const SimTime& recvTime)
    : BaseEvent <S, A> (sendObj, recvObj, action, sendTime, recvTime), type (type) {}

public:

  const Type& getType () const {
    return type;
  }

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
   * @param <M> the generic type representing the message type
   * @param sendObj the send obj
   * @param recvObj the recv obj
   * @param type the type
   * @param msg the msg
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
