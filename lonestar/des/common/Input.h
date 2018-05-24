#ifndef DES_INPUT_H_
#define DES_INPUT_H_

#include <string>
#include <cassert>

#include "SimGate.h"
#include "BasicPort.h"


namespace des {

template <typename S>
class Input: public SimGate<S> {

protected:
  typedef SimGate<S> Base;
  typedef typename Base::Event_ty Event_ty;

public: 
  /**
   * Instantiates a new Input.
   */
  Input(size_t id, des::BasicPort& impl)
    : Base (id, impl) {}


  virtual Input* clone () const {
    return new Input (*this);
  }

  virtual des::BasicPort& getImpl () const {
    assert (dynamic_cast<des::BasicPort*> (&Base::getImpl ()) != NULL);
    des::BasicPort* ptr = static_cast<des::BasicPort*> (&Base::getImpl ());
    assert (ptr != NULL);
    return *ptr;
  }

  /**
   * A string representation
   */
  virtual std::string str () const {
    std::ostringstream ss;
    ss << "Input: " << Base::str ();
    return ss.str ();
  }

protected:
  /**
   * Sends a copy of event at the input to all the outputs in the circuit
   * @see OneInputGate::execEvent().
   *
   * @param event the event
   * @param sendWrap
   * @param b begining of range
   * @param e end of range
   */
  virtual void execEventIntern (const Event_ty& event, 
      typename Base::SendWrapper& sendWrap, 
      typename Base::BaseOutDegIter& b, typename Base::BaseOutDegIter& e) {

    if (event.getType () == Event_ty::NULL_EVENT) {
      Base::execEventIntern (event, sendWrap, b, e);

    } else {

      const des::LogicUpdate& lu = event.getAction ();
      if (getImpl().getInputName () == lu.getNetName()) {
        getImpl().setInputVal (lu.getNetVal());
        getImpl().setOutputVal (lu.getNetVal());

        des::LogicUpdate drvFanout (getImpl ().getOutputName (), getImpl ().getOutputVal ());

        Base::sendEventsToFanout (event, drvFanout, Event_ty::REGULAR_EVENT, sendWrap, b, e);
      } else {
        getImpl ().netNameMismatch (lu);
      }
    }

  }


};

} // end namespace des
#endif /* DES_INPUT_H_ */
