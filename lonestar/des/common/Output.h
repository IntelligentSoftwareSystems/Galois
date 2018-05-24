#ifndef DES_OUTPUT_H_
#define DES_OUTPUT_H_


#include <iostream>
#include <string>
#include <cassert>


#include "Input.h"

namespace des {
/**
 * The Class Output.
 */
template <typename S>
class Output: public Input<S> {

  typedef Input<S> Base;
  typedef typename Base::Event_ty Event_ty;

public: 
  /**
   * Instantiates a new Output.
   */
  Output(size_t id, des::BasicPort& impl)
    : Input<S> (id, impl) {}

  virtual Output* clone () const {
    return new Output (*this);
  }

  /**
   * A string representation
   */
  virtual std::string str () const {
    std::ostringstream ss;
    ss << "Output: " << Base::Base::str ();
    return ss.str ();
  }

protected:

  /**
   * Output just receives events and updates its state, does not send out any events
   */
  virtual void execEventIntern (const Event_ty& event, 
      typename Base::SendWrapper& sendWrap, 
      typename Base::BaseOutDegIter& b, typename Base::BaseOutDegIter& e) {

    if (event.getType () != Event_ty::NULL_EVENT) {

      const des::LogicUpdate& lu = event.getAction ();
      if (lu.getNetName () == Base::getImpl ().getInputName ()) {
        Base::getImpl ().applyUpdate (lu);
      } else {
        Base::getImpl ().netNameMismatch (lu);
      }
    }

  }

};

} // end namespace des

#endif /* DES_OUTPUT_H_ */
