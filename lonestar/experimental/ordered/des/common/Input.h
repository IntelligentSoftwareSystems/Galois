/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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
 */

#ifndef DES_INPUT_H_
#define DES_INPUT_H_

#include <string>
#include <cassert>

#include "SimGate.h"
#include "BasicPort.h"

namespace des {

template <typename S>
class Input : public SimGate<S> {

protected:
  typedef SimGate<S> Base;
  typedef typename Base::Event_ty Event_ty;

public:
  /**
   * Instantiates a new Input.
   */
  Input(size_t id, des::BasicPort& impl) : Base(id, impl) {}

  virtual Input* clone() const { return new Input(*this); }

  virtual des::BasicPort& getImpl() const {
    assert(dynamic_cast<des::BasicPort*>(&Base::getImpl()) != NULL);
    des::BasicPort* ptr = static_cast<des::BasicPort*>(&Base::getImpl());
    assert(ptr != NULL);
    return *ptr;
  }

  /**
   * A string representation
   */
  virtual std::string str() const {
    std::ostringstream ss;
    ss << "Input: " << Base::str();
    return ss.str();
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
  virtual void execEventIntern(const Event_ty& event,
                               typename Base::SendWrapper& sendWrap,
                               typename Base::BaseOutDegIter& b,
                               typename Base::BaseOutDegIter& e) {

    if (event.getType() == Event_ty::NULL_EVENT) {
      Base::execEventIntern(event, sendWrap, b, e);

    } else {

      const des::LogicUpdate& lu = event.getAction();
      if (getImpl().getInputName() == lu.getNetName()) {
        getImpl().setInputVal(lu.getNetVal());
        getImpl().setOutputVal(lu.getNetVal());

        des::LogicUpdate drvFanout(getImpl().getOutputName(),
                                   getImpl().getOutputVal());

        Base::sendEventsToFanout(event, drvFanout, Event_ty::REGULAR_EVENT,
                                 sendWrap, b, e);
      } else {
        getImpl().netNameMismatch(lu);
      }
    }
  }
};

} // end namespace des
#endif /* DES_INPUT_H_ */
