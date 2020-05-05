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

#ifndef DES_SIMGATE_H
#define DES_SIMGATE_H

#include <string>
#include <iostream>

#include <cstdlib>
#include <cassert>

#include "logicDefs.h"
#include "LogicUpdate.h"
#include "LogicGate.h"

#include "Event.h"
#include "SimObject.h"

namespace des {
/**
 * The Class SimGate represents an abstract logic gate.
 */
template <typename S>
class SimGate : public S {

protected:
  typedef S Base;
  des::LogicGate& impl;

public:
  typedef Event<LogicUpdate> Event_ty;

  SimGate(size_t id, des::LogicGate& impl)
      : Base(id, impl.getNumOutputs(), impl.getNumInputs()), impl(impl) {}

  SimGate(const SimGate& that) : Base(that), impl(that.impl) {}

  virtual SimGate* clone() const { return new SimGate(*this); }

  virtual des::LogicGate& getImpl() const { return impl; }

  virtual size_t getInputIndex(const Event_ty& event) const {
    assert(dynamic_cast<SimGate*>(event.getRecvObj()) == this);

    const std::string& netName = event.getAction().getNetName();
    return impl.getInputIndex(netName);
  }

  /**
   * A string representation
   */
  virtual std::string str() const {
    std::ostringstream ss;
    ss << Base::str() << ": " << impl.str();
    return ss.str();
  }

protected:
  /**
   * Send events to fanout, which are the out going neighbors in the circuit
   * graph.
   */
  void sendEventsToFanout(const Event_ty& inputEvent,
                          const des::LogicUpdate& msg,
                          const Event_ty::Type& type,
                          typename Base::SendWrapper& sendWrap,
                          typename Base::BaseOutDegIter& b,
                          typename Base::BaseOutDegIter& e) {

    assert(dynamic_cast<SimGate*>(this) != NULL);
    SimGate* srcGate = static_cast<SimGate*>(this);

    const des::SimTime& sendTime = inputEvent.getRecvTime();

    for (; b != e; ++b) {

      assert(dynamic_cast<SimGate*>(*b) != NULL);
      SimGate* dstGate = static_cast<SimGate*>(*b);

      Event_ty ne =
          srcGate->makeEvent(dstGate, msg, type, sendTime, impl.getDelay());

      sendWrap.send(dstGate, ne);
    }
  }

  virtual void execEventIntern(const Event_ty& event,
                               typename Base::SendWrapper& sendWrap,
                               typename Base::BaseOutDegIter& b,
                               typename Base::BaseOutDegIter& e) {

    if (event.getType() != Event_ty::NULL_EVENT) {
      // update the inputs of fanout gates
      const des::LogicUpdate& lu = event.getAction();

      impl.applyUpdate(lu);

    } // else output is unchanged in case of NULL_EVENT

    des::LogicUpdate drvFanout(impl.getOutputName(), impl.getOutputVal());

    sendEventsToFanout(event, drvFanout, event.getType(), sendWrap, b, e);
  }

public:
  virtual size_t getStateSize() const { return impl.getStateSize(); }

  virtual void copyState(void* const buf, const size_t bufSize) const {
    impl.copyState(buf, bufSize);
  }

  virtual void restoreState(void* const buf, const size_t bufSize) {
    impl.restoreState(buf, bufSize);
  }
};

} // end namespace des
#endif // DES_SIMGATE_H
