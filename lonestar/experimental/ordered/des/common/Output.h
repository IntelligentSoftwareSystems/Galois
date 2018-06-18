/**
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of XYZ License (a
 * copy is located in LICENSE.txt at the top-level directory).
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
class Output : public Input<S> {

  typedef Input<S> Base;
  typedef typename Base::Event_ty Event_ty;

public:
  /**
   * Instantiates a new Output.
   */
  Output(size_t id, des::BasicPort& impl) : Input<S>(id, impl) {}

  virtual Output* clone() const { return new Output(*this); }

  /**
   * A string representation
   */
  virtual std::string str() const {
    std::ostringstream ss;
    ss << "Output: " << Base::Base::str();
    return ss.str();
  }

protected:
  /**
   * Output just receives events and updates its state, does not send out any
   * events
   */
  virtual void execEventIntern(const Event_ty& event,
                               typename Base::SendWrapper& sendWrap,
                               typename Base::BaseOutDegIter& b,
                               typename Base::BaseOutDegIter& e) {

    if (event.getType() != Event_ty::NULL_EVENT) {

      const des::LogicUpdate& lu = event.getAction();
      if (lu.getNetName() == Base::getImpl().getInputName()) {
        Base::getImpl().applyUpdate(lu);
      } else {
        Base::getImpl().netNameMismatch(lu);
      }
    }
  }
};

} // end namespace des

#endif /* DES_OUTPUT_H_ */
