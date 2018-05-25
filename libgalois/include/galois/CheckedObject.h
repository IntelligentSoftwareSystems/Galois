/**
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of XYZ License (a copy is located in
 * LICENSE.txt at the top-level directory).
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

#ifndef GALOIS_CHECKEDOBJECT_H
#define GALOIS_CHECKEDOBJECT_H

#include "galois/runtime/Context.h"

namespace galois {

/**
 * Conflict-checking wrapper for any type.  Performs global conflict detection
 * on the enclosed object.  This enables arbitrary types to be managed by the
 * Galois runtime.
 */
template<typename T>
class GChecked : public galois::runtime::Lockable {
  T val;

public:
  template<typename... Args>
  GChecked(Args&&... args): val(std::forward<Args>(args)...) { }

  T& get(galois::MethodFlag m = MethodFlag::WRITE) {
    galois::runtime::acquire(this, m);
    return val;
  }

  const T& get(galois::MethodFlag m = MethodFlag::WRITE) const {
    galois::runtime::acquire(const_cast<GChecked*>(this), m);
    return val;
  }
};

template<>
class GChecked<void>: public galois::runtime::Lockable {
public:
  void get(galois::MethodFlag m = MethodFlag::WRITE) const {
    galois::runtime::acquire(const_cast<GChecked*>(this), m);
  }
};

}

#endif // _GALOIS_CHECKEDOBJECT_H
