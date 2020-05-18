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

#ifndef GALOIS_SUBSTRATE_STATICINSTANCE_H
#define GALOIS_SUBSTRATE_STATICINSTANCE_H

#include "galois/config.h"
#include "galois/substrate/CompilerSpecific.h"

namespace galois {
namespace substrate {

// This should be much simpler in c++03 mode, but be general for now
// This exists because ptrlock is not a pod, but this is.
template <typename T>
struct StaticInstance {
  volatile T* V;
  volatile int _lock;

  inline void lock() {
    int oldval;
    do {
      while (_lock != 0) {
        substrate::asmPause();
      }
      oldval = __sync_fetch_and_or(&_lock, 1);
    } while (oldval & 1);
  }

  inline void unlock() {
    compilerBarrier();
    _lock = 0;
  }

  T* get() {
    volatile T* val = V;
    if (val)
      return (T*)val;
    lock();
    val = V;
    if (!val)
      V = val = new T();
    unlock();
    return (T*)val;
  }
};

} // end namespace substrate
} // end namespace galois

#endif
