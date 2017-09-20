/** Simple Safe Static Global Instance -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, version 2.1 of the
 * License.
 *
 * Galois is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Galois.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @section Copyright
 *
 * Copyright (C) 2015, The University of Texas at Austin. All rights
 * reserved.
 *
 * @section Description
 *
 * This contains a wrapper to declare non-pod globals in a safe way.
 *
 * @author Andrew Lenharth <andrew@lenharth.org>
 */

#ifndef GALOIS_SUBSTRATE_STATICINSTANCE_H
#define GALOIS_SUBSTRATE_STATICINSTANCE_H

#include "Galois/Substrate/CompilerSpecific.h"

namespace galois {
namespace Substrate {

//This should be much simpler in c++03 mode, but be general for now
//This exists because ptrlock is not a pod, but this is.
template<typename T>
struct StaticInstance {
  volatile T* V;
  volatile int _lock;

  inline void lock() {
    int oldval;
    do {
      while (_lock != 0) {
        Substrate::asmPause();
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

} // end namespace Substrate
} // end namespace galois

#endif
