/** Pointer Spin Lock -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in
 * irregular programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights
 * reserved.  UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES
 * CONCERNING THIS SOFTWARE AND DOCUMENTATION, INCLUDING ANY
 * WARRANTIES OF MERCHANTABILITY, FITNESS FOR ANY PARTICULAR PURPOSE,
 * NON-INFRINGEMENT AND WARRANTIES OF PERFORMANCE, AND ANY WARRANTY
 * THAT MIGHT OTHERWISE ARISE FROM COURSE OF DEALING OR USAGE OF
 * TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH RESPECT TO
 * THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect,
 * direct or consequential damages or loss of profits, interruption of
 * business, or related expenses which may arise from use of Software
 * or Documentation, including but not limited to those resulting from
 * defects in Software and/or Documentation, or loss or inaccuracy of
 * data of any kind.  
 *
 * @section Description
 *
 * This contains the pointer-based spinlock used in Galois.  We use a
 * test-and-test-and-set approach, with pause instructions on x86 and
 * compiler barriers on unlock.  A pointer-lock uses the low-order bit
 * in a pointer to store the lock, thus assumes a non-one-byte
 * alignment.
 *
 * @author Andrew Lenharth <andrew@lenharth.org>
 */

#ifndef GALOIS_RUNTIME_LL_PTRLOCK_H
#define GALOIS_RUNTIME_LL_PTRLOCK_H

#include <stdint.h>
#include <cassert>

#include "CompilerSpecific.h"

namespace GaloisRuntime {
namespace LL {

/// PtrLock is a spinlock and a pointer.  If the second template
/// parameter is false, the lock is a noop.  This wraps a pointer and
/// uses the low order bit for the lock flag
template<typename T, bool isALock>
class PtrLock;

template<typename T>
class PtrLock<T, true> {
  volatile uintptr_t _lock;
public:
  PtrLock() : _lock() {}

  inline void lock() {
    uintptr_t oldval;
    do {
      while ((_lock & 1) != 0) {
	asmPause();
      }
      oldval = __sync_fetch_and_or(&_lock, 1);
    } while (oldval & 1);
  }

  inline void unlock() {
    assert(_lock & 1);
    compilerBarrier();
    _lock = _lock & ~(uintptr_t)1;
  }

  inline void unlock_and_clear() {
    assert(_lock & 1);
    compilerBarrier();
    _lock = 0;
  }

  inline void unlock_and_set(T* val) {
    assert(_lock & 1);
    assert(!((uintptr_t)val & 1));
    compilerBarrier();
    _lock = (uintptr_t)val;
  }
  
  inline T* getValue() const {
    return (T*)(_lock & ~(uintptr_t)1);
  }

  inline void setValue(T* val) {
    uintptr_t nval = (uintptr_t)val;
    nval |= (_lock & 1);
    _lock = nval;
  }

  inline bool try_lock() {
    uintptr_t oldval = _lock;
    if ((oldval & 1) != 0)
      return false;
    oldval = __sync_fetch_and_or(&_lock, 1);
    return !(oldval & 1);
  }

  //! CAS only works on unlocked values
  //! the lock bit will prevent a successful cas
  inline bool CAS(T* oldval, T* newval) {
    assert(!((uintptr_t)oldval & 1) && !((uintptr_t)newval & 1));
    return __sync_bool_compare_and_swap(&_lock, (uintptr_t)oldval, (uintptr_t)newval);
  }

  //! CAS that works on locked values; this can be very dangerous
  //! when used incorrectly
  inline bool stealing_CAS(T* oldval, T* newval) {
    return __sync_bool_compare_and_swap(&_lock, (uintptr_t)oldval|1, (uintptr_t)newval|1);
  }
};

template<typename T>
class PtrLock<T, false> {
  T* _lock;
public:
  PtrLock() : _lock() {}
  
  inline void lock() {}
  inline void unlock() {}
  inline void unlock_and_clear() { _lock = 0; }
  inline void unlock_and_set(T* val) { _lock = val; }
  inline T* getValue() const { return _lock; }
  inline void setValue(T* val) { _lock = val; }
  inline bool try_lock() { return true; }
  inline bool CAS(T* oldval, T* newval) {
    if (_lock == oldval) {
      _lock = newval;
      return true;
    }
    return false;
  }
  inline bool stealing_CAS(T* oldval, T* newval) {
    return CAS(oldval, newval);
  }
};

}
}

#endif
