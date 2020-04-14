/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
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

#ifndef GALOIS_SUBSTRATE_PTRLOCK_H
#define GALOIS_SUBSTRATE_PTRLOCK_H

#include <cstdint>
#include <cassert>
#include <atomic>

#include "galois/config.h"
#include "galois/substrate/CompilerSpecific.h"

namespace galois {
namespace substrate {

namespace internal {
void ptr_slow_lock(std::atomic<uintptr_t>& l);
}

/// PtrLock is a spinlock and a pointer.  This wraps a pointer and
/// uses the low order bit for the lock flag Copying a lock is
/// unsynchronized (relaxed ordering)

template <typename T>
class PtrLock {
  std::atomic<uintptr_t> _lock;

  //  static_assert(alignof(T) > 1, "Bad data type alignment for PtrLock");

public:
  constexpr PtrLock() : _lock(0) {}
  // relaxed order for copy
  PtrLock(const PtrLock& p) : _lock(p._lock.load(std::memory_order_relaxed)) {}

  PtrLock& operator=(const PtrLock& p) {
    if (&p == this)
      return *this;
    // relaxed order for initialization
    _lock.store(p._lock.load(std::memory_order_relaxed),
                std::memory_order_relaxed);
    return *this;
  }

  inline void lock() {
    uintptr_t oldval = _lock.load(std::memory_order_relaxed);
    if (oldval & 1)
      goto slow_path;
    if (!_lock.compare_exchange_weak(oldval, oldval | 1,
                                     std::memory_order_acq_rel,
                                     std::memory_order_relaxed))
      goto slow_path;
    assert(is_locked());
    return;

  slow_path:
    internal::ptr_slow_lock(_lock);
  }

  inline void unlock() {
    assert(is_locked());
    _lock.store(_lock.load(std::memory_order_relaxed) & ~(uintptr_t)1,
                std::memory_order_release);
  }

  inline void unlock_and_clear() {
    assert(is_locked());
    _lock.store(0, std::memory_order_release);
  }

  inline void unlock_and_set(T* val) {
    assert(is_locked());
    assert(!((uintptr_t)val & 1));
    _lock.store((uintptr_t)val, std::memory_order_release);
  }

  inline T* getValue() const {
    return (T*)(_lock.load(std::memory_order_relaxed) & ~(uintptr_t)1);
  }

  inline void setValue(T* val) {
    uintptr_t nval = (uintptr_t)val;
    nval |= (_lock & 1);
    // relaxed OK since this doesn't clear lock
    _lock.store(nval, std::memory_order_relaxed);
  }

  inline bool try_lock() {
    uintptr_t oldval = _lock.load(std::memory_order_relaxed);
    if ((oldval & 1) != 0)
      return false;
    oldval = _lock.fetch_or(1, std::memory_order_acq_rel);
    return !(oldval & 1);
  }

  inline bool is_locked() const {
    return _lock.load(std::memory_order_acquire) & 1;
  }

  //! CAS only works on unlocked values
  //! the lock bit will prevent a successful cas
  inline bool CAS(T* oldval, T* newval) {
    assert(!((uintptr_t)oldval & 1) && !((uintptr_t)newval & 1));
    uintptr_t old = (uintptr_t)oldval;
    return _lock.compare_exchange_strong(old, (uintptr_t)newval);
  }

  //! CAS that works on locked values; this can be very dangerous
  //! when used incorrectly
  inline bool stealing_CAS(T* oldval, T* newval) {
    uintptr_t old = 1 | (uintptr_t)oldval;
    return _lock.compare_exchange_strong(old, 1 | (uintptr_t)newval);
  }
};

template <typename T>
class DummyPtrLock {
  T* _lock;

public:
  DummyPtrLock() : _lock() {}

  inline void lock() {}
  inline void unlock() {}
  inline void unlock_and_clear() { _lock = 0; }
  inline void unlock_and_set(T* val) { _lock = val; }
  inline T* getValue() const { return _lock; }
  inline void setValue(T* val) { _lock = val; }
  inline bool try_lock() const { return true; }
  inline bool is_locked() const { return false; }
  inline bool CAS(T* oldval, T* newval) {
    if (_lock == oldval) {
      _lock = newval;
      return true;
    }
    return false;
  }
  inline bool stealing_CAS(T* oldval, T* newval) { return CAS(oldval, newval); }
};

} // end namespace substrate
} // end namespace galois

#endif
