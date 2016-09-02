/** Pointer Spin Lock -*- C++ -*-
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
 * Copyright (C) 2016, The University of Texas at Austin. All rights
 * reserved.
 *
 * @section Description
 *
 * This contains the pointer-based spinlock used in Galois.  We use a
 * test-and-test-and-set approach, with pause instructions on x86 and
 * compiler barriers on unlock.  A pointer-lock uses the low-order bit
 * in a pointer to store the lock.
 *
 * @author Andrew Lenharth <andrew@lenharth.org>
 */

#ifndef GALOIS_RUNTIME_PTRLOCK_H
#define GALOIS_RUNTIME_PTRLOCK_H

#include "Galois/Runtime/CompilerSpecific.h"

#include <stdint.h>
#include <cassert>
#include <atomic>

namespace Galois {
namespace Runtime {

namespace detail { // separated out because of template type

class PtrLockBase {
  mutable std::atomic<uintptr_t> _lock;

  void slow_lock() const;

protected:
  constexpr PtrLockBase() : _lock(0) {}
  //relaxed order for copy
  PtrLockBase(const PtrLockBase& p) : _lock(p._lock.load(std::memory_order_relaxed)) {}

  void copy(const PtrLockBase* p) {
    //relaxed order for initialization
    _lock.store(p->_lock.load(std::memory_order_relaxed), std::memory_order_relaxed);
  }

  inline void _unlock_and_set(uintptr_t val) {
    assert(is_locked());
    assert(!(val & 1));
    _lock.store(val, std::memory_order_release);
  }
  
  inline uintptr_t _getValue() const {
    return _lock.load(std::memory_order_relaxed) & ~(uintptr_t)1;
  }

  inline void _setValue(uintptr_t val) {
    uintptr_t nval = (uintptr_t)val;
    nval |= (_lock & 1);
    //relaxed OK since this doesn't clear lock
    _lock.store(nval, std::memory_order_relaxed);
  }

  //! CAS only works on unlocked values
  //! the lock bit will prevent a successful cas
  inline bool _CAS(uintptr_t oldval, uintptr_t newval) {
    assert(!(oldval & 1) && !(newval & 1));
    return _lock.compare_exchange_strong(oldval, newval);
  }

  //! CAS that works on locked values; this can be very dangerous
  //! when used incorrectly
  inline bool _stealing_CAS(uintptr_t oldval, uintptr_t newval) {
    uintptr_t old = 1 | oldval;
    return _lock.compare_exchange_strong(old, 1 | newval);
  }

public:

  inline void lock() {
    uintptr_t oldval = _lock.load(std::memory_order_relaxed);
    if (oldval & 1)
      goto slow_path;
    if (!_lock.compare_exchange_weak(oldval, oldval | 1, std::memory_order_acq_rel, std::memory_order_relaxed))
      goto slow_path;
    assert(is_locked());
    return;

  slow_path:
    slow_lock();
  }

  inline void unlock() {
    assert(is_locked());
    _lock.store(_lock.load(std::memory_order_relaxed) & ~(uintptr_t)1, std::memory_order_release);
  }

  inline void unlock_and_clear() {
    assert(is_locked());
    _lock.store(0, std::memory_order_release);
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

};

}//namespace detail

/// PtrLock is a spinlock and a pointer.  This wraps a pointer and
/// uses the low order bit for the lock flag Copying a lock is
/// unsynchronized (relaxed ordering)

template<typename T>
class PtrLock : public detail::PtrLockBase {
  //  static_assert(alignof(T) > 1, "Bad data type alignment for PtrLock");

public:
  constexpr PtrLock() :PtrLockBase() {}
  //relaxed order for copy
  PtrLock(const PtrLock& p) :PtrLockBase(p) {}

  PtrLock& operator=(const PtrLock& p) {
    if (&p == this) return *this;
    copy(&p);
    return *this;
  }

  inline void unlock_and_set(T* val) {
    _unlock_and_set((uintptr_t)val);
  }
  
  inline T* getValue() const {
    return (T*)_getValue();
  }

  inline void setValue(T* val) {
    _setValue((uintptr_t)val);
  }

  //! CAS only works on unlocked values
  //! the lock bit will prevent a successful cas
  inline bool CAS(T* oldval, T* newval) {
    return _CAS((uintptr_t)oldval, (uintptr_t)newval);
  }

  //! CAS that works on locked values; this can be very dangerous
  //! when used incorrectly
  inline bool stealing_CAS(T* oldval, T* newval) {
    return _stealing_CAS((uintptr_t)oldval, (uintptr_t)newval);
  }
};

} // end namespace Runtime
} // end namespace Galois

#endif
