/** Pointer Spin Lock -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in
 * irregular programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights
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

#include "Galois/config.h"
#include "Galois/Runtime/ll/CompilerSpecific.h"

#include <stdint.h>
#include <cassert>
#include GALOIS_CXX11_STD_HEADER(atomic)

namespace Galois {
namespace Runtime {
namespace LL {

/// PtrLock is a spinlock and a pointer. 
/// This wraps a pointer and
/// uses the low order bit for the lock flag
/// Copying a lock is unsynchronized (relaxed ordering)

namespace details {

class PtrLockBase {
  std::atomic<uintptr_t> _lock;

  void slow_lock();

protected:

  //! CAS only works on unlocked values
  //! the lock bit will prevent a successful cas
  inline bool _CAS(uintptr_t oldval, uintptr_t newval) {
    assert(!(oldval & 1) && !(newval & 1));
    uintptr_t old = oldval;
    return _lock.compare_exchange_strong(old, newval);
  }

  //! CAS that works on locked values; this can be very dangerous
  //! when used incorrectly
  inline bool _stealing_CAS(uintptr_t oldval, uintptr_t newval) {
    uintptr_t old = 1 | oldval;
    return _lock.compare_exchange_strong(old, 1 | newval);
  }

  inline void _unlock_and_set(uintptr_t val) {
    assert(is_locked());
    assert(!(val & 1));
    _lock.store( val, std::memory_order_release);
  }
  
  inline uintptr_t _getValue() const {
    return _lock.load(std::memory_order_relaxed) & ~(uintptr_t)1;
  }

  inline void _setValue(uintptr_t val) {
    val |= (_lock & 1);
    //relaxed OK since this doesn't clear lock
    _lock.store(val, std::memory_order_relaxed);
  }


  constexpr PtrLockBase() : _lock(0) {}
  //relaxed order for copy
  PtrLockBase(const PtrLockBase& p) : _lock(p._lock.load(std::memory_order_relaxed)) {}

public:

  PtrLockBase& operator=(const PtrLockBase& p) {
    if (&p == this) return *this;
    //relaxed order for initialization
    _lock.store(p._lock.load(std::memory_order_relaxed), std::memory_order_relaxed);
    return *this;
  }

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

} // namespace details

template<typename T>
class PtrLock : public details::PtrLockBase {
public:
  constexpr PtrLock() : PtrLockBase() {}
  //relaxed order for copy
  PtrLock(const PtrLock& p) : PtrLockBase(p) {}

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

template<typename T>
class DummyPtrLock {
  T* ptr;
public:
  T* getValue() const { return ptr; }
  void setValue(T* v) { ptr = v; }
  void lock() {}
  void unlock() {}
  void unlock_and_clear() { ptr = nullptr; }
  void unlock_and_set(T* v) { ptr = v; }
  bool CAS(T* oldval, T* newval) {
    if (ptr == oldval) {
      ptr = oldval;
      return true;
    }
    return false;
  }
};

}
}
} // end namespace Galois

#endif
