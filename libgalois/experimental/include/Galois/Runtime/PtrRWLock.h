/** Pointer RW Spin Lock -*- C++ -*-
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
 * This contains the pointer-based reader writer spinlock used in
 * Galois.  We use a test-and-test-and-set approach, with pause
 * instructions on x86 and compiler barriers on unlock.  A
 * pointer-lock uses the low-order bit in a pointer to store the lock,
 * thus assumes a non-one-byte alignment.  The read lock is stored as
 * the second bit (mutually exclusive with the write lock) and the
 * counter stored in the remaining bits.  The pointer is only stored
 * for exclusive locks.
 *
 * @author Andrew Lenharth <andrew@lenharth.org>
 */
#ifndef GALOIS_RUNTIME_LL_PTRRWLOCK_H
#define GALOIS_RUNTIME_LL_PTRRWLOCK_H

//#include "Galois/config.h"
#include "Galois/Substrate/CompilerSpecific.h"

#include <stdint.h>
#include <cassert>
#include <atomic>

namespace galois {
namespace Runtime {
namespace LL {

/// PtrLock is a spinlock and a pointer. 
/// This wraps a pointer and
/// uses the low order bit for the lock flag
/// Copying a lock is unsynchronized (relaxed ordering)

namespace details {

class PtrRWLockBase {
  std::atomic<uintptr_t> _lock;

  void slow_lock();
  void slow_lock_shared();

protected:
  constexpr PtrRWLockBase() : _lock(0) {}
  //relaxed order for copy
  PtrRWLockBase(const PtrRWLockBase& p) : _lock(p._lock.load(std::memory_order_relaxed)) {}

  uintptr_t _getValue() const {
    return _lock;
  }

  void _setValue(uintptr_t val) {
    val |= 1;
    uintptr_t t;
    do {
      t = _lock;
      if (t & 2) return;
      if (!(t & 1)) return;
    } while (!_lock.compare_exchange_weak(t, val, std::memory_order_acq_rel, std::memory_order_relaxed));
  }

public:

  PtrRWLockBase& operator=(const PtrRWLockBase& p) {
    if (&p == this) return *this;
    //relaxed order for initialization
    _lock.store(p._lock.load(std::memory_order_relaxed), std::memory_order_relaxed);
    return *this;
  }

  inline void lock() {
    uintptr_t oldval = _lock.load(std::memory_order_relaxed);
    if (oldval & 3)
      goto slow_path;
    oldval &= ~(uintptr_t)3;
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
    if ((oldval & 3) != 0)
      return false;
    oldval = 0;
    return _lock.compare_exchange_weak(oldval, 1, std::memory_order_acq_rel, std::memory_order_relaxed);
  }

  inline bool is_locked() const {
    return _lock.load(std::memory_order_acquire) & 1;
  }

  inline bool is_locked_shared() const {
    return _lock.load(std::memory_order_acquire) & 2;
  }
    

  inline void lock_shared() {
    uintptr_t oldval = _lock.load(std::memory_order_relaxed);
    uintptr_t newval;
    if (oldval & 1)
      goto slow_path;
    newval = oldval;
    newval &= ~(uintptr_t)1;
    newval |= 2;
    newval += 4;
    if (!_lock.compare_exchange_weak(oldval, newval, std::memory_order_acq_rel, std::memory_order_relaxed))
      goto slow_path;
    assert(is_locked_shared());
    return;

  slow_path:
    slow_lock_shared();
  }

  inline bool try_lock_shared() {
    uintptr_t oldval, newval;
    do {
      oldval = _lock.load(std::memory_order_relaxed);
      if (oldval & 1)
        return false;
      newval = oldval;
      newval &= ~(uintptr_t)1;
      newval |= 2;
      newval += 4;
    } while (!_lock.compare_exchange_weak(oldval, newval, std::memory_order_acq_rel, std::memory_order_relaxed));
    assert(is_locked_shared());
    return true;
  }   

  inline void unlock_shared() {
    uintptr_t oldval, newval;
    do {
      oldval = _lock.load(std::memory_order_relaxed);
      newval = oldval - 4;
      if (newval >> 2 == 0)
        newval &= ~(uintptr_t)2;
    } while (!_lock.compare_exchange_weak(oldval, newval, std::memory_order_acq_rel, std::memory_order_relaxed));
  }
};

} // namespace details

template<typename T>
class PtrRWLock : public details::PtrRWLockBase {
  //  static_assert(alignof(T) > 2, "Insufficient spare bits");
public:
  constexpr PtrRWLock() : PtrRWLockBase() {}
  //relaxed order for copy
  PtrRWLock(const PtrRWLock& p) : PtrRWLockBase(p) {}

  T* getValue() const {
    uintptr_t val = _getValue();
    if (val & 2)
      return nullptr;
    return (T*)(val & ~(uintptr_t)1);
  }

  //only works for locked values
  void setValue(T* ptr) {
    uintptr_t val = (uintptr_t)ptr;
    _setValue(val | 1);
  }
};

}
}
} // end namespace galois

#endif
