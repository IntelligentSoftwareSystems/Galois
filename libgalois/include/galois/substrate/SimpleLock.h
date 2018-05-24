/** Simple Spin Lock -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galois is a framework to exploit
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
 * This contains the basic spinlock used in Galois.  We use a
 * test-and-test-and-set approach, with pause instructions on x86.
 * This implements C++11 lockable and try lockable concept
 *
 * @author Andrew Lenharth <andrew@lenharth.org>
 */
#ifndef GALOIS_SUBSTRATE_SIMPLELOCK_H
#define GALOIS_SUBSTRATE_SIMPLELOCK_H

#include "galois/substrate/CompilerSpecific.h"

#include <cassert>
#include <atomic>
#include <mutex>

namespace galois {
namespace substrate {

/// SimpleLock is a spinlock.
/// Copying a lock is unsynchronized (relaxed ordering)

class SimpleLock {
  mutable std::atomic<int> _lock;
  void slow_lock() const;

public:
  constexpr SimpleLock(): _lock(0) { }
  //relaxed order for copy
  SimpleLock(const SimpleLock& p): _lock(p._lock.load(std::memory_order_relaxed)) { }

  SimpleLock& operator=(const SimpleLock& p) {
    if (&p == this) return *this;
    //relaxed order for initialization
    _lock.store(p._lock.load(std::memory_order_relaxed), std::memory_order_relaxed);
    return *this;
  }

  inline void lock() const {
    int oldval = 0;
    if (_lock.load(std::memory_order_relaxed))
      goto slow_path;
    if (!_lock.compare_exchange_weak(oldval, 1, std::memory_order_acq_rel, std::memory_order_relaxed))
      goto slow_path;
    assert(is_locked());
    return;
  slow_path:
    slow_lock();
  }

  inline void unlock() const {
    assert(is_locked());
    //HMMMM
    _lock.store(0, std::memory_order_release);
    //_lock = 0;
  }

  inline bool try_lock() const {
    int oldval = 0;
    if (_lock.load(std::memory_order_relaxed))
      return false;
    if (!_lock.compare_exchange_weak(oldval, 1, std::memory_order_acq_rel))
      return false;
    assert(is_locked());
    return true;
  }

  inline bool is_locked() const {
    return _lock.load(std::memory_order_acquire) & 1;
  }
};

//! Dummy Lock implements the lock interface without a lock for serial code

namespace internal {
class DummyLock {
public:
  inline void lock() const {}
  inline void unlock() const {}
  inline bool try_lock() const { return true; }
  inline bool is_locked() const { return false; }
};
}


template <bool Enabled>
using CondLock = typename std::conditional<Enabled, SimpleLock, internal::DummyLock>::type;


using lock_guard_galois =  std::lock_guard<SimpleLock>;

#define MAKE_LOCK_GUARD(__x) galois::substrate::lock_guard_galois locker##___COUNTER__(__x)

} // end namespace substrate
} // end namespace galois

#endif
