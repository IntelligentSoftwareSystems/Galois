/** Simple Spin Lock -*- C++ -*-
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
 * This contains the basic spinlock used in Galois.  We use a
 * test-and-test-and-set approach, with pause instructions on x86.
 * This implements C++11 lockable and try lockable concept
 *
 * @author Andrew Lenharth <andrew@lenharth.org>
 */

#ifndef GALOIS_RUNTIME_SIMPLELOCK_H
#define GALOIS_RUNTIME_SIMPLELOCK_H

#include "Galois/Runtime/CompilerSpecific.h"

#include <cassert>
#include <atomic>

namespace Galois {
namespace Runtime {

/// SimpleLock is a spinlock.
/// Copying a lock is unsynchronized (relaxed ordering)

class SimpleLock {
  mutable std::atomic<int> _lock;
  void slow_lock() const __attribute__((hot)) ;

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
    if (_lock.load(std::memory_order_relaxed))
      goto slow_path;
    if (_lock.fetch_or(1, std::memory_order_acq_rel))
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
    if (_lock.load(std::memory_order_relaxed))
      return false;
    if (_lock.fetch_or(1, std::memory_order_acq_rel))
      return false;
    assert(is_locked());
    return true;
  }

  inline bool is_locked() const {
    return _lock.load(std::memory_order_acquire) & 1;
  }
};

} // end namespace Runtime
} // end namespace Galois

#endif
