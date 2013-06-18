/** Simple Spin Lock -*- C++ -*-
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
 * This contains the basic spinlock used in Galois.  We use a
 * test-and-test-and-set approach, with pause instructions on x86 and
 * compiler barriers on unlock.
 *
 * @author Andrew Lenharth <andrew@lenharth.org>
 */
#ifndef GALOIS_RUNTIME_LL_SIMPLE_LOCK_H
#define GALOIS_RUNTIME_LL_SIMPLE_LOCK_H

#include "Galois/config.h"
#include "Galois/Runtime/ll/CompilerSpecific.h"

#include <cassert>
#include GALOIS_CXX11_STD_HEADER(atomic)

namespace Galois {
namespace Runtime {
namespace LL {

/// SimpleLock is a spinlock.  If the template parameter is
/// false, the lock is a noop.
/// Copying a lock is unsynchronized (relaxed ordering)

template<bool isALock>
class SimpleLock;

template<>
class SimpleLock<true> {
  mutable std::atomic<int> _lock;
  GALOIS_ATTRIBUTE_NOINLINE
  void slow_lock() const {
    int oldval = 0;
    do {
      while (_lock.load(std::memory_order_acquire) != 0) {
	asmPause();
      }
      oldval = 0;
    } while (!_lock.compare_exchange_weak(oldval, 1, std::memory_order_acq_rel, std::memory_order_relaxed));
    assert(_lock);
  }

public:
  SimpleLock() : _lock(0) {  }
  //relaxed order for copy
  SimpleLock(const SimpleLock& p) :_lock(p._lock.load(std::memory_order_relaxed)) {}

  SimpleLock& operator= (const SimpleLock& p) {
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
    assert(_lock);
    return;
  slow_path:
    slow_lock();
  }

  inline void unlock() const {
    assert(_lock);
    //HMMMM
    _lock.store(0,std::memory_order_release);
    //_lock = 0;
  }

  inline bool try_lock() const {
    int oldval = 0;
    if (_lock.load(std::memory_order_acquire))
      return false;
    if (!_lock.compare_exchange_weak(oldval, 1, std::memory_order_acq_rel))
      return false;
    assert(_lock);
    return true;
  }

  inline bool is_locked() const {
    return _lock.load(std::memory_order_acquire) & 1;
  }
};

template<>
class SimpleLock<false> {
public:
  inline void lock() const {}
  inline void unlock() const {}
  inline bool try_lock() const { return true; }
  inline bool is_locked() const { return false; }
};


void LockPairOrdered(SimpleLock<true>& L1, SimpleLock<true>& L2);
bool TryLockPairOrdered(SimpleLock<true>& L1, SimpleLock<true>& L2);
void UnLockPairOrdered(SimpleLock<true>& L1, SimpleLock<true>& L2);
void LockPairOrdered(SimpleLock<false>& L1, SimpleLock<false>& L2);
bool TryLockPairOrdered(SimpleLock<false>& L1, SimpleLock<false>& L2);
void UnLockPairOrdered(SimpleLock<false>& L1, SimpleLock<false>& L2);

}
}
} // end namespace Galois

#endif
