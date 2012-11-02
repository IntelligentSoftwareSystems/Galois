/** Cache-line padded Simple Spin Lock -*- C++ -*-
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
 * This contains the basic spinlock padded and aligned to use a cache
 * line.
 *
 * @author Andrew Lenharth <andrew@lenharth.org>
 */

#ifndef GALOIS_RUNTIME_LL_PADDED_LOCK_H
#define GALOIS_RUNTIME_LL_PADDED_LOCK_H

#include "SimpleLock.h"
#include "CacheLineStorage.h"

namespace GaloisRuntime {
namespace LL {

/// PaddedLock is a spinlock.  If the second template parameter is
/// false, the lock is a noop.
template<bool concurrent>
class PaddedLock;

void LockPairOrdered(PaddedLock<true>& L1, PaddedLock<true>& L2); 
bool TryLockPairOrdered(PaddedLock<true>& L1, PaddedLock<true>& L2); 
void UnLockPairOrdered(PaddedLock<true>& L1, PaddedLock<true>& L2);
void LockPairOrdered(PaddedLock<false>& L1, PaddedLock<false>& L2);
bool TryLockPairOrdered(PaddedLock<false>& L1, PaddedLock<false>& L2);
void UnLockPairOrdered(PaddedLock<false>& L1, PaddedLock<false>& L2);

template<>
class PaddedLock<true> {
  mutable CacheLineStorage<SimpleLock<true> > Lock;

public:
  void lock() const { Lock.data.lock(); }
  bool try_lock() const { return Lock.data.try_lock(); }
  void unlock() const { Lock.data.unlock(); }
  friend void LockPairOrdered(PaddedLock<true>& L1, PaddedLock<true>& L2);
  friend bool TryLockPairOrdered(PaddedLock<true>& L1, PaddedLock<true>& L2);
  friend void UnLockPairOrdered(PaddedLock<true>& L1, PaddedLock<true>& L2);
};

template<>
class PaddedLock<false> {
public:
  void lock() const {}
  bool try_lock() const { return true; }
  void unlock() const {}
};

}
}

#endif
