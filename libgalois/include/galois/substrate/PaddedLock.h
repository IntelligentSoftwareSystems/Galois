/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
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

#ifndef GALOIS_SUBSTRATE_PADDEDLOCK_H
#define GALOIS_SUBSTRATE_PADDEDLOCK_H

#include "galois/substrate/SimpleLock.h"
#include "galois/substrate/CacheLineStorage.h"

namespace galois {
namespace substrate {

/// PaddedLock is a spinlock.  If the second template parameter is
/// false, the lock is a noop.
template <bool concurrent>
class PaddedLock;

template <>
class PaddedLock<true> {
  mutable CacheLineStorage<SimpleLock> Lock;

public:
  void lock() const { Lock.get().lock(); }
  bool try_lock() const { return Lock.get().try_lock(); }
  void unlock() const { Lock.get().unlock(); }
};

template <>
class PaddedLock<false> {
public:
  void lock() const {}
  bool try_lock() const { return true; }
  void unlock() const {}
};

} // end namespace substrate
} // end namespace galois

#endif
