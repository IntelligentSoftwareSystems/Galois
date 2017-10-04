/** Cache-line padded Simple Spin Lock -*- C++ -*-
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
 * This contains the basic spinlock padded and aligned to use a cache
 * line.  This implements c++11 lockable and try lockable concept
 *
 * @author Andrew Lenharth <andrew@lenharth.org>
 */

#ifndef GALOIS_SUBSTRATE_PADDEDLOCK_H
#define GALOIS_SUBSTRATE_PADDEDLOCK_H

#include "SimpleLock.h"
#include "CacheLineStorage.h"

namespace galois {
namespace substrate {

/// PaddedLock is a spinlock.  If the second template parameter is
/// false, the lock is a noop.
template<bool concurrent>
class PaddedLock;

template<>
class PaddedLock<true> {
  mutable CacheLineStorage<SimpleLock> Lock;

public:
  void lock() const { Lock.get().lock(); }
  bool try_lock() const { return Lock.get().try_lock(); }
  void unlock() const { Lock.get().unlock(); }
};

template<>
class PaddedLock<false> {
public:
  void lock() const {}
  bool try_lock() const { return true; }
  void unlock() const {}
};

} // end namespace substrate
} // end namespace galois

#endif
