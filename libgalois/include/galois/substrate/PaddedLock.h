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
