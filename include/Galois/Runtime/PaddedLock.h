// cache-line padded simple spin lock -*- C++ -*-

#ifndef _PADDED_LOCK_H
#define _PADDED_LOCK_H

#include "SimpleLock.h"
#include "CacheLineStorage.h"

namespace GaloisRuntime {

template<bool concurrent>
class PaddedLock;

template<>
class PaddedLock<true> {
  mutable cache_line_storage<SimpleLock<int, true> > Lock;

public:
  void lock() const { Lock.data.lock(); }
  bool try_lock() const { return Lock.data.try_lock(); }
  void unlock() const { Lock.data.unlock(); }
};

template<>
class PaddedLock<false> {
public:
  void lock() const {}
  bool try_lock() const { return true; }
  void unlock() const {}
};

}

#endif
