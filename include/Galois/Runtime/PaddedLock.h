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
  cache_line_storage<SimpleLock<int, true> > Lock;

public:
  void lock() { Lock.data.lock(); }
  bool try_lock() { return Lock.data.try_lock(); }
  void unlock() { Lock.data.unlock(); }
};

template<>
class PaddedLock<false> {
public:
  void lock() {}
  bool try_lock() { return true; }
  void unlock() {}
};

}

#endif
