// simple spin lock -*- C++ -*-

#ifndef _SIMPLE_LOCK_H
#define _SIMPLE_LOCK_H

#include "galois_config.h"

#ifdef WITH_CRAY_POOL
#include "Support/ThreadSafe/cray_simple_lock.h"
#else

#include <cassert>

namespace threadsafe {

template<typename T, bool isALock>
class simpleLock;

template<typename T>
class simpleLock<T, true> {
  T _lock;
public:
  simpleLock() : _lock(0) {}

  void lock(T val = 1) { 
    while (!try_lock(val)) {} 
  }

  void unlock() {
    assert(_lock);
    _lock = 0;
  }

  bool try_lock(T val) {
    return __sync_bool_compare_and_swap(&_lock, 0, val);
  }

  T getValue() {
    return _lock;
  }
};

template<typename T>
class simpleLock<T, false> {
public:
  simpleLock() {}
  void lock(T val = 0) {}
  void unlock() {}
  bool try_lock(T val = 0) { return true; }
  T getValue() { return 0; }
};


}

#endif
#endif
