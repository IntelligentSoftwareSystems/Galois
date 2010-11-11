// simple spin lock -*- C++ -*-

#ifndef _SIMPLE_LOCK_H
#define _SIMPLE_LOCK_H

#include "galois_config.h"

#ifdef WITH_CRAY_POOL
#include "Support/ThreadSafe/cray_simple_lock.h"
#else

#include <cassert>

namespace threadsafe {

  // The most stupid spinlock you can imagine
  class simpleLock {
    int _lock;
  public:

    simpleLock() : _lock(0) {}

    void read_lock() {
      //      assert(!_lock);
      while (!try_read_lock()) {}
    }
    void read_unlock() {
      assert(_lock);
      _lock = 0;
    }
    bool try_read_lock() {
      return __sync_bool_compare_and_swap(&_lock, 0, 1);
    }

    void promote() {}

    void write_lock() {
      read_lock();
    }
    void write_unlock() {
      read_unlock();
    }
    bool try_write_lock() {
      return try_read_lock();
    }
  };

  class ptrLock {
    volatile void* _lock;
  public:

    ptrLock() : _lock(0) {}

    void lock(void* val) {
      //      assert(!_lock);
      while (!try_lock(val)) {}
    }
    void unlock() {
      assert(_lock);
      _lock = 0;
    }
    bool try_lock(void* val) {
      return __sync_bool_compare_and_swap(&_lock, 0, val);
    }
    volatile void* getValue() {
      return _lock;
    }
  };

}

#endif
#endif
