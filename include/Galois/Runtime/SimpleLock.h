// simple spin lock -*- C++ -*-

#ifndef _SIMPLE_LOCK_H
#define _SIMPLE_LOCK_H

#include <cassert>

namespace GaloisRuntime {

template<typename T, bool isALock>
class SimpleLock;

template<typename T>
class SimpleLock<T, true> {
  T _lock;
public:
  SimpleLock() : _lock(0) {
#ifdef GALOIS_CRAY
    writexf(&_lock, 0); // sets to full
#endif
  }

  void lock(T val = 1) { 
    while (!try_lock(val)) {} 
  }

  void unlock() {
    assert(_lock);
#ifdef GALOIS_CRAY
    readfe(&_lock); // sets to empty, acquiring the lock lock
    writeef(&_lock, 0); // clears the lock and clears the lock lock
#else
    _lock = 0;
#endif
  }

  bool try_lock(T val) {
#ifdef GALOIS_CRAY
    T V = readfe(&_lock); // sets to empty, acquiring the lock lock
    if (V) {
      //failed
      writeef(&_lock, V); //write back the same value and clear the lock lock
      return false;
    } else {
      //can grab
      writeef(&_lock, val); //write our value into the lock (acquire) and clear the lock lock
      return true;
    }
#else
    return __sync_bool_compare_and_swap(&_lock, (T)0, val);
#endif
  }

  T getValue() {
    return _lock;
  }
};

template<typename T>
class SimpleLock<T, false> {
public:
  void lock(T val = 0) {}
  void unlock() {}
  bool try_lock(T val = 0) { return true; }
  T getValue() { return 0; }
};


}

#endif
