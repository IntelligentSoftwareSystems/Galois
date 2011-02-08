// simple spin lock -*- C++ -*-

#ifndef _SIMPLE_LOCK_H
#define _SIMPLE_LOCK_H

#include <cassert>

namespace GaloisRuntime {

template<typename T, bool isALock>
class SimpleLock;

template<typename T>
class SimpleLock<T, true> {
  volatile mutable T _lock; //Allow locking a const
public:
  SimpleLock() : _lock(0) {
#ifdef GALOIS_CRAY
    writexf(&_lock, 0); // sets to full
#endif
  }

  void lock(T val = 1) const {
    do {
      while (_lock != 0) {
#if defined(__i386__) || defined(__amd64__)
	asm volatile ( "pause");
#endif
      }
      if (try_lock(val))
	break;
    } while (true);
  }

  void unlock() const {
    assert(_lock);
#ifdef GALOIS_CRAY
    readfe(&_lock); // sets to empty, acquiring the lock lock
    writeef(&_lock, 0); // clears the lock and clears the lock lock
#else
    _lock = 0;
#endif
  }

  bool try_lock(T val) const {
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
    if (_lock != 0)
      return false;
    return __sync_bool_compare_and_swap(&_lock, 0, val);
#endif
  }

  T getValue() const {
    return _lock;
  }
};

template<typename T>
class SimpleLock<T, false> {
public:
  void lock(T val = 0) const {}
  void unlock() const {}
  bool try_lock(T val = 0) const { return true; }
  T getValue() const { return 0; }
};


}

#endif
