// simple spin lock -*- C++ -*-

#ifndef _SIMPLE_LOCK_H
#define _SIMPLE_LOCK_H

#include <stdint.h>
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

  inline void lock(T val) const {
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

  inline void lock() const {
    do {
      while (_lock != 0) {
#if defined(__i386__) || defined(__amd64__)
	asm volatile ( "pause");
#endif
      }
      if (try_lock())
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

  inline bool try_lock(T val) const {
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

  inline bool try_lock() const {
#ifdef GALOIS_CRAY
    return try_lock(1);
#else
    if (_lock != 0)
      return false;
    T oldval = __sync_fetch_and_or(&_lock, 1);
    return !(oldval & 1);
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
  bool try_lock(T val = 1) const { return true; }
  T getValue() const { return 0; }
};


//This wraps a pointer and uses the low order bit for the lock flag
template<typename T, bool isALock>
class PtrLock;

template<typename T>
class PtrLock<T, true> {
  volatile uintptr_t _lock;
public:
  PtrLock() : _lock(0) {}
  explicit PtrLock(T val) : _lock(val) {}

  inline void lock() {
    do {
      while ((_lock & 1) != 0) {
#if defined(__i386__) || defined(__amd64__)
	asm volatile ( "pause");
#endif
      }
      if (try_lock())
	break;
    } while (true);
  }

  void unlock() {
    assert(_lock & 1);
    _lock = _lock & ~1;
  }

  void unlock_and_clear() {
    assert(_lock & 1);
    _lock = 0;
  }

  void unlock_and_set(T val) {
    assert(_lock & 1);
    assert(!((uintptr_t)val & 1));
    _lock = (uintptr_t)val;
  }
  
  T getValue() const {
    return (T)(_lock & ~1);
  }

  void setValue(T val) {
    _lock = ((uintptr_t)val) | (_lock & 1);
  }

  inline bool try_lock() {
    uintptr_t oldval = _lock;
    if ((oldval & 1) != 0)
      return false;
    oldval = __sync_fetch_and_or(&_lock, 1);
    return !(oldval & 1);
  }

  //CAS only works on unlocked values
  //the lock bit will prevent a successful cas
  bool CAS(T oldval, T newval) {
    assert(!((uintptr_t)oldval & 1) && !((uintptr_t)newval & 1));
    return __sync_bool_compare_and_swap(&_lock, oldval, newval);
  }
};

template<typename T>
class PtrLock<T, false> {
  T _lock;
public:
  PtrLock() : _lock(0) {}
  explicit PtrLock(T val) : _lock(val) {}

  void lock() {}
  void unlock() {}
  void unlock_and_clear() { _lock = 0; }
  void unlock_and_set(T val) { _lock = val; }
  T getValue() const { return _lock; }
  void setValue(T val) { _lock = val; }
  bool try_lock() { return true; }
  bool CAS(T oldval, T newval) {
    if (_lock == oldval) {
      _lock = newval;
      return true;
    }
    return false;
  }
};

}

#endif
