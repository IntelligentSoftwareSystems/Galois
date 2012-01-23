// Simple Spin Lock -*- C++ -*-
/*
Galois, a framework to exploit amorphous data-parallelism in irregular
programs.

Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS SOFTWARE
AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY, FITNESS FOR ANY
PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF PERFORMANCE, AND ANY
WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF DEALING OR USAGE OF TRADE.
NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH RESPECT TO THE USE OF THE
SOFTWARE OR DOCUMENTATION. Under no circumstances shall University be liable
for incidental, special, indirect, direct or consequential damages or loss of
profits, interruption of business, or related expenses which may arise from use
of Software or Documentation, including but not limited to those resulting from
defects in Software and/or Documentation, or loss or inaccuracy of data of any
kind.
*/

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

  inline void unlock() const {
    assert(_lock);
    asm volatile ("":::"memory");
    _lock = 0;
    asm volatile ("":::"memory");
  }

  inline bool try_lock(T val) const {
    if (_lock != 0)
      return false;
    return __sync_bool_compare_and_swap(&_lock, 0, val);
  }

  inline bool try_lock() const {
    if (_lock != 0)
      return false;
    T oldval = __sync_fetch_and_or(&_lock, 1);
    return !(oldval & 1);
  }

  inline T getValue() const {
    return _lock;
  }
};

template<typename T>
class SimpleLock<T, false> {
public:
  inline void lock(T val = 0) const {}
  inline void unlock() const {asm volatile ("":::"memory");}
  inline bool try_lock(T val = 1) const { return true; }
  inline T getValue() const { return 0; }
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

  inline void unlock() {
    assert(_lock & 1);
    asm volatile ("":::"memory");
    _lock = _lock & ~1;
    asm volatile ("":::"memory");
  }

  inline void unlock_and_clear() {
    assert(_lock & 1);
    asm volatile ("":::"memory");
    _lock = 0;
    asm volatile ("":::"memory");
  }

  inline void unlock_and_set(T val) {
    assert(_lock & 1);
    assert(!((uintptr_t)val & 1));
    asm volatile ("":::"memory");
    _lock = (uintptr_t)val;
    asm volatile ("":::"memory");
  }
  
  inline T getValue() const {
    return (T)(_lock & ~1);
  }

  inline void setValue(T val) {
    uintptr_t nval = (uintptr_t)val;
    nval |= (_lock & 1);
    _lock = nval;
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
  inline bool CAS(T oldval, T newval) {
    assert(!((uintptr_t)oldval & 1) && !((uintptr_t)newval & 1));
    return __sync_bool_compare_and_swap(&_lock, (uintptr_t)oldval, (uintptr_t)newval);
  }
};

template<typename T>
class PtrLock<T, false> {
  T _lock;
public:
  PtrLock() : _lock(0) {}
  explicit PtrLock(T val) : _lock(val) {}

  inline void lock() {}
  inline void unlock() { asm volatile ("":::"memory"); }
  inline void unlock_and_clear() { asm volatile ("":::"memory"); _lock = 0; asm volatile ("":::"memory"); }
  inline void unlock_and_set(T val) { asm volatile ("":::"memory"); _lock = val; asm volatile ("":::"memory"); }
  inline T getValue() const { return _lock; }
  inline void setValue(T val) { _lock = val; }
  inline bool try_lock() { return true; }
  inline bool CAS(T oldval, T newval) {
    if (_lock == oldval) {
      _lock = newval;
      return true;
    }
    return false;
  }
};

}

#endif
