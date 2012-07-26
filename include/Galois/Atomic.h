/** Atomic Types type -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#ifndef GALOIS_ATOMIC_H
#define GALOIS_ATOMIC_H

#include <iterator>

#include "Galois/Runtime/ll/CacheLineStorage.h"

namespace Galois {


namespace internal {
//! Atomic Wrapper for any integer or bool type
/*!  An atomic wrapper that provides sensible atomic behavior for most
  primative data types.  Operators return the value of type T so as to
  retain atomic RMW semantics.  */

template<typename T, template <typename _> class W>
class GAtomicImpl {
  // GaloisRuntime::LL::CacheLineStorage<T> val;
  W<T> val;

public:
  //! Initialize with a value
  explicit GAtomicImpl(const T& i): val(i) {}
  //! default constructor
  GAtomicImpl() {}

  //! atomic add and fetch
  T operator+=(const T& rhs) {
    return __sync_add_and_fetch(&(val.data), rhs); 
  }
  //! atomic sub and fetch
  T operator-=(const T& rhs) {
    return __sync_sub_and_fetch(&(val.data), rhs); 
  }
  //! atomic increment and fetch
  T operator++() {
    return __sync_add_and_fetch(&(val.data), 1);
  }
  //! atomic fetch and increment
  T operator++(int) {
    return __sync_fetch_and_add(&(val.data), 1);
  }
  //! atomic decrement and fetch
  T operator--() { 
    return __sync_sub_and_fetch(&(val.data), 1); 
  }
  //! atomic fetch and decrement
  T operator--(int) {
    return __sync_fetch_and_sub(&(val.data), 1);
  }
  //! conversion operator to base data type 
  operator T() const {
    return (val.data);
  }
  //! assign from underlying type
  T operator=(const T& i) {
    (val.data) = i;
    return i;
  }
  //! assignment operator
  T operator=(const GAtomicImpl& i) {
    T iv = (T)i;
    val.data = iv;
    return iv;
  }
  //! direct compare and swap
  bool cas (const T& expected, const T& updated) {
#if defined(__INTEL_COMPILER)
    return __sync_bool_compare_and_swap (
        &(val.data), 
        *(reinterpret_cast<const ptrdiff_t*> (&expected)), 
        *(reinterpret_cast<const ptrdiff_t*> (&updated)));
#else 
    return __sync_bool_compare_and_swap (&(val.data), expected, updated);
#endif
  }
};



// basic type
template <typename T, template <typename _> class W>
class GAtomicBase: public GAtomicImpl<T, W> {
  typedef GAtomicImpl<T, W> Super_ty;

public:
  //! Initialize with a value
  explicit GAtomicBase(const T& i) : Super_ty (i) {}

  //! default constructor
  GAtomicBase(): Super_ty () {}

  T operator = (const GAtomicBase& that) {
    return Super_ty::operator = (that);
  }

  T operator = (const T& that) {
    return Super_ty::operator = (that);
  }
};

// specialized for pointers
template <typename T, template <typename _> class W>
class GAtomicBase<T*, W> : public GAtomicImpl<T*, W>  {
  typedef GAtomicImpl<T*, W> Super_ty;

public:
  typedef typename std::iterator_traits<T*>::difference_type Diff_ty;

  GAtomicBase (): Super_ty () {}

  GAtomicBase (T* i): Super_ty (i) {}

  T* operator = (const GAtomicBase& that) {
    return Super_ty::operator = (that);
  }

  T* operator = (T* that) {
    return Super_ty::operator = (that);
  }

  T* operator += (const Diff_ty rhs) {
    return __sync_add_and_fetch (&(Super_ty::val.data), rhs); 
  }

  T* operator -= (const Diff_ty rhs) {
    return __sync_sub_and_fetch (&(Super_ty::val.data), rhs);
  }

};

template <typename T, template <typename _> class W>
class GAtomicBase<const T*, W> : public GAtomicImpl<const T*, W>  {
  typedef GAtomicImpl<const T*, W> Super_ty;

public:
  typedef typename std::iterator_traits<const T*>::difference_type Diff_ty;

  GAtomicBase (): Super_ty () {}

  GAtomicBase (const T* i): Super_ty (i) {}

  const T* operator = (const GAtomicBase& that) {
    return Super_ty::operator = (that);
  }

  const T* operator = (const T* that) { 
    return Super_ty::operator = (that);
  }

  const T* operator += (const Diff_ty rhs) {
    return __sync_add_and_fetch (&(Super_ty::val.data), rhs); 
  }

  const T* operator -= (const Diff_ty rhs) {
    return __sync_sub_and_fetch (&(Super_ty::val.data), rhs);
  }

};

// specialized for bools
template<template <typename _> class W>
class GAtomicBase<bool, W> : private GAtomicImpl<bool, W> {
  typedef GAtomicImpl<bool, W> Super_ty;

public:
  //! Initialize with a value
  explicit GAtomicBase(bool i): Super_ty(i) {}

  GAtomicBase (): Super_ty () {}

  //! conversion operator to base data type 
  operator bool() const {
    return Super_ty::operator bool ();
  }

  //! assignment operator
  bool operator=(const GAtomicBase<bool, W>& i) {
    return Super_ty::operator = (i);
  }

  //! assign from underlying type
  bool operator=(bool i) {
    return Super_ty::operator = (i);
  }
  //! direct compare and swap
  bool cas (bool expected, bool updated) {
    return Super_ty::cas (expected, updated);
  }
};


template <typename T>
struct DummyWrapper {
  T data;

  explicit DummyWrapper (const T& d): data (d) {}

  DummyWrapper () {}

};

} // end namespace internal

template <typename T>
class GAtomic: public internal::GAtomicBase <T, internal::DummyWrapper> {

  typedef internal::GAtomicBase<T, internal::DummyWrapper> Super_ty;

public:
  GAtomic (): Super_ty () {}
  explicit GAtomic (const T& v): Super_ty (v) {}

  T operator = (const GAtomic& that) {
    return Super_ty::operator = (that);
  }

  T operator = (const T& that) {
    return Super_ty::operator = (that);
  }
};

template <typename T>
class GAtomicPadded: 
  public internal::GAtomicBase<T, GaloisRuntime::LL::CacheLineStorage> {

  typedef  internal::GAtomicBase<T, GaloisRuntime::LL::CacheLineStorage> Super_ty;

public:
  GAtomicPadded (): Super_ty () {}
  explicit GAtomicPadded (const T& v): Super_ty (v) {}

  T operator = (const GAtomicPadded& that) {
    return Super_ty::operator = (that);
  }

  T operator = (const T& that) {
    return Super_ty::operator = (that);
  }
};



} // end namespace Galois



#endif //  GALOIS_ATOMIC_H

