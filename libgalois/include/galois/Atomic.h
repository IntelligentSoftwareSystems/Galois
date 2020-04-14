/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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
 */

#ifndef GALOIS_ATOMIC_H
#define GALOIS_ATOMIC_H

#include <iterator>

#include "galois/config.h"
#include "galois/substrate/CacheLineStorage.h"

namespace galois {

namespace internal {
/**
 * Common implementation.
 */
template <typename T, template <typename _> class W, bool CONCURRENT>
class GAtomicImpl {
  // galois::runtime::LL::CacheLineStorage<T> val;
  W<T> val;

public:
  //! Initialize with a value
  explicit GAtomicImpl(const T& i) : val(i) {}
  //! default constructor
  GAtomicImpl() {}

  //! atomic add and fetch
  T operator+=(const T& rhs) { return __sync_add_and_fetch(&val.data, rhs); }
  //! atomic sub and fetch
  T operator-=(const T& rhs) { return __sync_sub_and_fetch(&(val.data), rhs); }
  //! atomic increment and fetch
  T operator++() { return __sync_add_and_fetch(&(val.data), 1); }
  //! atomic fetch and increment
  T operator++(int) { return __sync_fetch_and_add(&(val.data), 1); }
  //! atomic decrement and fetch
  T operator--() { return __sync_sub_and_fetch(&(val.data), 1); }
  //! atomic fetch and decrement
  T operator--(int) { return __sync_fetch_and_sub(&(val.data), 1); }
  //! conversion operator to base data type
  operator T() const { return val.data; }
  //! assign from underlying type
  T& operator=(const T& i) { return val.data = i; }
  //! assignment operator
  T& operator=(const GAtomicImpl& i) { return val.data = i.val.data; }
  //! direct compare and swap
  bool cas(const T& expected, const T& updated) {
    if (val.data != expected) {
      return false;
    }
#if defined(__INTEL_COMPILER)
    return __sync_bool_compare_and_swap(
        &val.data, *reinterpret_cast<const ptrdiff_t*>(&expected),
        *reinterpret_cast<const ptrdiff_t*>(&updated));
#else
    return __sync_bool_compare_and_swap(&val.data, expected, updated);
#endif
  }
};

// non-current version
template <typename T, template <typename _> class W>
class GAtomicImpl<T, W, false> {
  // galois::runtime::LL::CacheLineStorage<T> val;
  W<T> val;

public:
  //! Initialize with a value
  explicit GAtomicImpl(const T& i) : val(i) {}
  //! default constructor
  GAtomicImpl() {}

  //! atomic add and fetch
  T operator+=(const T& rhs) { return (val.data += rhs); }
  //! atomic sub and fetch
  T operator-=(const T& rhs) { return (val.data -= rhs); }
  //! atomic increment and fetch
  T operator++() { return ++(val.data); }
  //! atomic fetch and increment
  T operator++(int) { return (val.data)++; }
  //! atomic decrement and fetch
  T operator--() { return --(val.data); }
  //! atomic fetch and decrement
  T operator--(int) { return (val.data)--; }
  //! conversion operator to base data type
  operator T() const { return val.data; }
  //! assign from underlying type
  T& operator=(const T& i) { return val.data = i; }
  //! assignment operator
  T& operator=(const GAtomicImpl& i) { return val.data = i.val.data; }
  //! direct compare and swap
  bool cas(const T& expected, const T& updated) {
    if (val.data != expected) {
      return false;
    } else {
      val.data = updated;
      return true;
    }
  }
};

//! Basic atomic
template <typename T, template <typename _> class W, bool CONCURRENT>
class GAtomicBase : public GAtomicImpl<T, W, CONCURRENT> {
  typedef GAtomicImpl<T, W, CONCURRENT> Super_ty;

public:
  //! Initialize with a value
  explicit GAtomicBase(const T& i) : Super_ty(i) {}

  //! default constructor
  GAtomicBase() : Super_ty() {}

  T& operator=(const GAtomicBase& that) { return Super_ty::operator=(that); }

  T& operator=(const T& that) { return Super_ty::operator=(that); }
};

//! Specialization for pointers
template <typename T, template <typename _> class W, bool CONCURRENT>
class GAtomicBase<T*, W, CONCURRENT> : public GAtomicImpl<T*, W, CONCURRENT> {
  typedef GAtomicImpl<T*, W, CONCURRENT> Super_ty;

public:
  typedef typename std::iterator_traits<T*>::difference_type difference_type;

  GAtomicBase() : Super_ty() {}

  GAtomicBase(T* i) : Super_ty(i) {}

  T*& operator=(const GAtomicBase& that) { return Super_ty::operator=(that); }

  T*& operator=(T* that) { return Super_ty::operator=(that); }

  T* operator+=(const difference_type& rhs) {
    if (CONCURRENT) {
      return __sync_add_and_fetch(&Super_ty::val.data, rhs);
    } else {
      return (Super_ty::val.data += rhs);
    }
  }

  T* operator-=(const difference_type& rhs) {
    if (CONCURRENT) {
      return __sync_sub_and_fetch(&Super_ty::val.data, rhs);
    } else {
      return (Super_ty::val.data -= rhs);
    }
  }
};

//! Specialization for const pointers
template <typename T, template <typename _> class W, bool CONCURRENT>
class GAtomicBase<const T*, W, CONCURRENT>
    : public GAtomicImpl<const T*, W, CONCURRENT> {
  typedef GAtomicImpl<const T*, W, CONCURRENT> Super_ty;

public:
  typedef
      typename std::iterator_traits<const T*>::difference_type difference_type;

  GAtomicBase() : Super_ty() {}

  GAtomicBase(const T* i) : Super_ty(i) {}

  const T*& operator=(const GAtomicBase& that) {
    return Super_ty::operator=(that);
  }

  const T*& operator=(const T* that) { return Super_ty::operator=(that); }

  const T* operator+=(const difference_type& rhs) {
    if (CONCURRENT) {
      return __sync_add_and_fetch(&Super_ty::val.data, rhs);
    } else {
      return (Super_ty::val.data += rhs);
    }
  }

  const T* operator-=(const difference_type& rhs) {
    if (CONCURRENT) {
      return __sync_sub_and_fetch(&Super_ty::val.data, rhs);
    } else {
      return (Super_ty::val.data -= rhs);
    }
  }
};

//! Specialization for bools
template <template <typename _> class W, bool CONCURRENT>
class GAtomicBase<bool, W, CONCURRENT>
    : private GAtomicImpl<bool, W, CONCURRENT> {
  typedef GAtomicImpl<bool, W, CONCURRENT> Super_ty;

public:
  //! Initialize with a value
  explicit GAtomicBase(bool i) : Super_ty(i) {}

  GAtomicBase() : Super_ty() {}

  //! conversion operator to base data type
  operator bool() const { return Super_ty::operator bool(); }

  //! assignment operator
  bool& operator=(const GAtomicBase& i) { return Super_ty::operator=(i); }

  //! assign from underlying type
  bool& operator=(bool i) { return Super_ty::operator=(i); }
  //! direct compare and swap
  bool cas(bool expected, bool updated) {
    return Super_ty::cas(expected, updated);
  }
};

template <typename T>
struct DummyWrapper {
  T data;

  explicit DummyWrapper(const T& d) : data(d) {}
  DummyWrapper() {}
};

} // namespace internal

/**
 * An atomic wrapper that provides sensible atomic behavior for most
 * primative data types.  Operators return the value of type T so as to
 * retain atomic RMW semantics.
 */
template <typename T, bool CONCURRENT = true>
class GAtomic
    : public internal::GAtomicBase<T, internal::DummyWrapper, CONCURRENT> {
  typedef internal::GAtomicBase<T, internal::DummyWrapper, CONCURRENT> Super_ty;

public:
  GAtomic() : Super_ty() {}
  explicit GAtomic(const T& v) : Super_ty(v) {}

  T& operator=(const GAtomic& that) { return Super_ty::operator=(that); }

  T& operator=(const T& that) { return Super_ty::operator=(that); }
};

/**
 * Cache-line padded version of {@link GAtomic}.
 */
template <typename T, bool CONCURRENT = true>
class GAtomicPadded
    : public internal::GAtomicBase<T, galois::substrate::CacheLineStorage,
                                   CONCURRENT> {

  typedef internal::GAtomicBase<T, galois::substrate::CacheLineStorage,
                                CONCURRENT>
      Super_ty;

public:
  GAtomicPadded() : Super_ty() {}
  explicit GAtomicPadded(const T& v) : Super_ty(v) {}

  T& operator=(const GAtomicPadded& that) { return Super_ty::operator=(that); }

  T& operator=(const T& that) { return Super_ty::operator=(that); }
};

} // namespace galois

#endif
