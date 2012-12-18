/** Specializations for void type -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2012, The University of Texas at Austin. All rights reserved.
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
 * @section Description
 *
 * Various object specializations for the void type.
 *
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#ifndef GALOIS_UTIL_VOIDSPECIALIZATIONS_H
#define GALOIS_UTIL_VOIDSPECIALIZATIONS_H

//! Array of objects with proper specialization for void type.
template<typename T>
class VoidArray {
  T* data;
  size_t size;
public:
  typedef T& reference_type;
  typedef T value_type;
  const static bool has_value = true;

  VoidArray(): data(0), size(0) { }
  ~VoidArray() { destroy(); }
  reference_type get(ptrdiff_t x) const { return data[x]; }
  void set(ptrdiff_t x, const value_type& v) { data[x] = v; }
  void allocate(size_t n) {
    assert(!data);
    size = n;
    data = reinterpret_cast<T*>(GaloisRuntime::MM::largeInterleavedAlloc(sizeof(T) * size));
  }
  void destroy() {
    if (!data) return;
    for (T* ii = data, *ei = data + size; ii != ei; ++ii)
      data->~T();
    GaloisRuntime::MM::largeInterleavedFree(data, sizeof(T) * size);
    data = 0;
    size = 0;
  }
  template<typename It>
  void copyIn(It begin, It end) {
    std::copy(begin, end, &data[0]);
  }
};

template<>
struct VoidArray<void> {
  typedef void* reference_type;
  typedef void* value_type;
  const static bool has_value = false;
  reference_type get(ptrdiff_t x) const { return 0; }
  void set(ptrdiff_t x, const value_type& v) { }
  void allocate(size_t n) { }
  void destroy() { }
  template<typename It> void copyIn(It begin, It end) { }
};

//! Single object with specialization for void type.
template<typename T>
class VoidObject {
  T data;
public:
  typedef T value_type;
  typedef T& reference_type;
  typedef const T& const_reference_type;
  const static bool has_value = true;

  VoidObject() { }
  VoidObject(const_reference_type t): data(t) { }
  const_reference_type get() const { return data; }
  reference_type get() { return data; }
};

template<>
struct VoidObject<void> {
  typedef void* value_type;
  typedef void* reference_type;
  typedef void* const_reference_type;
  const static bool has_value = false;

  VoidObject() { }
  VoidObject(const_reference_type) { }
  reference_type get() const { return 0; }
};

//! Single (uninitialized) object with specialization for void type.
template<typename T>
class LazyVoidObject {
  char data[sizeof(T)];

  T* cast() { return reinterpret_cast<T*>(&data[0]); }
public:
  typedef T value_type;
  typedef T& reference_type;
  typedef const T& const_reference_type;
  const static bool has_value = true;

  LazyVoidObject() { }
  ~LazyVoidObject() {
    // XXX(ddn): For consistency with other "lazy" objects, user is
    // responsible for calling destructor
    //destroy();
  }
  void destroy() { cast()->~T(); }
  void construct(const_reference_type x) { new (cast()) T(x); }
  const_reference_type get() const { return *cast(); }
  reference_type get() { return *cast(); }
};

template<>
struct LazyVoidObject<void> {
  typedef void* value_type;
  typedef void* reference_type;
  typedef void* const_reference_type;
  const static bool has_value = false;

  LazyVoidObject() { }
  void construct(const_reference_type x) { }
  const_reference_type get() const { return 0; }
};

#endif

