// Atomic Types type -*- C++ -*-
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

#ifndef _GALOIS_UTIL_ATOMIC_H
#define _GALOIS_UTIL_ATOMIC_H

namespace Galois {

//! Atomic Wrapper for any integer or bool type
/*!  An atomic wrapper that provides sensible atomic behavior for most
  primative data types.  Operators return the value of type T so as to
  retain atomic RMW semantics.  */
template<typename T>
class GAtomic {
  T val;

public:
  //! Initialize with a value
  explicit GAtomic(const T& i) :val(i) {}
  //! atomic add and fetch
  T operator+=(const T& rhs) {
    return __sync_add_and_fetch(&val, rhs); 
  }
  //! atomic sub and fetch
  T operator-=(const T& rhs) {
    return __sync_sub_and_fetch(&val, rhs); 
  }
  //! atomic increment and fetch
  T operator++() {
    return __sync_add_and_fetch(&val, 1);
  }
  //! atomic fetch and increment
  T operator++(int) {
    return __sync_fetch_and_add(&val, 1);
  }
  //! atomic decrement and fetch
  T operator--() { 
    return __sync_sub_and_fetch(&val, 1); 
  }
  //! atomic fetch and decrement
  T operator--(int) {
    return __sync_fetch_and_sub(&val, 1);
  }
  //! conversion operator to base data type (atomic get)
  operator T() const {
    return val;
  }
  //! atomic assign
  T operator=(const T& i) {
    val = i;
    return i;
  }
  //! atomic assign
  T operator=(const GAtomic& i) {
    T iv = (int)i;
    val = iv;
    return iv;
  }
  //! direct compare and swap
  bool cas (const T& expected, const T& updated) {
    return __sync_bool_compare_and_swap (&val, expected, updated);
  }
};

}



#endif //  _GALOIS_UTIL_ATOMIC_H
