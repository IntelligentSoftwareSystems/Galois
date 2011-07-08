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

//TODO: it may be a good idea to add buffering to these classes so that the object is store one per cache line

//! wrap a data item with atomic updates
//! operators return void so as to force only one update per statement
template<typename T>
class GAtomic {
  T val;

public:
  explicit GAtomic(const T& i) :val(i) {}
  GAtomic& operator+=(const T& rhs) {
    __sync_add_and_fetch(&val, rhs); 
    return *this; 
  }
  GAtomic& operator-=(const T& rhs) {
    __sync_sub_and_fetch(&val, rhs); 
    return *this;
  }
  GAtomic& operator++() {
    __sync_add_and_fetch(&val, 1);
    return *this;
  }
  GAtomic& operator++(int) {
    __sync_fetch_and_add(&val, 1);
    return *this;
  }
  GAtomic& operator--() { 
    __sync_sub_and_fetch(&val, 1); 
    return *this;
  }
  GAtomic& operator--(int) {
    __sync_fetch_and_sub(&val, 1);
    return *this;
  }
  
  operator T() {
    return val;
  }
  
  GAtomic& operator=(const T& i) {
    val = i;
    return *this;
  }
  GAtomic& operator=(const GAtomic& i) {
    val = i.val;
    return *this;
  }
  bool cas (const T& expected, const T& updated) {
    return __sync_bool_compare_and_swap (&val, expected, updated);
  }
};

}



#endif //  _GALOIS_UTIL_ATOMIC_H
