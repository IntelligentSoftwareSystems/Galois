/** Lazy and strict object types -*- C++ -*-
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
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#ifndef GALOIS_LAZYOBJECT_H
#define GALOIS_LAZYOBJECT_H

#include "Galois/config.h"
#include GALOIS_C11_STD_HEADER(type_traits)

namespace Galois {

/**
 * Single object with specialization for void type. To take advantage of empty
 * member optimization, users should subclass this class, otherwise the
 * compiler will insert non-zero padding for fields (even when empty).
 */
template<typename T>
class StrictObject {
  T data;
public:
  typedef T value_type;
  typedef T& reference;
  typedef const T& const_reference;
  const static bool has_value = true;

  StrictObject() { }
  StrictObject(const_reference t): data(t) { }
  const_reference get() const { return data; }
  reference get() { return data; }
};

template<>
struct StrictObject<void> {
  typedef void* value_type;
  typedef void* reference;
  typedef void* const_reference;
  const static bool has_value = false;

  StrictObject() { }
  StrictObject(const_reference) { }
  reference get() const { return 0; }
};

/**
 * Single (uninitialized) object with specialization for void type. To take
 * advantage of empty member optimization, users should subclass this class,
 * otherwise the compiler will insert non-zero padding for fields (even when
 * empty).
 */
template<typename T>
class LazyObject {
  typedef typename std::aligned_storage<sizeof(T), std::alignment_of<T>::value>::type Data;

  //__attribute__((may_alias))
  Data data_[1];

  //XXX(ddn): These casts are not correct according to ANSI strict aliasing rules
  T* cast() { 
    union { Data* as_pdata; T* as_pvalue; } caster = { data_ };
    return caster.as_pvalue;
  }

  const T* cast() const {
    union { const Data* as_pdata; const T* as_pvalue; } caster = { data_ };
    return caster.as_pvalue;
  }

public:
  typedef T value_type;
  typedef T& reference;
  typedef const T& const_reference;
  const static bool has_value = true;
  const static unsigned sizeof_value = sizeof(T);

  void destroy() { cast()->~T(); }
  void construct(const_reference x) { new (cast()) T(x); }
  const_reference get() const { return *cast(); }
  reference get() { return *cast(); }
};

template<>
struct LazyObject<void> {
  typedef void* value_type;
  typedef void* reference;
  typedef void* const_reference;
  const static bool has_value = false;
  const static unsigned sizeof_value = 0;

  void destroy() { }
  void construct(const_reference x) { }
  const_reference get() const { return 0; }
};

}
#endif

