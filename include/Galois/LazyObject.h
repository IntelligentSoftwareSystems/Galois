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
#include "Galois/Runtime/ll/gio.h"
#include "Galois/TypeTraits.h"

// For consistent name, use boost rather than C++11 std::is_trivially_constuctible
#include <boost/type_traits/has_trivial_constructor.hpp>

#include GALOIS_CXX11_STD_HEADER(type_traits)
#include GALOIS_CXX11_STD_HEADER(utility)

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

#if defined(__IBMCPP__) && __IBMCPP__ <= 1210
namespace LazyObjectDetail {

template<typename T, typename CharData, bool>
struct SafeDataBase {
  union type {
    CharData buf;
    T value_;
    T& value() { return value_; }
    const T& value() const { return value_; }
  };
};

template<typename T, typename CharData>
struct SafeDataBase<T, CharData, false> {
  union type {
    CharData buf;
    T& value() { return *reinterpret_cast<T*>(&buf); }
    const T& value() const { return *reinterpret_cast<const T*>(&buf); }

    type() {
      // XXX: Keep this as a runtime exception rather than a compile-time one
      GALOIS_DIE("Unsafe construct for type '", __PRETTY_FUNCTION__, "' when expecting strict aliasing");
    }
  };
};

/**
 * Works around compilers that do not support non-trivially constructible
 * members in unions.
 */
template<typename T, typename CharData>
struct SafeData: public SafeDataBase<T, CharData,
  boost::has_trivial_constructor<T>::value || Galois::has_known_trivial_constructor<T>::value > { };

} // end detail
#endif

/**
 * Single (uninitialized) object with specialization for void type. To take
 * advantage of empty member optimization, users should subclass this class,
 * otherwise the compiler will insert non-zero padding for fields (even when
 * empty).
 */
template<typename T>
class LazyObject {
  typedef typename std::aligned_storage<sizeof(T), std::alignment_of<T>::value>::type CharData;

#if defined(__IBMCPP__) && __IBMCPP__ <= 1210 
  typedef typename LazyObjectDetail::SafeData<T, CharData>::type Data;
#else
  union Data {
    CharData buf;
    T value_;

    Data() { }
    ~Data() { }

    T& value() { return value_; }
    const T& value() const { return value_; }
  };
#endif

  Data data_;

  T* cast() { return &data_.value(); }
  const T* cast() const { return &data_.value(); }

public:
  typedef T value_type;
  typedef T& reference;
  typedef const T& const_reference;
  const static bool has_value = true;
  const static unsigned sizeof_value = sizeof(T);

  void destroy() { cast()->~T(); }
  void construct(const_reference x) { new (cast()) T(x); }

  template<typename... Args>
  void construct(Args&&... args) { new (cast()) T(std::forward<Args>(args)...); }
  
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

  template<typename... Args>
  void construct(Args&&... args) { }

  const_reference get() const { return 0; }
};

}
#endif

