/** Lazy and strict object types -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galois is a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * Galois is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Galois.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @section Copyright
 *
 * Copyright (C) 2015, The University of Texas at Austin. All rights
 * reserved.
 *
 * @section Description
 *
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#ifndef GALOIS_LAZYOBJECT_H
#define GALOIS_LAZYOBJECT_H

#include "galois/gIO.h"
#include "galois/TypeTraits.h"

// For consistent name, use boost rather than C++11 std::is_trivially_constuctible
#include <boost/type_traits/has_trivial_constructor.hpp>

#include <type_traits>
#include <utility>

namespace galois {

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
  typedef typename std::aligned_storage<sizeof(T), std::alignment_of<T>::value>::type CharData;

  union Data {
    CharData buf;
    T value_;

    Data() { }
    ~Data() { }

    T& value() { return value_; }
    const T& value() const { return value_; }
  };

  Data data_;

  T* cast() { return &data_.value(); }
  const T* cast() const { return &data_.value(); }

public:
  typedef T value_type;
  typedef T& reference;
  typedef const T& const_reference;
  const static bool has_value = true;
  // Can't support incomplete T's but provide same interface as
  // {@link galois::LargeArray} for consistency
  struct size_of {
    const static size_t value = sizeof(T);
  };

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
  struct size_of {
    const static size_t value = 0;
  };

  void destroy() { }
  void construct(const_reference x) { }

  template<typename... Args>
  void construct(Args&&... args) { }

  const_reference get() const { return 0; }
};

}
#endif
