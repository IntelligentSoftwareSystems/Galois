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

#ifndef GALOIS_LAZYOBJECT_H
#define GALOIS_LAZYOBJECT_H

#include <type_traits>
#include <utility>

#include "galois/config.h"
#include "galois/gIO.h"

namespace galois {

/**
 * Single object with specialization for void type. To take advantage of empty
 * member optimization, users should subclass this class, otherwise the
 * compiler will insert non-zero padding for fields (even when empty).
 */
template <typename T>
class StrictObject {
  T data;

public:
  typedef T value_type;
  typedef T& reference;
  typedef const T& const_reference;
  const static bool has_value = true;

  StrictObject() {}
  StrictObject(const_reference t) : data(t) {}
  const_reference get() const { return data; }
  reference get() { return data; }
};

template <>
struct StrictObject<void> {
  typedef void* value_type;
  typedef void* reference;
  typedef void* const_reference;
  const static bool has_value = false;

  StrictObject() {}
  StrictObject(const_reference) {}
  reference get() const { return 0; }
};

/**
 * Single (uninitialized) object with specialization for void type. To take
 * advantage of empty member optimization, users should subclass this class,
 * otherwise the compiler will insert non-zero padding for fields (even when
 * empty).
 */
template <typename T>
class LazyObject {
  typedef
      typename std::aligned_storage<sizeof(T),
                                    std::alignment_of<T>::value>::type CharData;

  union Data {
    CharData buf;
    T value_;

    // Declare constructor explicitly because Data must be default
    // constructable regardless of the constructability of T.
    Data() {} // NOLINT(modernize-use-equals-default)
    ~Data() {} // NOLINT(modernize-use-equals-default)

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

  template <typename... Args>
  void construct(Args&&... args) {
    new (cast()) T(std::forward<Args>(args)...);
  }

  const_reference get() const { return *cast(); }
  reference get() { return *cast(); }
};

template <>
struct LazyObject<void> {
  typedef void* value_type;
  typedef void* reference;
  typedef void* const_reference;
  const static bool has_value = false;
  struct size_of {
    const static size_t value = 0;
  };

  void destroy() {}
  void construct(const_reference x) {}

  template <typename... Args>
  void construct(Args&&... args) {}

  const_reference get() const { return 0; }
};

} // namespace galois
#endif
