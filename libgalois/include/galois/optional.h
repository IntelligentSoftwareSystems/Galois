/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
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

#ifndef GALOIS_OPTIONAL_H
#define GALOIS_OPTIONAL_H

#include "galois/LazyObject.h"
#include <cassert>

namespace galois {

/**
 * Galois version of <code>boost::optional</code>.
 */
template <typename T>
class optional {
  LazyObject<T> data_;
  bool initialized_;

  void construct(const T& val) {
    data_.construct(val);
    initialized_ = true;
  }

  void assign_impl(const T& val) { get_impl() = val; }

  void destroy() {
    if (initialized_) {
      data_.destroy();
      initialized_ = false;
    }
  }

  T& get_impl() { return data_.get(); }
  const T& get_impl() const { return data_.get(); }

public:
  typedef bool (optional::*unspecified_bool_type)() const;

  optional() : initialized_(false) {}

  optional(const T& val) : initialized_(false) { construct(val); }

  optional(const optional& rhs) : initialized_(false) {
    if (rhs.is_initialized())
      construct(rhs.get_impl());
  }

  template <typename U>
  explicit optional(const optional<U>& rhs) : initialized_(false) {
    assign(rhs);
  }

  ~optional() { destroy(); }

  void assign(const optional& rhs) {
    if (is_initialized()) {
      if (rhs.is_initialized())
        assign_impl(rhs.get_impl());
      else
        destroy();
    } else {
      if (rhs.is_initialized())
        construct(rhs.get_impl());
    }
  }

  template <typename U>
  void assign(const optional<U>& rhs) {
    if (is_initialized()) {
      if (rhs.is_initialized())
        assign_impl(rhs.get_impl());
      else
        destroy();
    } else {
      if (rhs.is_initialized())
        construct(rhs.get_impl());
    }
  }

  void assign(const T& val) {
    if (is_initialized())
      assign_impl(val);
    else
      construct(val);
  }

  bool is_initialized() const { return initialized_; }

  optional& operator=(const optional& rhs) {
    assign(rhs);
    return *this;
  }

  template <typename U>
  optional& operator=(const optional<U>& rhs) {
    assign(rhs);
    return *this;
  }

  optional& operator=(const T& val) {
    assign(val);
    return *this;
  }

  T& get() {
    assert(initialized_);
    return get_impl();
  }
  const T& get() const {
    assert(initialized_);
    return get_impl();
  }
  T& operator*() { return get(); }
  const T& operator*() const { return get(); }
  T* operator->() {
    assert(initialized_);
    return &get_impl();
  }
  const T* operator->() const {
    assert(initialized_);
    return &get_impl();
  }

  operator unspecified_bool_type() const {
    return initialized_ ? &optional::is_initialized : 0;
  }
};

} // namespace galois

#endif
