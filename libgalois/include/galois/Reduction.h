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

#ifndef GALOIS_REDUCTION_H
#define GALOIS_REDUCTION_H

#include "galois/substrate/PerThreadStorage.h"

#include <functional>
#include <limits>

namespace galois {

/**
 * A Reducible stores per-thread values of a variable of type T and merges
 * multiple values into one.
 *
 * The reduced value is obtained by merging per thread values using the binary
 * functor MergeFunc. MergeFunc takes two values of type T and produces the
 * resulting merged value:
 *
 *   T operator()(T lhs, T rhs)
 *
 * If T is expensive to copy, a moving merge function is more appropriate:
 *
 *   T& operator()(T& lhs, T&& rhs)
 *
 * IdFunc returns the identity element, which is used to initialize and reset
 * the per thread values.
 *
 * Both MergeFunc and IdFunc should be copy constructable.
 *
 * The MergeFunc and IdFunc should be related as follows:
 * 
 *   MergeFunc(x, IdFunc()) == x    for all x in X
 *
 * An example of using a move merge function:
 *
 *   auto merge_func = [](T& lhs, T&& rhs) -> T& { ... }
 *   auto identity_func = []() -> T { ... }
 *
 *   auto r = make_reducible(merge_func, identity_func);
 *   T u = ...
 *   r.update(std::move(u));
 *   T& result = r.reduce();
 */
template <typename T, typename MergeFunc, typename IdFunc>
class Reducible: public MergeFunc, public IdFunc {

  galois::substrate::PerThreadStorage<T> data_;

  void merge(T& lhs, T&& rhs) {
    T v{std::move(MergeFunc::operator()(lhs, std::move(rhs)))};
    lhs = std::move(v);
  }

  void merge(T& lhs, const T& rhs) {
    lhs = MergeFunc::operator()(lhs, rhs);
  }

public:
  using value_type = T;

  Reducible(MergeFunc merge_func, IdFunc id_func):
    MergeFunc(merge_func), IdFunc(id_func)
  {
    for (unsigned i = 0; i < data_.size(); ++i) {
      *(data_.getRemote(i)) = IdFunc::operator()();
    }
  }

  /**
   * Updates the thread local value by applying the reduction operator to
   * current and newly provided value
   */
  void update(T&& rhs) {
    merge(*data_.getLocal(), std::move(rhs));
  }

  void update(const T& rhs) {
    merge(*data_.getLocal(), rhs);
  }

  /**
   * Returns a reference to the local value of T.
   */
  T& getLocal() {
    return *data_.getLocal();
  }

  /**
   * Returns the final reduction value. Only valid outside the parallel region.
   */
  T& reduce() {
    T& lhs = *data_.getLocal();
    for (unsigned int i = 1; i < data_.size(); ++i) {
      T& rhs = *data_.getRemote(i);
      merge(lhs, std::move(rhs));
      rhs = IdFunc::operator()();
    }

    return lhs;
  }

  void reset() {
    for (unsigned int i = 0; i < data_.size(); ++i) {
      *data_.getRemote(i) = IdFunc::operator()();
    }
  }
};

/**
 * make_reducible creates a Reducible from a merge function and identity
 * function.
 */
template <typename MergeFn, typename IdFn>
auto make_reducible(const MergeFn& mergeFn, const IdFn& idFn) {
  return Reducible<std::invoke_result_t<IdFn>, MergeFn, IdFn>(mergeFn, idFn);
}

//! gmax is the functional form of std::max
template <typename T>
struct gmax {
  constexpr T operator()(const T& lhs, const T& rhs) const {
    return std::max<T>(lhs, rhs);
  }
};

//! gmax is the functional form of std::max
template <typename T>
struct gmin {
  constexpr T operator()(const T& lhs, const T& rhs) const {
    return std::min<T>(lhs, rhs);
  }
};

template <typename T, T value>
struct identity_value {
  constexpr T operator()() const {
    return T{value};
  }
};

// The following identity_value specializations exist because floating point
// numbers cannot be template arguments.

template <typename T>
struct identity_value_zero {
  constexpr T operator()() const {
    return T{0};
  }
};

template <typename T>
struct identity_value_min {
  constexpr T operator()() const {
    return std::numeric_limits<T>::min();
  }
};

template <typename T>
struct identity_value_max {
  constexpr T operator()() const {
    return std::numeric_limits<T>::max();
  }
};

//! Accumulator for T where accumulation is plus
template <typename T>
class GAccumulator : public Reducible<T, std::plus<T>, identity_value_zero<T>> {
  using base_type = Reducible<T, std::plus<T>, identity_value_zero<T>>;

public:
  GAccumulator(): base_type(std::plus<T>(), identity_value_zero<T>()) { }

  GAccumulator& operator+=(const T& rhs) {
    base_type::update(rhs);
    return *this;
  }

  GAccumulator& operator-=(const T& rhs) {
    base_type::update(rhs);
    return *this;
  }
};

//! Accumulator for T where accumulation is max
template <typename T>
class GReduceMax : public Reducible<T, gmax<T>, identity_value_min<T>> {
  using base_type = Reducible<T, gmax<T>, identity_value_min<T>>;

public:
  GReduceMax(): base_type(gmax<T>(), identity_value_min<T>()) { }
};

//! Accumulator for T where accumulation is min
template <typename T>
class GReduceMin : public Reducible<T, gmin<T>, identity_value_max<T>> {
  using base_type = Reducible<T, gmin<T>, identity_value_max<T>>;

public:
  GReduceMin(): base_type(gmin<T>(), identity_value_max<T>()) { }
};

//! logical AND reduction
class GReduceLogicalAnd : public Reducible<bool, std::logical_and<bool>, identity_value<bool, true>> {
  using base_type = Reducible<bool, std::logical_and<bool>, identity_value<bool, true>>;

public:
  GReduceLogicalAnd(): base_type(std::logical_and<bool>(), identity_value<bool, true>()) { }
};

//! logical OR reduction
class GReduceLogicalOr : public Reducible<bool, std::logical_or<bool>, identity_value<bool, false>> {
  using base_type = Reducible<bool, std::logical_or<bool>, identity_value<bool, false>>;

public:
  GReduceLogicalOr(): base_type(std::logical_or<bool>(), identity_value<bool, false>()) { }
};

} // namespace galois
#endif // GALOIS_REDUCTION_H
