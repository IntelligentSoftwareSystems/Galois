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

#ifndef __GALOIS_RUNTIME_ATOMIC_HELPER_FUNCTIONS_H__
#define __GALOIS_RUNTIME_ATOMIC_HELPER_FUNCTIONS_H__
#include <atomic>
#include <algorithm>
#include <vector>
namespace galois {
/** galois::atomicMax + non-atomic max calls **/
template <typename Ty>
const Ty atomicMax(std::atomic<Ty>& a, const Ty b) {
  Ty old_a = a;
  // if old value is less than new value, atomically exchange
  while (old_a < b &&
         !a.compare_exchange_weak(old_a, b, std::memory_order_relaxed))
    ;
  return old_a;
}

template <typename Ty>
const Ty max(std::atomic<Ty>& a, const Ty& b) {
  Ty old_a = a;

  if (a < b) {
    a = b;
  }
  return old_a;
}

template <typename Ty>
const Ty max(Ty& a, const Ty& b) {
  Ty old_a = a;

  if (a < b) {
    a = b;
  }
  return old_a;
}

/** galois::atomicMin **/
template <typename Ty>
const Ty atomicMin(std::atomic<Ty>& a, const Ty b) {
  Ty old_a = a;
  while (old_a > b &&
         !a.compare_exchange_weak(old_a, b, std::memory_order_relaxed))
    ;
  return old_a;
}

template <typename Ty>
const Ty min(std::atomic<Ty>& a, const Ty& b) {
  Ty old_a = a;
  if (a > b) {
    a = b;
  }
  return old_a;
}

template <typename Ty>
const Ty min(Ty& a, const Ty& b) {
  Ty old_a = a;
  if (a > b) {
    a = b;
  }
  return old_a;
}

/** galois::atomicAdd **/
template <typename Ty>
const Ty atomicAdd(std::atomic<Ty>& val, Ty delta) {
  Ty old_val = val;
  while (!val.compare_exchange_weak(old_val, old_val + delta,
                                    std::memory_order_relaxed))
    ;
  return old_val;
}

template <typename Ty>
const Ty add(std::atomic<Ty>& a, const Ty& b) {
  Ty old_a = a;
  a        = a + b;
  return old_a;
}

template <typename Ty>
const Ty add(Ty& a, std::atomic<Ty>& b) {
  Ty old_a = a;
  a        = a + b.load();
  return old_a;
}

template <typename Ty>
const Ty add(Ty& a, const Ty& b) {
  Ty old_a = a;
  a += b;
  return old_a;
}

/**
 * atomic subtraction of delta (because atomicAdd with negative numbers implies
 * a signed integer cast)
 */
template <typename Ty>
const Ty atomicSubtract(std::atomic<Ty>& val, Ty delta) {
  Ty old_val = val;
  while (!val.compare_exchange_weak(old_val, old_val - delta,
                                    std::memory_order_relaxed));
  return old_val;
}

template <typename Ty>
const Ty set(Ty& a, const Ty& b) {
  a = b;
  return a;
}

template <typename Ty>
const Ty set(std::atomic<Ty>& a, const Ty& b) {
  a = b;
  return a;
}

/** Pair Wise Average function **/
template <typename Ty>
const Ty pairWiseAvg(Ty a, Ty b) {
  return (a + b) / 2.0;
}

template <typename Ty>
void pairWiseAvg_vec(std::vector<Ty>& a_vec, std::vector<Ty>& b_vec) {
  for (unsigned i = 0; i < a_vec.size(); ++i) {
    a_vec[i] = (a_vec[i] + b_vec[i]) / 2.0;
  }
}

template <typename Ty>
void resetVec(Ty& a_arr) {
  // std::for_each(a_arr.begin(), a_arr.end(),[](Ty &ele){ele = 0;} );
  std::fill(a_arr.begin(), a_arr.end(), 0);
}

template <typename Ty>
void pairWiseAvg_vec(Ty& a_arr, Ty& b_arr) {
  for (unsigned i = 0; i < a_arr.size(); ++i) {
    a_arr[i] = (a_arr[i] + b_arr[i]) / 2.0;
  }
}

template <typename Ty>
void addArray(Ty& a_arr, Ty& b_arr) {
  for (unsigned i = 0; i < a_arr.size(); ++i) {
    a_arr[i] = (a_arr[i] + b_arr[i]);
  }
}

template <typename Ty>
void resetVec(std::vector<Ty>& a_vec) {
  std::for_each(a_vec.begin(), a_vec.end(), [](Ty& ele) { ele = 0; });
}

// like std::inner_product
template <typename ItrTy, typename Ty>
Ty innerProduct(ItrTy a_begin, ItrTy a_end, ItrTy b_begin, Ty init_value) {
  auto jj = b_begin;
  for (auto ii = a_begin; ii != a_end; ++ii, ++jj) {
    init_value += (*ii) * (*jj);
  }
  return init_value;
}

// like std::inner_product
template <typename ItrTy, typename Ty>
Ty innerProduct(ItrTy& a_arr, ItrTy& b_arr, Ty init_value) {
  auto jj = b_arr.begin();
  for (auto ii = a_arr.begin(); ii != a_arr.end(); ++ii, ++jj) {
    init_value += (*ii) * (*jj);
  }
  return init_value;
}

template <typename Ty>
void reset(Ty& var, Ty val) {
  var = val;
}

template <typename Ty>
void reset(std::atomic<Ty>& var, Ty val) {
  var = val;
}
} // end namespace galois
#endif
