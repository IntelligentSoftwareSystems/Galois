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

#ifndef GALOIS_LOOPS_H
#define GALOIS_LOOPS_H

#include "galois/config.h"
#include "galois/runtime/Executor_Deterministic.h"
#include "galois/runtime/Executor_DoAll.h"
#include "galois/runtime/Executor_ForEach.h"
#include "galois/runtime/Executor_OnEach.h"
#include "galois/runtime/Executor_Ordered.h"
#include "galois/runtime/Executor_ParaMeter.h"
#include "galois/worklists/WorkList.h"

namespace galois {

////////////////////////////////////////////////////////////////////////////////
// Foreach
////////////////////////////////////////////////////////////////////////////////

/**
 * Galois unordered set iterator.
 * Operator should conform to <code>fn(item, UserContext<T>&)</code> where item
 * is a value from the iteration range and T is the type of item.
 *
 * @param rangeMaker an iterate range maker typically returned by
 * <code>galois::iterate(...)</code>
 * (@see galois::iterate()). rangeMaker is a functor which when called returns a
 * range object
 * @param fn operator
 * @param args optional arguments to loop, e.g., {@see loopname}, {@see wl}
 */

template <typename RangeFunc, typename FunctionTy, typename... Args>
void for_each(const RangeFunc& rangeMaker, FunctionTy&& fn,
              const Args&... args) {
  auto tpl = std::make_tuple(args...);
  runtime::for_each_gen(rangeMaker(tpl), std::forward<FunctionTy>(fn), tpl);
}

/**
 * Standard do-all loop. All iterations should be independent.
 * Operator should conform to <code>fn(item)</code> where item is a value from
 * the iteration range.
 *
 * @param rangeMaker an iterate range maker typically returned by
 * <code>galois::iterate(...)</code>
 * (@see galois::iterate()). rangeMaker is a functor which when called returns a
 * range object
 * @param fn operator
 * @param args optional arguments to loop
 */
template <typename RangeFunc, typename FunctionTy, typename... Args>
void do_all(const RangeFunc& rangeMaker, FunctionTy&& fn, const Args&... args) {
  auto tpl = std::make_tuple(args...);
  runtime::do_all_gen(rangeMaker(tpl), std::forward<FunctionTy>(fn), tpl);
}

/**
 * Low-level parallel loop. Operator is applied for each running thread.
 * Operator should confirm to <code>fn(tid, numThreads)</code> where tid is
 * the id of the current thread and numThreads is the total number of running
 * threads.
 *
 * @param fn operator, which is never copied
 * @param args optional arguments to loop
 */
template <typename FunctionTy, typename... Args>
void on_each(FunctionTy&& fn, const Args&... args) {
  runtime::on_each_gen(std::forward<FunctionTy>(fn), std::make_tuple(args...));
}

/**
 * Preallocates hugepages on each thread.
 *
 * @param num number of pages to allocate of size {@link
 * galois::runtime::MM::hugePageSize}
 */
static inline void preAlloc(int num) {
  static const bool DISABLE_PREALLOC = false;
  if (DISABLE_PREALLOC) {
    galois::gWarn("preAlloc disabled");

  } else {
    runtime::preAlloc_impl(num);
  }
}

/**
 * Reports number of hugepages allocated by the Galois system so far. The value
 * is printing using the statistics infrastructure.
 *
 * @param label Label to associated with report at this program point
 */
static inline void reportPageAlloc(const char* label) {
  runtime::reportPageAlloc(label);
}

/**
 * Galois ordered set iterator for stable source algorithms.
 *
 * Operator should conform to <code>fn(item, UserContext<T>&)</code> where item
 * is a value from the iteration range and T is the type of item. Comparison
 * function should conform to <code>bool r = cmp(item1, item2)</code> where r is
 * true if item1 is less than or equal to item2. Neighborhood function should
 * conform to <code>nhFunc(item)</code> and should visit every element in the
 * neighborhood of active element item.
 *
 * @param b begining of range of initial items
 * @param e end of range of initial items
 * @param cmp comparison function
 * @param nhFunc neighborhood function
 * @param fn operator
 * @param loopname string to identity loop in statistics output
 */
template <typename Iter, typename Cmp, typename NhFunc, typename OpFunc>
void for_each_ordered(Iter b, Iter e, const Cmp& cmp, const NhFunc& nhFunc,
                      const OpFunc& fn, const char* loopname = 0) {
  runtime::for_each_ordered_impl(b, e, cmp, nhFunc, fn, loopname);
}

/**
 * Galois ordered set iterator for unstable source algorithms.
 *
 * Operator should conform to <code>fn(item, UserContext<T>&)</code> where item
 * is a value from the iteration range and T is the type of item. Comparison
 * function should conform to <code>bool r = cmp(item1, item2)</code> where r is
 * true if item1 is less than or equal to item2. Neighborhood function should
 * conform to <code>nhFunc(item)</code> and should visit every element in the
 * neighborhood of active element item. The stability test should conform to
 * <code>bool r = stabilityTest(item)</code> where r is true if item is a stable
 * source.
 *
 * @param b begining of range of initial items
 * @param e end of range of initial items
 * @param cmp comparison function
 * @param nhFunc neighborhood function
 * @param fn operator
 * @param stabilityTest stability test
 * @param loopname string to identity loop in statistics output
 */
template <typename Iter, typename Cmp, typename NhFunc, typename OpFunc,
          typename StableTest>
void for_each_ordered(Iter b, Iter e, const Cmp& cmp, const NhFunc& nhFunc,
                      const OpFunc& fn, const StableTest& stabilityTest,
                      const char* loopname = 0) {
  runtime::for_each_ordered_impl(b, e, cmp, nhFunc, fn, stabilityTest,
                                 loopname);
}

/**
 * Helper functor class to invoke galois::do_all on provided args
 * Can be used to choose between galois::do_all and other equivalents such as
 * std::for_each
 */
struct DoAll {
  template <typename RangeFunc, typename F, typename... Args>
  void operator()(const RangeFunc& rangeMaker, const F& f,
                  Args&&... args) const {
    galois::do_all(rangeMaker, f, std::forward<Args>(args)...);
  }
};

/**
 * Helper functor to invoke std::for_each with the same interface as
 * galois::do_all
 */

struct StdForEach {
  template <typename RangeFunc, typename F, typename... Args>
  void operator()(const RangeFunc& rangeMaker, const F& f,
                  Args&&... args) const {
    auto range = rangeMaker(std::make_tuple(args...));
    std::for_each(range.begin(), range.end(), f);
  }
};

struct ForEach {
  template <typename RangeFunc, typename F, typename... Args>
  void operator()(const RangeFunc& rangeMaker, const F& f,
                  Args&&... args) const {
    galois::for_each(rangeMaker, f, std::forward<Args>(args)...);
  }
};

template <typename Q>
struct WhileQ {
  Q m_q;

  WhileQ(Q&& q = Q()) : m_q(std::move(q)) {}

  template <typename RangeFunc, typename F, typename... Args>
  void operator()(const RangeFunc& rangeMaker, const F& f, Args&&... args) {

    auto range = rangeMaker(std::make_tuple(args...));

    m_q.push(range.begin(), range.end());

    while (!m_q.empty()) {
      auto val = m_q.pop();

      f(val, m_q);
    }
  }
};

} // namespace galois

#endif // GALOIS_LOOPS_H
