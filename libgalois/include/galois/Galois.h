/** Galois user interface -*- C++ -*-
 * @file
 * This is the only file to include for basic Galois functionality.
 *
 * @section License
 *
 * This file is part of Galois.  Galoisis a framework to exploit
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
 * Copyright (C) 2017, The University of Texas at Austin. All rights
 * reserved.
 *
 */

#ifndef GALOIS_GALOIS_H
#define GALOIS_GALOIS_H

#include "galois/runtime/Init.h"
#include "galois/runtime/Executor_Deterministic.h"
#include "galois/runtime/Executor_DoAll.h"
#include "galois/runtime/Executor_ForEach.h"
#include "galois/runtime/Executor_OnEach.h"
#include "galois/runtime/Executor_Ordered.h"
#include "galois/runtime/Mem.h"

#include "galois/worklists/WorkList.h"

#ifdef GALOIS_USE_EXP
//#include "galois/runtime/Executor_BulkSynchronous.h"
//#include "galois/runtime/Executor_ParaMeter.h"
#endif

#include <utility>
#include <tuple>

/**
 * Main Galois namespace. All the core Galois functionality will be found in here.
 */
namespace galois {

/**
 * explicit class to initialize the Galois Runtime
 * Runtime is destroyed when this object is destroyed
 */
class SharedMemSys: public runtime::SharedMemRuntime<runtime::StatManager> {

public:
  explicit SharedMemSys(void);

  ~SharedMemSys(void);
};

////////////////////////////////////////////////////////////////////////////////
// Foreach
////////////////////////////////////////////////////////////////////////////////

/**
 * Galois unordered set iterator.
 * Operator should conform to <code>fn(item, UserContext<T>&)</code> where item is a value from the iteration
 * range and T is the type of item.
 *
 * @param rangeMaker an iterate range maker typically returned by <code>galois::iterate(...)</code> 
 * (@see galois::iterate()). rangeMaker is a functor which when called returns a range object
 * @param fn operator
 * @param args optional arguments to loop, e.g., {@see loopname}, {@see wl}
 */

template <typename RangeFunc, typename FunctionTy, typename... Args>
void for_each(const RangeFunc& rangeMaker, const FunctionTy& fn, const Args&... args) {
  auto tpl = std::make_tuple(args...);
  runtime::for_each_gen(rangeMaker(tpl), fn, tpl);
}


template <typename RangeFunc, typename FunctionTy, typename... Args>
/**
 * Standard do-all loop. All iterations should be independent.
 * Operator should conform to <code>fn(item)</code> where item is a value from the iteration range.
 *
 * @param rangeMaker an iterate range maker typically returned by <code>galois::iterate(...)</code> 
 * (@see galois::iterate()). rangeMaker is a functor which when called returns a range object
 * @param fn operator
 * @param args optional arguments to loop
 */
void do_all(const RangeFunc& rangeMaker, const FunctionTy& fn, const Args&... args) {
  auto tpl = std::make_tuple(args...);
  runtime::do_all_gen(rangeMaker(tpl), fn, tpl);
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
template<typename FunctionTy, typename... Args>
void on_each(const FunctionTy& fn, const Args&... args) {
  runtime::on_each_impl(fn, std::make_tuple(args...));
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
template<typename FunctionTy, typename... Args>
void on_each(FunctionTy& fn, const Args&... args) {
  runtime::on_each_impl(fn, std::make_tuple(args...));
}

/**
 * Preallocates hugepages on each thread.
 *
 * @param num number of pages to allocate of size {@link galois::runtime::MM::hugePageSize}
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
 * Reports number of hugepages allocated by the Galois system so far. The value is printing using
 * the statistics infrastructure. 
 *
 * @param label Label to associated with report at this program point
 */
static inline void reportPageAlloc(const char* label) {
  runtime::reportPageAlloc(label);
}

/**
 * Galois ordered set iterator for stable source algorithms.
 *
 * Operator should conform to <code>fn(item, UserContext<T>&)</code> where item is a value from the iteration
 * range and T is the type of item. Comparison function should conform to <code>bool r = cmp(item1, item2)</code>
 * where r is true if item1 is less than or equal to item2. Neighborhood function should conform to
 * <code>nhFunc(item)</code> and should visit every element in the neighborhood of active element item.
 *
 * @param b begining of range of initial items
 * @param e end of range of initial items
 * @param cmp comparison function
 * @param nhFunc neighborhood function
 * @param fn operator
 * @param loopname string to identity loop in statistics output
 */
template<typename Iter, typename Cmp, typename NhFunc, typename OpFunc>
void for_each_ordered(Iter b, Iter e, const Cmp& cmp, const NhFunc& nhFunc, const OpFunc& fn, const char* loopname=0) {
  runtime::for_each_ordered_impl(b, e, cmp, nhFunc, fn, loopname);
}

/**
 * Galois ordered set iterator for unstable source algorithms.
 *
 * Operator should conform to <code>fn(item, UserContext<T>&)</code> where item is a value from the iteration
 * range and T is the type of item. Comparison function should conform to <code>bool r = cmp(item1, item2)</code>
 * where r is true if item1 is less than or equal to item2. Neighborhood function should conform to
 * <code>nhFunc(item)</code> and should visit every element in the neighborhood of active element item.
 * The stability test should conform to <code>bool r = stabilityTest(item)</code> where r is true if
 * item is a stable source.
 *
 * @param b begining of range of initial items
 * @param e end of range of initial items
 * @param cmp comparison function
 * @param nhFunc neighborhood function
 * @param fn operator
 * @param stabilityTest stability test
 * @param loopname string to identity loop in statistics output
 */
template<typename Iter, typename Cmp, typename NhFunc, typename OpFunc, typename StableTest>
void for_each_ordered(Iter b, Iter e, const Cmp& cmp, const NhFunc& nhFunc, const OpFunc& fn, const StableTest& stabilityTest, const char* loopname=0) {
  runtime::for_each_ordered_impl(b, e, cmp, nhFunc, fn, stabilityTest, loopname);
}

} //namespace galois
#endif
