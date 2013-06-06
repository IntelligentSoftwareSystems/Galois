/** Galois user interface -*- C++ -*-
 * @file
 * This is the only file to include for basic Galois functionality.
 *
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
 */
#ifndef GALOIS_GALOIS_H
#define GALOIS_GALOIS_H

#include "Galois/WorkList/WorkList.h"
#include "Galois/UserContext.h"
#include "Galois/Threads.h"
#include "Galois/Runtime/ParallelWork.h"
#include "Galois/Runtime/DoAll.h"
#include "Galois/Runtime/DeterministicWork.h"
#include "Galois/Runtime/OrderedWork.h"

#ifdef GALOIS_USE_EXP
#include "Galois/Runtime/ParallelWorkInline.h"
#include "Galois/Runtime/ParaMeter.h"
#endif

/**
 * Main Galois namespace. All the core Galois functionality will be found in here.
 */
namespace Galois {

static const unsigned GALOIS_DEFAULT_CHUNK_SIZE = 32;

////////////////////////////////////////////////////////////////////////////////
// Foreach
////////////////////////////////////////////////////////////////////////////////

/**
 * Galois unordered set iterator.
 * Operator should conform to <code>fn(item, UserContext<T>&)</code> where item is a value from the iteration
 * range and T is the type of item.
 *
 * @tparam WLTy Worklist policy {@see Galois::WorkList}
 * @param b begining of range of initial items
 * @param e end of range of initial items
 * @param fn operator
 * @param loopname string to identity loop in statistics output
 */
template<typename WLTy, typename IterTy, typename FunctionTy>
void for_each(IterTy b, IterTy e, FunctionTy fn, const char* loopname = 0) {
  Runtime::for_each_impl<WLTy>(Runtime::makeStandardRange(b, e), fn, loopname);
}

/**
 * Galois unordered set iterator with default worklist policy.
 * Operator should conform to <code>fn(item, UserContext<T>&)</code> where item is a value from the iteration
 * range and T is the type of item.
 *
 * @param b begining of range of initial items
 * @param e end of range of initial items
 * @param fn operator
 * @param loopname string to identity loop in statistics output
 */
template<typename IterTy, typename FunctionTy>
void for_each(IterTy b, IterTy e, FunctionTy fn, const char* loopname = 0) {
  typedef WorkList::dChunkedFIFO<GALOIS_DEFAULT_CHUNK_SIZE> WLTy;
  for_each<WLTy, IterTy, FunctionTy>(b, e, fn, loopname);
}

/**
 * Galois unordered set iterator.
 * Operator should conform to <code>fn(item, UserContext<T>&)</code> where item is i and T 
 * is the type of item.
 *
 * @tparam WLTy Worklist policy {@link Galois::WorkList}
 * @param i initial item
 * @param fn operator
 * @param loopname string to identity loop in statistics output
 */
template<typename WLTy, typename InitItemTy, typename FunctionTy>
void for_each(InitItemTy i, FunctionTy fn, const char* loopname = 0) {
  InitItemTy wl[1] = {i};
  for_each<WLTy>(&wl[0], &wl[1], fn, loopname);
}

/**
 * Galois unordered set iterator with default worklist policy.
 * Operator should conform to <code>fn(item, UserContext<T>&)</code> where item is i and T 
 * is the type of item.
 *
 * @param i initial item
 * @param fn operator
 * @param loopname string to identity loop in statistics output
 */
template<typename InitItemTy, typename FunctionTy>
void for_each(InitItemTy i, FunctionTy fn, const char* loopname = 0) {
  typedef WorkList::dChunkedFIFO<GALOIS_DEFAULT_CHUNK_SIZE> WLTy;
  for_each<WLTy, InitItemTy, FunctionTy>(i, fn, loopname);
}

/**
 * Galois unordered set iterator with locality-aware container.
 * Operator should conform to <code>fn(item, UserContext<T>&)</code> where item is an element of c and T 
 * is the type of item.
 *
 * @tparam WLTy Worklist policy {@link Galois::WorkList}
 * @param c locality-aware container
 * @param fn operator
 * @param loopname string to identity loop in statistics output
 */
template<typename WLTy, typename ConTy, typename FunctionTy>
void for_each_local(ConTy& c, FunctionTy fn, const char* loopname = 0) {
  Runtime::for_each_impl<WLTy>(Runtime::makeLocalRange(c), fn, loopname);
}

/**
 * Galois unordered set iterator with locality-aware container and default worklist policy.
 * Operator should conform to <code>fn(item, UserContext<T>&)</code> where item is an element of c and T 
 * is the type of item.
 *
 * @param c locality-aware container
 * @param fn operator
 * @param loopname string to identity loop in statistics output
 */
template<typename ConTy, typename FunctionTy>
void for_each_local(ConTy& c, FunctionTy fn, const char* loopname = 0) {
  typedef WorkList::dChunkedFIFO<GALOIS_DEFAULT_CHUNK_SIZE> WLTy;
  for_each_local<WLTy, ConTy, FunctionTy>(c, fn, loopname);
}

/**
 * Standard do-all loop. All iterations should be independent.
 * Operator should conform to <code>fn(item)</code> where item is a value from the iteration range.
 *
 * @param b beginning of range of items
 * @param e end of range of items
 * @param fn operator
 * @param loopname string to identify loop in statistics output
 * @returns fn
 */
template<typename IterTy,typename FunctionTy>
FunctionTy do_all(const IterTy& b, const IterTy& e, FunctionTy fn, const char* loopname = 0) {
  return Runtime::do_all_impl(Runtime::makeStandardRange(b, e), fn);
}

/**
 * Standard do-all loop with locality-aware container. All iterations should be independent.
 * Operator should conform to <code>fn(item)</code> where item is an element of c.
 *
 * @param c locality-aware container
 * @param fn operator
 * @param loopname string to identify loop in statistics output
 * @returns fn
 */
template<typename ConTy,typename FunctionTy>
FunctionTy do_all_local(ConTy& c, FunctionTy fn, const char* loopname = 0) {
  return Runtime::do_all_impl(Runtime::makeLocalRange(c), fn);
}

/**
 * Low-level parallel loop. Operator is applied for each running thread. Operator
 * should confirm to <code>fn(tid, numThreads)</code> where tid is the id of the current thread and
 * numThreads is the total number of running threads.
 *
 * @param fn operator
 * @param loopname string to identify loop in statistics output
 */
template<typename FunctionTy>
static inline void on_each(FunctionTy fn, const char* loopname = 0) {
  Runtime::on_each_impl(fn, loopname);
}

/**
 * Preallocates pages on each thread.
 *
 * @param num number of pages to allocate of size {@link Galois::Runtime::MM::pageSize}
 */
static inline void preAlloc(int num) {
  Runtime::preAlloc_impl(num);
}

/**
 * Reports number of pages allocated by the Galois system so far. The value is printing using
 * the statistics infrastructure. 
 *
 * @param label Label to associated with report at this program point
 */
static inline void reportPageAlloc(const char* label) {
  Runtime::reportPageAlloc(label);
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
void for_each_ordered(Iter b, Iter e, Cmp cmp, NhFunc nhFunc, OpFunc fn, const char* loopname=0) {
  Runtime::for_each_ordered_impl(b, e, cmp, nhFunc, fn, loopname);
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
void for_each_ordered(Iter b, Iter e, Cmp cmp, NhFunc nhFunc, OpFunc fn, StableTest stabilityTest, const char* loopname=0) {
  Runtime::for_each_ordered_impl(b, e, cmp, nhFunc, fn, stabilityTest, loopname);
}

} //namespace Galois
#endif
