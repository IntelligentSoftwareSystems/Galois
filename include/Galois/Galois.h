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

#include "Galois/config.h"
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

#include GALOIS_CXX11_STD_HEADER(utility)
#include GALOIS_CXX11_STD_HEADER(type_traits)
#include GALOIS_CXX11_STD_HEADER(tuple)

/**
 * Main Galois namespace. All the core Galois functionality will be found in here.
 */
namespace Galois {

/**
 * Specify name to appear in statistics. Optional argument to {@link do_all()}
 * and {@link for_each()} loops.
 */
struct loopname {
  const char* n;
  loopname(const char* n = 0) :n(n) {}
};

/**
 * Specify whether @{link do_all()} loops should perform work-stealing. Optional
 * argument to {@link do_all()} loops.
 */
struct do_all_steal {
  bool b;
  do_all_steal(bool b = false) :b(b) {}
};

struct wl_tag {};

/**
 * Specify worklist to use. Optional argument to {@link for_each()} loops.
 */
template<typename WLTy>
struct wl : public wl_tag {
  typedef WLTy WL;
};


namespace HIDDEN {

static constexpr unsigned GALOIS_DEFAULT_CHUNK_SIZE = 32;
typedef WorkList::dChunkedFIFO<GALOIS_DEFAULT_CHUNK_SIZE> defaultWL;

template <typename T, typename S, int i = std::tuple_size<T>::value - 1>
struct tuple_index {
  enum {
    value = std::is_base_of<S, typename std::tuple_element<i, T>::type>::value 
    || std::is_same<S, typename std::tuple_element<i, T>::type>::value
    ? i : tuple_index<T, S, i-1>::value
  };
};

template <typename T, typename S>
struct tuple_index<T, S, -1> {
  enum { value = -1 };
};

template<typename RangeTy, typename FunctionTy, typename Tuple>
void for_each_gen(RangeTy r, FunctionTy fn, Tuple tpl) {
  typedef Tuple tupleType;
  static_assert(-1 == tuple_index<tupleType, char*>::value, "old loopname");
  static_assert(-1 == tuple_index<tupleType, char const*>::value, "old loopname");
  static_assert(-1 == tuple_index<tupleType, bool>::value, "old steal");
  // std::cout << tuple_index<tupleType, char*>::value << " "
  //           << tuple_index<tupleType, char const*>::value << "\n";
  constexpr unsigned iloopname = tuple_index<tupleType, loopname>::value;
  constexpr unsigned iwl = tuple_index<tupleType, wl_tag>::value;
  const char* ln = std::get<iloopname>(tpl).n;
  typedef typename std::tuple_element<iwl,tupleType>::type::WL WLTy;
  Runtime::for_each_impl<WLTy>(r, fn, ln);
}

template<typename RangeTy, typename FunctionTy, typename Tuple>
FunctionTy do_all_gen(RangeTy r, FunctionTy fn, Tuple tpl) {
  typedef Tuple tupleType;
  static_assert(-1 == tuple_index<tupleType, char*>::value, "old loopname");
  static_assert(-1 == tuple_index<tupleType, char const*>::value, "old loopname");
  static_assert(-1 == tuple_index<tupleType, bool>::value, "old steal");
  // std::cout << tuple_index<tupleType, char*>::value << " "
  //           << tuple_index<tupleType, char const*>::value << "\n";
  constexpr unsigned iloopname = tuple_index<tupleType, loopname>::value;
  constexpr unsigned isteal = tuple_index<tupleType, do_all_steal>::value;
  const char* ln = std::get<iloopname>(tpl).n;
  bool steal = std::get<isteal>(tpl).b;
  return Runtime::do_all_impl(r, fn, ln, steal);
}

} // namespace HIDDEN

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
 * @param args optional arguments to loop, e.g., {@see loopname}, {@see wl}
 */
template<typename IterTy, typename FunctionTy, typename... Args>
void for_each(IterTy b, IterTy e, FunctionTy fn, Args... args) {
  HIDDEN::for_each_gen(Runtime::makeStandardRange(b,e), fn, std::make_tuple(loopname(), wl<HIDDEN::defaultWL>(), args...));
}

/**
 * Galois unordered set iterator.
 * Operator should conform to <code>fn(item, UserContext<T>&)</code> where item is i and T 
 * is the type of item.
 *
 * @tparam WLTy Worklist policy {@link Galois::WorkList}
 * @param i initial item
 * @param fn operator
 * @param args optional arguments to loop
 */
template<typename ItemTy, typename FunctionTy, typename... Args>
void for_each(ItemTy i, FunctionTy fn, Args... args) {
  ItemTy iwl[1] = {i};
  HIDDEN::for_each_gen(Runtime::makeStandardRange(&iwl[0], &iwl[1]), fn, std::make_tuple(loopname(), wl<HIDDEN::defaultWL>(), args...));
}

/**
 * Galois unordered set iterator with locality-aware container.
 * Operator should conform to <code>fn(item, UserContext<T>&)</code> where item is an element of c and T 
 * is the type of item.
 *
 * @tparam WLTy Worklist policy {@link Galois::WorkList}
 * @param c locality-aware container
 * @param fn operator
 * @param args optional arguments to loop
 */
template<typename ConTy, typename FunctionTy, typename... Args>
void for_each_local(ConTy& c, FunctionTy fn, Args... args) {
  HIDDEN::for_each_gen(Runtime::makeLocalRange(c), fn, std::make_tuple(loopname(), wl<HIDDEN::defaultWL>(), args...));
}

/**
 * Standard do-all loop. All iterations should be independent.
 * Operator should conform to <code>fn(item)</code> where item is a value from the iteration range.
 *
 * @param b beginning of range of items
 * @param e end of range of items
 * @param fn operator
 * @param args optional arguments to loop
 * @returns fn
 */
template<typename IterTy,typename FunctionTy, typename... Args>
FunctionTy do_all(const IterTy& b, const IterTy& e, FunctionTy fn, Args... args) {
  return HIDDEN::do_all_gen(Runtime::makeStandardRange(b, e), fn, std::make_tuple(loopname(), do_all_steal(), args...));
}

/**
 * Standard do-all loop with locality-aware container. All iterations should be independent.
 * Operator should conform to <code>fn(item)</code> where item is an element of c.
 *
 * @param c locality-aware container
 * @param fn operator
 * @param args optional arguments to loop
 * @returns fn
 */
template<typename ConTy,typename FunctionTy, typename... Args>
FunctionTy do_all_local(ConTy& c, FunctionTy fn, Args... args) {
  return HIDDEN::do_all_gen(Runtime::makeLocalRange(c), fn, std::make_tuple(loopname(), do_all_steal(), args...));
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
void for_each_ordered(Iter b, Iter e, const Cmp& cmp, const NhFunc& nhFunc, const OpFunc& fn, const char* loopname=0) {
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
void for_each_ordered(Iter b, Iter e, const Cmp& cmp, const NhFunc& nhFunc, const OpFunc& fn, const StableTest& stabilityTest, const char* loopname=0) {
  Runtime::for_each_ordered_impl(b, e, cmp, nhFunc, fn, stabilityTest, loopname);
}

} //namespace Galois
#endif
