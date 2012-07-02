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

#include "Galois/Threads.h"
#include "Galois/UserContext.h"
#include "Galois/Runtime/ParallelWork.h"
#include "Galois/Runtime/LocalIterator.h"

#ifdef GALOIS_USE_EXP
#include "Galois/Runtime/ParallelWorkInline.h"
#include "Galois/Runtime/ParaMeter.h"
#include "Galois/Runtime/Deterministic.h"
#endif

#include "boost/iterator/transform_iterator.hpp"

namespace Galois {

////////////////////////////////////////////////////////////////////////////////
// Foreach
////////////////////////////////////////////////////////////////////////////////


//Iterator based versions
template<typename WLTy, typename IterTy, typename FunctionTy>
static inline void for_each(IterTy b, IterTy e, FunctionTy f, const char* loopname = 0) {
  GaloisRuntime::for_each_impl<WLTy>(b, e, f, loopname);
}

template<typename IterTy, typename FunctionTy>
static inline void for_each(IterTy b, IterTy e, FunctionTy f, const char* loopname = 0) {
  typedef GaloisRuntime::WorkList::dChunkedFIFO<256> WLTy;
  Galois::for_each<WLTy, IterTy, FunctionTy>(b, e, f, loopname);
}

//Single initial item versions
template<typename WLTy, typename InitItemTy, typename FunctionTy>
static inline void for_each(InitItemTy i, FunctionTy f, const char* loopname = 0) {
  InitItemTy wl[1];
  wl[0] = i;
  Galois::for_each<WLTy>(&wl[0], &wl[1], f, loopname);
}

template<typename InitItemTy, typename FunctionTy>
static inline void for_each(InitItemTy i, FunctionTy f, const char* loopname = 0) {
  typedef GaloisRuntime::WorkList::ChunkedFIFO<256> WLTy;
  Galois::for_each<WLTy, InitItemTy, FunctionTy>(i, f, loopname);
}
//Local based versions
template<typename WLTy, typename ConTy, typename Function>
static inline void for_each_local(ConTy& c, Function f, const char* loopname = 0) {
  typedef typename ConTy::local_iterator IterTy;
  typedef GaloisRuntime::WorkList::LocalAccessDist<IterTy, WLTy> WL;
  GaloisRuntime::for_each_impl<WL>(GaloisRuntime::LocalBounce<ConTy>(&c, true), GaloisRuntime::LocalBounce<ConTy>(&c, false), f, loopname);
}

template<typename ConTy, typename Function>
static inline void for_each_local(ConTy& c, Function f, const char* loopname = 0) {
  typedef GaloisRuntime::WorkList::dChunkedFIFO<256> WLTy;
  Galois::for_each_local<WLTy, ConTy, Function>(c, f, loopname);
}


////////////////////////////////////////////////////////////////////////////////
// do_all
// Does not modify container
// Takes advantage of tiled iterator where applicable
// Experimental!
////////////////////////////////////////////////////////////////////////////////

//Random access iterator do_all
template<typename IterTy,typename FunctionTy>
static inline void do_all_dispatch(const IterTy& begin, const IterTy& end, FunctionTy fn, const char* loopname, std::random_access_iterator_tag) {
  typedef GaloisRuntime::WorkList::RandomAccessRange<false,IterTy> WL;
  GaloisRuntime::do_all_impl<WL>(begin, end, fn, loopname);
}

//Forward iterator do_all
template<typename IterTy,typename FunctionTy>
static inline void do_all_dispatch(const IterTy& begin, const IterTy& end, FunctionTy fn, const char* loopname, std::input_iterator_tag) {
  typedef GaloisRuntime::WorkList::ForwardAccessRange<IterTy> WL;
  GaloisRuntime::do_all_impl<WL>(begin, end, fn, loopname);
}

template<typename IterTy,typename FunctionTy>
static inline void do_all(const IterTy& begin, const IterTy& end, FunctionTy fn, const char* loopname = 0) {
  if (GaloisRuntime::inGaloisForEach) {
#if 0
    GaloisRuntime::TaskContext<IterTy,FunctionTy> ctx;
    GaloisRuntime::SimpleTaskPool& pool = GaloisRuntime::getSystemTaskPool();
    pool.enqueue(ctx, begin, end, fn);
    ctx.run(pool);
#else
    std::for_each(begin, end, fn);
#endif
  } else {
    typename std::iterator_traits<IterTy>::iterator_category category;
    do_all_dispatch(begin,end,fn,loopname,category); 
  }
}

//Local iterator do_all
template<typename ConTy,typename FunctionTy>
static inline void do_all_local(ConTy& c, FunctionTy fn, const char* loopname = 0) {
  typedef typename ConTy::local_iterator IterTy;
  typedef GaloisRuntime::WorkList::LocalAccessRange<IterTy> WL;
  GaloisRuntime::do_all_impl<WL>(GaloisRuntime::LocalBounce<ConTy>(&c, true), GaloisRuntime::LocalBounce<ConTy>(&c, false), fn, loopname);
}



////////////////////////////////////////////////////////////////////////////////
// OnEach
// Low level loop executing work on each processor passing thread id and number
// of threads to the work
////////////////////////////////////////////////////////////////////////////////

template<typename FunctionTy>
static inline void on_each(FunctionTy fn, const char* loopname = 0) {
  GaloisRuntime::on_each_impl(fn, loopname);
}

////////////////////////////////////////////////////////////////////////////////
// PreAlloc
////////////////////////////////////////////////////////////////////////////////

static inline void preAlloc(int num) {
  GaloisRuntime::preAlloc_impl(num);
}

////////////////////////////////////////////////////////////////////////////////
// STL compatible-ish operators
////////////////////////////////////////////////////////////////////////////////

template<typename Predicate, typename T>
struct count_if_helper {
  GaloisRuntime::PerCPU<ptrdiff_t>& local;
  Predicate& f;
  count_if_helper(Predicate& p, GaloisRuntime::PerCPU<ptrdiff_t>& l):local(l), f(p) { }
  void operator()(const T& v) {
    if (f(v)) 
      local.get()++;
  }
};

template<class InputIterator, class Predicate>
ptrdiff_t count_if(InputIterator first, InputIterator last, Predicate pred)
{
  typedef typename std::iterator_traits<InputIterator>::value_type T;
  GaloisRuntime::PerCPU<ptrdiff_t> v;
  count_if_helper<Predicate, T> c(pred, v);
  ptrdiff_t ret = 0;
  Galois::do_all(first, last, c);
  for (unsigned i = 0; i < v.size(); ++i)
    ret += v.get(i);
  return ret;
}

//! Modify an iterator so that *it == it
template<typename Iterator>
struct NoDerefIterator: public boost::iterator_adaptor<
  NoDerefIterator<Iterator>,
  Iterator,
  Iterator,
  boost::use_default,
  const Iterator&>
{
  NoDerefIterator(): NoDerefIterator::iterator_adaptor_() { }
  explicit NoDerefIterator(Iterator it): NoDerefIterator::iterator_adaptor_(it) { }
  const Iterator& dereference() const {
    return NoDerefIterator::iterator_adaptor_::base_reference();
  }
  Iterator& dereference() {
    return NoDerefIterator::iterator_adaptor_::base_reference();
  }
};

template<typename InputIterator, class Predicate>
struct find_if_helper {
  typedef int tt_does_not_need_stats;
  typedef int tt_does_not_need_parallel_push;
  typedef int tt_does_not_need_aborts;
  typedef int tt_needs_parallel_break;

  typedef boost::optional<InputIterator> ElementTy;
  typedef GaloisRuntime::PerCPU<ElementTy> AccumulatorTy;
  AccumulatorTy& accum;
  Predicate& f;
  find_if_helper(AccumulatorTy& a, Predicate& p): accum(a), f(p) { }
  void operator()(const InputIterator& v, UserContext<InputIterator>& ctx) {
    if (f(*v)) {
      accum.get() = v;
      ctx.breakLoop();
    }
  }
};

template<class InputIterator, class Predicate>
InputIterator find_if(InputIterator first, InputIterator last, Predicate pred)
{
  typedef find_if_helper<InputIterator,Predicate> HelperTy;
  typedef typename HelperTy::AccumulatorTy AccumulatorTy;
  AccumulatorTy accum;
  HelperTy helper(accum, pred);
  Galois::for_each(NoDerefIterator<InputIterator>(first), NoDerefIterator<InputIterator>(last), helper);
  for (unsigned i = 0; i < accum.size(); ++i) {
    if (accum.get(i))
      return *accum.get(i);
  }
  return last;
}

}
#endif

