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

#include "Galois/UserContext.h"
#include "Galois/Threads.h"
#include "Galois/Runtime/ParallelWork.h"
#include "Galois/Runtime/DoAll.h"
#include "Galois/Runtime/LocalIterator.h"
#include "Galois/Runtime/DeterministicWork.h"
#include "Galois/Runtime/OrderedWork.h"

#ifdef GALOIS_USE_EXP
#include "Galois/Runtime/ParallelWorkInline.h"
#include "Galois/Runtime/ParaMeter.h"
#endif

namespace Galois {

////////////////////////////////////////////////////////////////////////////////
// Foreach
////////////////////////////////////////////////////////////////////////////////


//Iterator based versions
template<typename WLTy, typename IterTy, typename FunctionTy>
void for_each(IterTy b, IterTy e, FunctionTy f, const char* loopname = 0) {
  GaloisRuntime::for_each_impl<WLTy>(b, e, f, loopname);
}

template<typename IterTy, typename FunctionTy>
void for_each(IterTy b, IterTy e, FunctionTy f, const char* loopname = 0) {
  typedef GaloisRuntime::WorkList::dChunkedFIFO<256> WLTy;
  Galois::for_each<WLTy, IterTy, FunctionTy>(b, e, f, loopname);
}

//Single initial item versions
template<typename WLTy, typename InitItemTy, typename FunctionTy>
void for_each(InitItemTy i, FunctionTy f, const char* loopname = 0) {
  InitItemTy wl[1] = {i};
  Galois::for_each<WLTy>(&wl[0], &wl[1], f, loopname);
}

template<typename InitItemTy, typename FunctionTy>
void for_each(InitItemTy i, FunctionTy f, const char* loopname = 0) {
  typedef GaloisRuntime::WorkList::ChunkedFIFO<256> WLTy;
  Galois::for_each<WLTy, InitItemTy, FunctionTy>(i, f, loopname);
}
//Local based versions
template<typename WLTy, typename ConTy, typename Function>
void for_each_local(ConTy& c, Function f, const char* loopname = 0) {
  typedef typename ConTy::local_iterator IterTy;
  typedef GaloisRuntime::WorkList::LocalAccessDist<IterTy, WLTy> WL;
  GaloisRuntime::for_each_impl<WL>(GaloisRuntime::LocalBounce<ConTy>(&c, true), GaloisRuntime::LocalBounce<ConTy>(&c, false), f, loopname);
}

template<typename ConTy, typename Function>
void for_each_local(ConTy& c, Function f, const char* loopname = 0) {
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
  GaloisRuntime::do_all_impl_old<WL>(begin, end, fn, loopname);
}

//Forward iterator do_all
template<typename IterTy,typename FunctionTy>
static inline void do_all_dispatch(const IterTy& begin, const IterTy& end, FunctionTy fn, const char* loopname, std::input_iterator_tag) {
  typedef GaloisRuntime::WorkList::ForwardAccessRange<IterTy> WL;
  GaloisRuntime::do_all_impl_old<WL>(begin, end, fn, loopname);
}

template<typename IterTy,typename FunctionTy>
FunctionTy do_all(const IterTy& begin, const IterTy& end, FunctionTy fn, const char* loopname = 0) {
  if (GaloisRuntime::inGaloisForEach) {
    return std::for_each(begin, end, fn);
  } else {
    //typename std::iterator_traits<IterTy>::iterator_category category;
    //do_all_dispatch(begin,end,fn,loopname,category); 
    return GaloisRuntime::do_all_impl(begin, end, fn, GaloisRuntime::EmptyFn(), false);
  }
}

//Local iterator do_all
template<typename ConTy,typename FunctionTy>
static inline void do_all_local(ConTy& c, FunctionTy fn, const char* loopname = 0) {
  typedef typename ConTy::local_iterator IterTy;
  typedef GaloisRuntime::WorkList::LocalAccessRange<IterTy> WL;
  GaloisRuntime::do_all_impl_old<WL>(GaloisRuntime::LocalBounce<ConTy>(&c, true), GaloisRuntime::LocalBounce<ConTy>(&c, false), fn, loopname);
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

} //namespace Galois
#endif
