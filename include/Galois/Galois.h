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
#include "Galois/Runtime/DeterministicWork.h"
#include "Galois/Runtime/OrderedWork.h"

#ifdef GALOIS_USE_EXP
#include "Galois/Runtime/ParallelWorkInline.h"
#include "Galois/Runtime/ParaMeter.h"
#endif


namespace Galois {

static const unsigned GALOIS_DEFAULT_CHUNK_SIZE = 32;

////////////////////////////////////////////////////////////////////////////////
// Foreach
////////////////////////////////////////////////////////////////////////////////

//Iterator based versions
template<typename WLTy, typename IterTy, typename FunctionTy>
void for_each(IterTy b, IterTy e, FunctionTy f, const char* loopname = 0) {
  Galois::Runtime::for_each_impl<WLTy>(Galois::Runtime::makeStandardRange(b, e), f, loopname);
}

template<typename IterTy, typename FunctionTy>
void for_each(IterTy b, IterTy e, FunctionTy f, const char* loopname = 0) {
  typedef Galois::Runtime::WorkList::dChunkedFIFO<GALOIS_DEFAULT_CHUNK_SIZE> WLTy;
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
  typedef Galois::Runtime::WorkList::ChunkedFIFO<GALOIS_DEFAULT_CHUNK_SIZE> WLTy;
  Galois::for_each<WLTy, InitItemTy, FunctionTy>(i, f, loopname);
}

//Local based versions
template<typename WLTy, typename ConTy, typename FunctionTy>
void for_each_local(ConTy& c, FunctionTy f, const char* loopname = 0) {
  Galois::Runtime::for_each_impl<WLTy>(Galois::Runtime::makeLocalRange(c), f, loopname);
}

template<typename ConTy, typename FunctionTy>
void for_each_local(ConTy& c, FunctionTy f, const char* loopname = 0) {
  typedef Galois::Runtime::WorkList::dChunkedFIFO<GALOIS_DEFAULT_CHUNK_SIZE> WLTy;
  Galois::for_each_local<WLTy, ConTy, FunctionTy>(c, f, loopname);
}

////////////////////////////////////////////////////////////////////////////////
// do_all
// Does not modify container
////////////////////////////////////////////////////////////////////////////////

template<typename IterTy,typename FunctionTy>
FunctionTy do_all(const IterTy& begin, const IterTy& end, FunctionTy fn, const char* loopname = 0) {
  return Galois::Runtime::do_all_impl(Galois::Runtime::makeStandardRange(begin, end), fn, Galois::Runtime::EmptyFn(), false);
}

//Local iterator do_all
template<typename ConTy,typename FunctionTy>
FunctionTy do_all_local(ConTy& c, FunctionTy fn, const char* loopname = 0) {
  return Galois::Runtime::do_all_impl(Galois::Runtime::makeLocalRange(c), fn, Galois::Runtime::EmptyFn(), false);
}

////////////////////////////////////////////////////////////////////////////////
// OnEach
// Low level loop executing work on each processor passing thread id and number
// of threads to the work
////////////////////////////////////////////////////////////////////////////////

template<typename FunctionTy>
static inline void on_each(FunctionTy fn, const char* loopname = 0) {
  Galois::Runtime::on_each_impl(fn, loopname);
}

////////////////////////////////////////////////////////////////////////////////
// PreAlloc
////////////////////////////////////////////////////////////////////////////////

static inline void preAlloc(int num) {
  Galois::Runtime::preAlloc_impl(num);
}

////////////////////////////////////////////////////////////////////////////////
// Ordered Foreach 
////////////////////////////////////////////////////////////////////////////////

template<typename Iter, typename Cmp, typename NhFunc, typename OpFunc>
void for_each_ordered(Iter beg, Iter end, Cmp cmp, NhFunc nhFunc, OpFunc opFunc, const char* loopname=0) {
  Galois::Runtime::for_each_ordered_impl(beg, end, cmp, nhFunc, opFunc, loopname);
}

template<typename Iter, typename Cmp, typename NhFunc, typename OpFunc, typename StableTest>
void for_each_ordered(Iter beg, Iter end, Cmp cmp, NhFunc nhFunc, OpFunc opFunc, StableTest stabilityTest, const char* loopname=0) {
  Galois::Runtime::for_each_ordered_impl(beg, end, cmp, nhFunc, opFunc, stabilityTest, loopname);
}

} //namespace Galois
#endif
