/** DoAll wrapper -*- C++ -*-
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
#ifndef GALOIS_DOALL_WRAPPER_H
#define GALOIS_DOALL_WRAPPER_H

#include "Galois/Galois.h"
#include "Galois/Runtime/DoAllCoupled.h"

#ifdef GALOIS_USE_TBB
#include "tbb/parallel_for_each.h"
#endif

#if defined(__INTEL_COMPILER)
#include <cilk/cilk.h>
#endif

#include "llvm/Support/CommandLine.h"

namespace cll = llvm::cl;


namespace Galois {

enum DoAllTypes { 
  GALOIS_STEAL, GALOIS, COUPLED, CILK, TBB 
};

static cll::opt<DoAllTypes> doAllKind (
    cll::desc ("DoAll Implementation"),
    cll::values (
      clEnumVal (GALOIS_STEAL, "GALOIS_STEAL"),
      clEnumVal (GALOIS, "GALOIS"),
      clEnumVal (COUPLED, "COUPLED"),
      clEnumVal (CILK, "CILK"),
      clEnumVal (TBB, "TBB"),
      clEnumValEnd),
    cll::init (GALOIS_STEAL));

template <DoAllTypes TYPE> 
struct DoAllImpl {
  template <typename I, typename F>
  static inline void go (I beg, I end, F func, const char* loopname) {
    std::abort ();
  }

  template <typename PW, typename F>
  static inline void go (PW& perThrdWL, F func, const char* loopname) {
    std::abort ();
  }
};

template <>
struct DoAllImpl<GALOIS_STEAL> {
  template <typename I, typename F>
  static inline void go (I beg, I end, F func, const char* loopname) {
    GaloisRuntime::do_all_impl<true> (beg, end, func, loopname);
  }

  template <typename PW, typename F>
  static inline void go (PW& perThrdWL, F func, const char* loopname) {
    go (perThrdWL.begin_all (), perThrdWL.end_all (), func, loopname);
  }
};

template <>
struct DoAllImpl<GALOIS> {
  template <typename I, typename F>
  static inline void go (I beg, I end, F func, const char* loopname) {
    GaloisRuntime::do_all_impl<false> (beg, end, func, loopname);
  }

  template <typename PW, typename F>
  static inline void go (PW& perThrdWL, F func, const char* loopname) {
    go (perThrdWL.begin_all (), perThrdWL.end_all (), func, loopname);
  }
};

template <>
struct DoAllImpl<COUPLED> {
  template <typename I, typename F>
  static inline void go (I beg, I end, F func, const char* loopname) {
    GaloisRuntime::do_all_coupled (beg, end, func, loopname);
  }

  template <typename PW, typename F>
  static inline void go (PW& perThrdWL, F func, const char* loopname) {
    GaloisRuntime::do_all_coupled (perThrdWL, func, loopname);
  }
};


// #ifdef GALOIS_USE_TBB
// 
// template <>
// struct DoAllImpl<TBB> {
  // template <typename I, typename F>
  // static inline void go (I beg, I end, F func, const char* loopname) {
    // int n = Galois::getActiveThreads ();
    // tbb::task_scheduler_init  t(n);
    // tbb::parallel_for_each (beg, end, func);
  // }
// };
// 
// #endif

#if defined(__INTEL_COMPILER)

template <>
struct DoAllImpl<CILK> {

  template <typename I, typename F>
  static inline void go (I beg, I end, F func, const char* loopname) {
    cilk_for(I it = beg; it != end; ++it) {
      func (*it);
    }
  }

  template <typename PW, typename F>
  static inline void go (PW& perThrdWL, F func, const char* loopname) {
    go (perThrdWL.begin_all (), perThrdWL.end_all (), func, loopname);
  }
};

#endif

template <typename I, typename F> 
void do_all_choice (I beg, I end, F func, const char* loopname=0) {
  switch (doAllKind) {
    case GALOIS_STEAL:
      DoAllImpl<GALOIS_STEAL>::go (beg, end, func, loopname);
      break;
    case GALOIS:
      DoAllImpl<GALOIS>::go (beg, end, func, loopname);
      break;
    case COUPLED:
      DoAllImpl<COUPLED>::go (beg, end, func, loopname);
      break;
    case CILK:
      DoAllImpl<CILK>::go (beg, end, func, loopname);
      break;
    case TBB:
      DoAllImpl<TBB>::go (beg, end, func, loopname);
      break;
    default:
      abort ();
      break;
  }
}

template <typename PW, typename F> 
void do_all_choice (PW& perThrdWL, F func, const char* loopname=0) {
  switch (doAllKind) {
    case GALOIS_STEAL:
      DoAllImpl<GALOIS_STEAL>::go (perThrdWL, func, loopname);
      break;
    case GALOIS:
      DoAllImpl<GALOIS>::go (perThrdWL, func, loopname);
      break;
    case COUPLED:
      DoAllImpl<COUPLED>::go (perThrdWL, func, loopname);
      break;
    case CILK:
      DoAllImpl<CILK>::go (perThrdWL, func, loopname);
      break;
    case TBB:
      DoAllImpl<TBB>::go (perThrdWL, func, loopname);
      break;
    default:
      abort ();
      break;
  }
}

} // end namespace Galois

#endif //  GALOIS_DOALL_WRAPPER_H

