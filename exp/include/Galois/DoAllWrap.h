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
#include "Galois/Runtime/ll/EnvCheck.h"

#ifdef GALOIS_USE_TBB
#include "tbb/parallel_for_each.h"
#endif

#include "CilkInit.h"
#include <unistd.h>

#include "llvm/Support/CommandLine.h"



namespace Galois {

enum DoAllTypes { 
  GALOIS, GALOIS_STEAL, COUPLED, CILK, OPENMP 
};

namespace cll = llvm::cl;
extern cll::opt<DoAllTypes> doAllKind;

template <DoAllTypes TYPE> 
struct DoAllImpl {
  template <typename R, typename F>
  static inline void go (const R& range, const F& func, const char* loopname) {
    std::abort ();
  }
};

template <>
struct DoAllImpl<GALOIS> {
  template <typename R, typename F>
  static inline void go (const R& range, const F& func, const char* loopname) {
    Galois::Runtime::do_all_impl (range, func, loopname, false);
  }
};

template <>
struct DoAllImpl<GALOIS_STEAL> {
  template <typename R, typename F>
  static inline void go (const R& range, const F& func, const char* loopname) {
    Galois::Runtime::do_all_impl (range, func, loopname, false);
  }
};

template <>
struct DoAllImpl<COUPLED> {
  template <typename R, typename F>
  static inline void go (const R& range, const F& func, const char* loopname) {
    // Galois::Runtime::do_all_coupled (range, func, loopname);
    std::abort ();
  }
};


#ifdef HAVE_CILK

template <>
struct DoAllImpl<CILK> {
  template <typename R, typename F>
  static inline void go (const R& range, const F& func, const char* loopname) {
    CilkInit ();
    cilk_for(auto it = range.begin (), end = range.end (); it != end; ++it) {
      func (*it);
    }
  }



};
#else 

template <> struct DoAllImpl<CILK> {
  template <typename R, typename F>
  static inline void go (const R& range, const F& func, const char* loopname) {
    GALOIS_DIE("Cilk not found\n");
  }
};
#endif

template <>
struct DoAllImpl<OPENMP> {
  template <typename R, typename F>
  static inline void go (const R& range, const F& func, const char* loopname) {
  const auto end = range.end ();
#pragma omp parallel for schedule(guided)
    for (auto it = range.begin (); it < end; ++it) {
      func (*it);
    }
  }
};

template <typename R, typename F> 
void do_all_choice (const R& range, const F& func, const char* loopname=0) {
  switch (doAllKind) {
    case GALOIS_STEAL:
      DoAllImpl<GALOIS_STEAL>::go (range, func, loopname);
      break;
    case GALOIS:
      DoAllImpl<GALOIS>::go (range, func, loopname);
      break;
    case COUPLED:
      DoAllImpl<COUPLED>::go (range, func, loopname);
      break;
    case CILK:
      DoAllImpl<CILK>::go (range, func, loopname);
      break;
    case OPENMP:
      // DoAllImpl<OPENMP>::go (range, func, loopname);
      std::abort ();
      break;
    default:
      abort ();
      break;
  }
}

} // end namespace Galois

#endif //  GALOIS_DOALL_WRAPPER_H

