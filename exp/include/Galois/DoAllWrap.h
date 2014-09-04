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
#include "Galois/GaloisForwardDecl.h"
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
  DOALL_GALOIS, DOALL_GALOIS_STEAL, DOALL_GALOIS_FOREACH, DOALL_COUPLED, DOALL_CILK, DOALL_OPENMP 
};

namespace cll = llvm::cl;
extern cll::opt<DoAllTypes> doAllKind;

void setDoAllImpl (const DoAllTypes& type);

DoAllTypes getDoAllImpl (void);

template <unsigned CZ>
struct doall_chunk_size {
  static const unsigned value = CZ;
};

template <> 
struct doall_chunk_size<0> {};

template <DoAllTypes TYPE> 
struct DoAllImpl {
  template <typename R, typename F>
  static inline void go (const R& range, const F& func, const char* loopname) {
    std::abort ();
  }
};

template <>
struct DoAllImpl<DOALL_GALOIS> {
  template <typename R, typename F>
  static inline void go (const R& range, const F& func, const char* loopname) {
    Galois::Runtime::do_all_impl (range, func, loopname, false);
  }
};

template <>
struct DoAllImpl<DOALL_GALOIS_STEAL> {
  template <typename R, typename F>
  static inline void go (const R& range, const F& func, const char* loopname) {
    Galois::Runtime::do_all_impl (range, func, loopname, false);
  }
};

template <>
struct DoAllImpl<DOALL_GALOIS_FOREACH> {

  template <typename T, typename _F>
  struct FuncWrap {
    typedef char tt_does_not_need_push;
    typedef int tt_does_not_need_aborts;

    _F func;

    template <typename C>
    void operator () (T& x, C&) {
      func (x);
    }
  };

  template <const unsigned CHUNK_SIZE, typename R, typename F>
  static inline void go (const R& range, const F& func, const char* loopname) {
    typedef typename R::value_type T;
    // typedef Galois::WorkList::dChunkedFIFO<CHUNK_SIZE, T> WL_ty;
    typedef Galois::WorkList::dChunkedLIFO<CHUNK_SIZE, T> WL_ty;
    // typedef Galois::WorkList::AltChunkedLIFO<CHUNK_SIZE, T> WL_ty;

    Galois::Runtime::for_each_gen(range, FuncWrap<T, F> {func},
        std::make_tuple(Galois::loopname(loopname), Galois::wl<WL_ty>()));
  }
};

template <>
struct DoAllImpl<DOALL_COUPLED> {
  template <const unsigned CHUNK_SIZE, typename R, typename F>
  static inline void go (const R& range, const F& func, const char* loopname) {
    Galois::Runtime::do_all_coupled (range, func, loopname, CHUNK_SIZE);
  }
};


#ifdef HAVE_CILK
template <>
struct DoAllImpl<DOALL_CILK> {
  template <typename R, typename F>
  static inline void go (const R& range, const F& func, const char* loopname) {
    CilkInit ();
    cilk_for(auto it = range.begin (), end = range.end (); it != end; ++it) {
      func (*it);
    }
  }
};
#else 
template <> struct DoAllImpl<DOALL_CILK> {
  template <typename R, typename F>
  static inline void go (const R& range, const F& func, const char* loopname) {
    GALOIS_DIE("Cilk not found\n");
  }
};
#endif

template <>
struct DoAllImpl<DOALL_OPENMP> {
  template <typename R, typename F>
  static inline void go (const R& range, const F& func, const char* loopname) {
  const auto end = range.end ();
#pragma omp parallel for schedule(guided)
    for (auto it = range.begin (); it < end; ++it) {
      func (*it);
    }
  }
};

template <typename R, typename F, typename CS> 
void do_all_choice (const R& range, const F& func, const DoAllTypes& type, const char* loopname=0, const CS& x=CS ()) {

  const unsigned CHUNK_SIZE = CS::value;

  switch (type) {
    case DOALL_GALOIS_STEAL:
      DoAllImpl<DOALL_GALOIS_STEAL>::go (range, func, loopname);
      break;
    case DOALL_GALOIS_FOREACH:
      DoAllImpl<DOALL_GALOIS_FOREACH>::go<CHUNK_SIZE> (range, func, loopname);
      break;
    case DOALL_GALOIS:
      DoAllImpl<DOALL_GALOIS>::go (range, func, loopname);
      break;
    case DOALL_COUPLED:
      DoAllImpl<DOALL_COUPLED>::go<CHUNK_SIZE> (range, func, loopname);
      break;
    case DOALL_CILK:
      DoAllImpl<DOALL_CILK>::go (range, func, loopname);
      break;
    case DOALL_OPENMP:
      // DoAllImpl<DOALL_OPENMP>::go (range, func, loopname);
      std::abort ();
      break;
    default:
      abort ();
      break;
  }
}

template <typename R, typename F, typename CS>
void do_all_choice (const R& range, const F& func, const char* loopname=0, const CS& x=CS ()) {
  do_all_choice (range, func, doAllKind, loopname, x);
}

} // end namespace Galois

#endif //  GALOIS_DOALL_WRAPPER_H
