/** DoAll wrapper -*- C++ -*-
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
 * published by the Free Software Foundation, version 2.1 of the
 * License.
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
 * Copyright (C) 2015, The University of Texas at Austin. All rights
 * reserved.
 *
 */

#ifndef GALOIS_DOALL_WRAPPER_H
#define GALOIS_DOALL_WRAPPER_H

#include "Galois/Galois.h"
#include "Galois/GaloisForwardDecl.h"
#include "Galois/OrderedTraits.h"
#include "Galois/Runtime/DoAllCoupled.h"
#include "Galois/Substrate/EnvCheck.h"

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


template <DoAllTypes TYPE> 
struct DoAllImpl {
  template <typename R, typename F>
  static inline void go (const R& range, const F& func, const char* loopname) {
    std::abort ();
  }
};

template <>
struct DoAllImpl<DOALL_GALOIS> {
  template <typename R, typename F, typename... Args>
  static inline void go (const R& range, const F& func, const Args&... args) {
    Galois::Runtime::do_all_gen (range, func, 
        std::make_tuple (do_all_steal<false> (), args...));
  }
};

template <>
struct DoAllImpl<DOALL_GALOIS_STEAL> {
  template <typename R, typename F, typename... Args>
  static inline void go (const R& range, const F& func, const Args&... args) {
    Galois::Runtime::do_all_gen (range, func, 
        std::make_tuple (do_all_steal<true> (), args...));
  }
};

template <>
struct DoAllImpl<DOALL_GALOIS_FOREACH> {

  template <typename T, typename _F>
  struct FuncWrap {
    _F func;

    template <typename C>
    void operator () (T& x, C&) {
      func (x);
    }
  };

  template <typename R, typename F, typename... Args>
  static inline void go (const R& range, const F& func, const Args&... args) {

    using T = typename R::value_type;

    auto argsTuple = std::make_tuple (args...);
    using ArgsTuple = decltype (argsTuple);

    unsigned CHUNK_SIZE = typename get_type_by_supertype<chunk_size_tag, ArgsTuple>::type::value;

    using WL_ty =  Galois::WorkList::AltChunkedLIFO<CHUNK_SIZE, T>;

    Galois::Runtime::for_each_gen(range, FuncWrap<T, F> {func},
        std::make_tuple(Galois::wl<WL_ty>(), 
          does_not_need_push (),
          does_not_need_aborts (),
          args...));
  }
};

template <>
struct DoAllImpl<DOALL_COUPLED> {
  template <typename R, typename F, typename... Args>
  static inline void go (const R& range, const F& func, const Args&... args) {
    Galois::Runtime::do_all_coupled (range, func, args...);
  }
};


#ifdef HAVE_CILK
template <>
struct DoAllImpl<DOALL_CILK> {
  template <typename R, typename F, typename... Args>
  static inline void go (const R& range, const F& func, const Args&... args) {
    CilkInit ();
    cilk_for(auto it = range.begin (), end = range.end (); it != end; ++it) {
      func (*it);
    }
  }
};
#else 
template <> struct DoAllImpl<DOALL_CILK> {
  template <typename R, typename F, typename... Args>
  static inline void go (const R& range, const F& func, const Args&... args) {
    GALOIS_DIE("Cilk not found\n");
  }
};
#endif

template <>
struct DoAllImpl<DOALL_OPENMP> {
  template <typename R, typename F, typename... Args>
  static inline void go (const R& range, const F& func, const Args&... args) {
  const auto end = range.end ();
#pragma omp parallel for schedule(guided)
    for (auto it = range.begin (); it < end; ++it) {
      func (*it);
    }
  }
};

template <typename R, typename F, typename... Args> 
void do_all_choice (const R& range, const F& func, const DoAllTypes& type, const Args&... args) {

  switch (type) {
    case DOALL_GALOIS_STEAL:
      DoAllImpl<DOALL_GALOIS_STEAL>::go (range, func, args...);
      break;
    case DOALL_GALOIS_FOREACH:
      DoAllImpl<DOALL_GALOIS_FOREACH>::go<CHUNK_SIZE> (range, func, args...);
      break;
    case DOALL_GALOIS:
      DoAllImpl<DOALL_GALOIS>::go (range, func, args...);
      break;
    case DOALL_COUPLED:
      DoAllImpl<DOALL_COUPLED>::go<CHUNK_SIZE> (range, func, args...);
      break;
    case DOALL_CILK:
      DoAllImpl<DOALL_CILK>::go (range, func, args...);
      break;
    case DOALL_OPENMP:
      // DoAllImpl<DOALL_OPENMP>::go (range, func, args...);
      std::abort ();
      break;
    default:
      abort ();
      break;
  }
}

template <typename R, typename F, typename... Args>
void do_all_choice (const R& range, const F& func, const Args&... args) { 
  do_all_choice (range, func, doAllKind, loopname, args...);
}

} // end namespace Galois

#endif //  GALOIS_DOALL_WRAPPER_H
