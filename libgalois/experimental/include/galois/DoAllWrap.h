/** DoAll wrapper -*- C++ -*-
 * @File
 * This is the only file to include for basic Galois functionality.
 *
 * @section License
 *
 * This file is part of Galois.  Galois is a framework to exploit
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
 * Copyright (C) 2017, The University of Texas at Austin. All rights
 * reserved.
 *
 */

#ifndef GALOIS_DOALL_WRAPPER_H
#define GALOIS_DOALL_WRAPPER_H

#include "galois/Galois.h"
#include "galois/GaloisForwardDecl.h"
#include "galois/OrderedTraits.h"
#include "galois/runtime/Executor_DoAll_Old.h"
#include "galois/runtime/Executor_DoAll.h"
#include "galois/substrate/EnvCheck.h"

#ifdef GALOIS_USE_TBB
#include "tbb/parallel_for_each.h"
#endif

#include "CilkInit.h"
#include <unistd.h>

#include "llvm/Support/CommandLine.h"

namespace galois {

enum DoAllTypes {
  DO_ALL_OLD, DO_ALL_OLD_STEAL, DOALL_GALOIS_FOREACH, DO_ALL,
  DO_ALL_RANGE,
  DOALL_CILK, DOALL_OPENMP, DOALL_RANGE
};

namespace cll = llvm::cl;
//extern cll::opt<DoAllTypes> doAllKind;
static cll::opt<DoAllTypes> doAllKind (
        "doAllKind",
        cll::desc ("DoAll Implementation"),
        cll::values (
          clEnumVal (DO_ALL_OLD, "DO_ALL_OLD"),
          clEnumVal (DO_ALL_OLD_STEAL, "DO_ALL_OLD_STEAL"),
          clEnumVal (DOALL_GALOIS_FOREACH, "DOALL_GALOIS_FOREACH"),
          clEnumVal (DO_ALL, "DO_ALL"),
          clEnumVal (DO_ALL_RANGE, "DO_ALL_RANGE"),
          clEnumVal (DOALL_CILK, "DOALL_CILK"),
          clEnumVal (DOALL_OPENMP, "DOALL_OPENMP"),
          clEnumVal (DOALL_RANGE, "DOALL_RANGE"),
          clEnumValEnd),
        cll::init (DO_ALL_OLD)); // default is regular DOALL


void setDoAllImpl (const DoAllTypes& type);

DoAllTypes getDoAllImpl (void);

template <DoAllTypes TYPE>
struct DoAllImpl {
  template <typename R, typename F, typename ArgsTuple>
  static inline void go(const R& range, const F& func,
                        const ArgsTuple& argsTuple) {
    std::abort();
  }
};

template <>
struct DoAllImpl<DO_ALL_OLD> {
  template <typename R, typename F, typename ArgsTuple>
  static inline void go (const R& range, const F& func,
                         const ArgsTuple& argsTuple) {
    galois::runtime::do_all_gen_old(range, func,
        std::tuple_cat(std::make_tuple(steal<false> ()), argsTuple));
  }
};

template <>
struct DoAllImpl<DO_ALL_OLD_STEAL> {
  template <typename R, typename F, typename ArgsTuple>
  static inline void go(const R& range, const F& func,
                        const ArgsTuple& argsTuple) {
    galois::runtime::do_all_gen_old (range, func,
        std::tuple_cat(std::make_tuple(steal<true>()), argsTuple));
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

  template <typename R, typename F, typename ArgsTuple>
  static inline void go (const R& range, const F& func, const ArgsTuple& argsTuple) {

    using T = typename R::value_type;

    const unsigned CHUNK_SIZE = 128;
    //const unsigned CHUNK_SIZE = get_type_by_supertype<chunk_size_tag, ArgsTuple>::type::value;

    using WL_ty =  galois::worklists::AltChunkedLIFO<CHUNK_SIZE, T>;

    galois::runtime::for_each_gen(range, FuncWrap<T, F> {func},
        std::tuple_cat(
          std::make_tuple(galois::wl<WL_ty>(),
             no_pushes(),
             no_conflicts()),
          argsTuple));
  }
};

template <>
struct DoAllImpl<DO_ALL> {
  template <typename R, typename F, typename ArgsTuple>
  static inline void go (const R& range, const F& func, const ArgsTuple& argsTuple) {
    galois::runtime::do_all_gen(range, func, argsTuple);
  }
};

/* Better do all coupled that takes into account even distribution of edges
 * among threads
 */
template <>
struct DoAllImpl<DO_ALL_RANGE> {
  template <typename R, typename F, typename ArgsTuple>
  static inline void go (const R& range, const F& func, const ArgsTuple& argsTuple) {
    auto defaultArgsTuple = std::tuple_cat(
      argsTuple,
      get_default_trait_values(
        argsTuple,
        std::make_tuple(thread_range_tag{}),
        std::make_tuple(thread_range{})
      )
    );

    const uint32_t *thread_ranges =
      get_by_supertype<thread_range_tag>(defaultArgsTuple).getValue();

    assert(thread_ranges != nullptr);

    galois::runtime::do_all_gen(runtime::makeSpecificRange(range.begin(),
                                    range.end(), thread_ranges),
                                    func, argsTuple);
  }
};


#ifdef HAVE_CILK
template <>
struct DoAllImpl<DOALL_CILK> {
  template <typename R, typename F, typename ArgsTuple>
  static inline void go (const R& range, const F& func, const ArgsTuple& argsTuple) {
    CilkInit ();
    cilk_for(auto it = range.begin (), end = range.end (); it != end; ++it) {
      func (*it);
    }
  }
};
#else
template <> struct DoAllImpl<DOALL_CILK> {
  template <typename R, typename F, typename ArgsTuple>
  static inline void go (const R& range, const F& func, const ArgsTuple& argsTuple) {
    GALOIS_DIE("Cilk not found\n");
  }
};
#endif

template <>
struct DoAllImpl<DOALL_OPENMP> {
  template <typename R, typename F, typename ArgsTuple>
  static inline void go (const R& range, const F& func, const ArgsTuple& argsTuple) {
  const auto end = range.end ();
#pragma omp parallel for schedule(guided)
    for (auto it = range.begin (); it < end; ++it) {
      func (*it);
    }
  }
};

/**
 * Not a standard do-all loop as the work distribution among threads is
 * specified by an iterator array. All iterations should be independent.
 * Operator should conform to <code>fn(item)</code> where item is i
 *
 * @param range Begin/end of range of do-all
 * @param func operator
 * @param argsTuple optional arguments to loop;
 * TODO currently should have a non-optional iterator array that tells you
 * where each thread starts/stops work; make this optional in the future
 */
template <>
struct DoAllImpl<DOALL_RANGE> {
  // TODO make it so the work splitting happens within this function(?) instead
  // of at graph initialization (then it would work with all ranges
  // and not just an all vertices range as is being assumed currently)
  template <typename R, typename F, typename ArgsTuple>
  static inline void go(const R& range, const F& func,
                        const ArgsTuple& argsTuple) {

    auto defaultArgsTuple = std::tuple_cat(
      argsTuple,
      get_default_trait_values(
        argsTuple,
        std::make_tuple(thread_range_tag{}),
        std::make_tuple(thread_range{})
      )
    );

    const uint32_t *thread_ranges =
      get_by_supertype<thread_range_tag>(defaultArgsTuple).getValue();
    assert(thread_ranges != nullptr);

    galois::runtime::do_all_gen(
      runtime::makeSpecificRange(range.begin(), range.end(), thread_ranges),
      func,
      std::tuple_cat(std::make_tuple(steal<false>()), argsTuple)
    );
  }
};


template <typename R, typename F, typename ArgsTuple>
void do_all_choice(const R& range, const F& func, const DoAllTypes& type,
                   const ArgsTuple& argsTuple) {
  switch (type) {
    case DO_ALL_OLD_STEAL:
      DoAllImpl<DO_ALL_OLD_STEAL>::go(range, func, argsTuple);
      break;
    case DOALL_GALOIS_FOREACH:
      DoAllImpl<DOALL_GALOIS_FOREACH>::go(range, func, argsTuple);
      break;
    case DO_ALL_OLD:
      DoAllImpl<DO_ALL_OLD>::go(range, func, argsTuple);
      break;
    case DO_ALL:
      DoAllImpl<DO_ALL>::go(range, func, argsTuple);
      break;
    case DO_ALL_RANGE:
      DoAllImpl<DO_ALL_RANGE>::go(range, func, argsTuple);
      break;
    case DOALL_CILK:
      DoAllImpl<DOALL_CILK>::go(range, func, argsTuple);
      break;
    case DOALL_OPENMP:
      // DoAllImpl<DOALL_OPENMP>::go(range, func, argsTuple);
      std::abort ();
      break;
    case DOALL_RANGE:
      DoAllImpl<DOALL_RANGE>::go(range, func, argsTuple);
      break;
    default:
      abort ();
      break;
  }
}

template <typename R, typename F, typename ArgsTuple>
void do_all_choice (const R& range, const F& func, const ArgsTuple& argsTuple) {
  do_all_choice (range, func, doAllKind, argsTuple);
}

} // end namespace galois

#endif //  GALOIS_DOALL_WRAPPER_H
