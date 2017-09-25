/** Galois Simple Function Executor -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
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
 * @section Description
 *
 * Simple wrapper for the thread pool
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#ifndef GALOIS_RUNTIME_EXECUTOR_ONEACH_H
#define GALOIS_RUNTIME_EXECUTOR_ONEACH_H

#include "galois/gtuple.h"
#include "galois/Traits.h"
#include "galois/Timer.h"
#include "galois/runtime/Statistics.h"
#include "galois/Threads.h"
#include "galois/gIO.h"
#include "galois/substrate/ThreadPool.h"

#include <tuple>

namespace galois {
namespace runtime {

namespace internal {
template <typename FunctionTy, typename ArgsTy> 
inline void on_each_exec(FunctionTy& fn, const ArgsTy& argsTuple) {
  
  static constexpr bool MORE_STATS = exists_by_supertype<more_stats_tag, ArgsTy>::value;

  const char* const loopname = get_by_supertype<loopname_tag>(argsTuple).value;

  PerThreadTimer<MORE_STATS> execTime(loopname, "Execute");

  const auto numT = getActiveThreads();

  auto runFun = [&] {

    execTime.start();

    fn(substrate::ThreadPool::getTID(), numT);

    execTime.stop();

  };

  substrate::getThreadPool().run(numT, runFun);
}

template<typename FunctionTy, typename TupleTy>
inline void on_each_gen(FunctionTy& fn, const TupleTy& tpl) {
  static_assert(!exists_by_supertype<char*, TupleTy>::value, "old loopname");
  static_assert(!exists_by_supertype<char const *, TupleTy>::value, "old loopname");

  auto dtpl = std::tuple_cat(tpl,
      get_default_trait_values(tpl,
        std::make_tuple(loopname_tag{}),
        std::make_tuple(loopname{})));

  constexpr bool TIME_IT = exists_by_supertype<timeit_tag, decltype(dtpl)>::value;
  CondStatTimer<TIME_IT> timer(get_by_supertype<loopname_tag>(dtpl).value);

  timer.start();

  on_each_exec(fn, dtpl);

  timer.stop();
}

}

template<typename FunctionTy, typename TupleTy>
inline void on_each_impl(FunctionTy& fn, const TupleTy& tpl) {
  internal::on_each_gen<FunctionTy, TupleTy>(fn, tpl);
}

template<typename FunctionTy, typename TupleTy>
inline void on_each_impl(const FunctionTy& fn, const TupleTy& tpl) {
  internal::on_each_gen<const FunctionTy, TupleTy>(fn, tpl);
}


} // end namespace runtime
} // end namespace galois

#endif
