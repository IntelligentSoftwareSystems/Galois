/** Galois Simple Function Executor -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galois is a framework to exploit
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
inline void on_each_impl(FunctionTy& fn, const ArgsTy& argsTuple) {

  static_assert(!exists_by_supertype<char*, ArgsTy>::value, "old loopname");
  static_assert(!exists_by_supertype<char const *, ArgsTy>::value, "old loopname");

  static constexpr bool NEEDS_STATS = exists_by_supertype<loopname_tag, ArgsTy>::value;
  static constexpr bool MORE_STATS = NEEDS_STATS && exists_by_supertype<more_stats_tag, ArgsTy>::value;

  const char* const loopname = galois::internal::getLoopName(argsTuple);

  CondStatTimer<NEEDS_STATS> timer(loopname);

  PerThreadTimer<MORE_STATS> execTime(loopname, "Execute");

  const auto numT = getActiveThreads();

  auto runFun = [&] {

    execTime.start();

    fn(substrate::ThreadPool::getTID(), numT);

    execTime.stop();

  };

  timer.start();
  substrate::getThreadPool().run(numT, runFun);
  timer.stop();
}

}

template<typename FunctionTy, typename TupleTy>
inline void on_each_gen(FunctionTy& fn, const TupleTy& tpl) {
  internal::on_each_impl<FunctionTy, TupleTy>(fn, tpl);
}

template<typename FunctionTy, typename TupleTy>
inline void on_each_gen(const FunctionTy& fn, const TupleTy& tpl) {
  internal::on_each_impl<const FunctionTy, TupleTy>(fn, tpl);
}


} // end namespace runtime
} // end namespace galois

#endif
