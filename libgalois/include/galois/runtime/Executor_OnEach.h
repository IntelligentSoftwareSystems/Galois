/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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

#ifndef GALOIS_RUNTIME_EXECUTOR_ONEACH_H
#define GALOIS_RUNTIME_EXECUTOR_ONEACH_H

#include "galois/config.h"
#include "galois/gIO.h"
#include "galois/gtuple.h"
#include "galois/runtime/OperatorReferenceTypes.h"
#include "galois/runtime/Statistics.h"
#include "galois/runtime/ThreadTimer.h"
#include "galois/substrate/ThreadPool.h"
#include "galois/Threads.h"
#include "galois/Timer.h"
#include "galois/Traits.h"

namespace galois {
namespace runtime {

namespace internal {

template <typename FunctionTy, typename ArgsTy>
inline void on_each_impl(FunctionTy&& fn, const ArgsTy& argsTuple) {

  static_assert(!exists_by_supertype<char*, ArgsTy>::value, "old loopname");
  static_assert(!exists_by_supertype<char const*, ArgsTy>::value,
                "old loopname");

  static constexpr bool NEEDS_STATS =
      exists_by_supertype<loopname_tag, ArgsTy>::value;
  static constexpr bool MORE_STATS =
      NEEDS_STATS && exists_by_supertype<more_stats_tag, ArgsTy>::value;

  const char* const loopname = galois::internal::getLoopName(argsTuple);

  CondStatTimer<NEEDS_STATS> timer(loopname);

  PerThreadTimer<MORE_STATS> execTime(loopname, "Execute");

  const auto numT = getActiveThreads();

  OperatorReferenceType<decltype(std::forward<FunctionTy>(fn))> fn_ref = fn;

  auto runFun = [&] {
    execTime.start();

    fn_ref(substrate::ThreadPool::getTID(), numT);

    execTime.stop();
  };

  timer.start();
  substrate::getThreadPool().run(numT, runFun);
  timer.stop();
}

} // namespace internal

template <typename FunctionTy, typename TupleTy>
inline void on_each_gen(FunctionTy&& fn, const TupleTy& tpl) {
  internal::on_each_impl(std::forward<FunctionTy>(fn), tpl);
}

} // end namespace runtime
} // end namespace galois

#endif
