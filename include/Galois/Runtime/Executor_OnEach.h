/** Galois Simple Function Executor -*- C++ -*-
 * @file
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
 *
 * @section Description
 *
 * Simple wrapper for the thread pool
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#ifndef GALOIS_RUNTIME_EXECUTOR_ONEACH_H
#define GALOIS_RUNTIME_EXECUTOR_ONEACH_H

#include "Galois/gtuple.h"
#include "Galois/Traits.h"
#include "Galois/Threads.h"
#include "Galois/Runtime/ll/TID.h"
#include "Galois/Runtime/ll/gio.h"
#include "Galois/Runtime/ThreadPool.h"

#include <tuple>

namespace Galois {
namespace Runtime {

template<typename FunctionTy>
struct OnEachExecutor {
  const FunctionTy& origFunction;
  explicit OnEachExecutor(const FunctionTy& f): origFunction(f) { }
  void operator()(void) {
    FunctionTy fn(origFunction);
    fn(LL::getTID(), activeThreads);   
  }
};

template<typename FunctionTy>
void on_each_impl(const FunctionTy& fn, const char* loopname = nullptr) {
  getSystemThreadPool().run(activeThreads, OnEachExecutor<FunctionTy>(fn));
}

template<typename FunctionTy, typename TupleTy>
void on_each_gen(const FunctionTy& fn, const TupleTy& tpl) {
  static_assert(!exists_by_supertype<char*, TupleTy>::value, "old loopname");
  static_assert(!exists_by_supertype<char const *, TupleTy>::value, "old loopname");
  auto dtpl = std::tuple_cat(tpl,
      get_default_trait_values(tpl,
        std::make_tuple(loopname_tag{}),
        std::make_tuple(loopname{})));

  on_each_impl(fn, get_by_supertype<loopname_tag>(dtpl).value);
}

void preAlloc_impl(int num);

} // end namespace Runtime
} // end namespace Galois

#endif
