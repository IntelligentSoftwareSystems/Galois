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

#include "Galois/gtuple.h"
#include "Galois/Traits.h"
#include "Galois/Threads.h"
#include "Galois/gIO.h"
#include "Galois/Substrate/ThreadPool.h"

#include <tuple>

namespace Galois {
namespace Runtime {

template<typename FunctionTy>
struct OnEachExecutor {
  const FunctionTy& origFunction;
  unsigned int activeThreads;
  explicit OnEachExecutor(const FunctionTy& f, unsigned int actT): origFunction(f), activeThreads(actT) { }
  void operator()(void) {
    FunctionTy fn(origFunction);
    fn(Substrate::ThreadPool::getTID(), activeThreads);   
  }
};

template<typename FunctionTy>
void on_each_impl(const FunctionTy& fn, const char* loopname = nullptr) {
  auto activeThreads = getActiveThreads();
  Substrate::getThreadPool().run(activeThreads, OnEachExecutor<FunctionTy>(fn, activeThreads));
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

} // end namespace Runtime
} // end namespace Galois

#endif
