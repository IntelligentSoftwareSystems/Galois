/** Galois to Runtime translation -*- C++ -*-
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
 * Copyright (C) 2016, The University of Texas at Austin. All rights
 * reserved.
 *
 */

#ifndef GALOIS_RUNTIME_GALOISIMPL_H
#define GALOIS_RUNTIME_GALOISIMPL_H

#include "Galois/Runtime/Executor_DoAll.h"
#include "Galois/Runtime/Executor_OnEach.h"

#include <tuple>

namespace Galois {
namespace Runtime {

template<typename RangeTy, typename FunctionTy, typename TupleTy>
void do_all_gen(const RangeTy& r, const FunctionTy& fn, const TupleTy& tpl) {
  using Galois::loopname_tag;
  using Galois::loopname;

  static_assert(!exists_by_supertype<char*, TupleTy>::value, "old loopname");
  static_assert(!exists_by_supertype<char const *, TupleTy>::value, "old loopname");
  static_assert(!exists_by_supertype<bool, TupleTy>::value, "old steal");

  auto ntpl = std::tuple_cat(std::make_tuple(loopname{"(NULL)"}, do_all_steal<false>()), tpl);

  do_all_impl(
              r, fn, getActiveThreads(),
              get_by_supertype_last<loopname_tag>(ntpl).getValue(),
              get_by_supertype_last<do_all_steal_tag>(ntpl).getValue());
}

template<typename RangeTy, typename FunctionTy, typename... Args>
void for_each_gen(const RangeTy& r, const FunctionTy& fn, std::tuple<Args...> tpl);

template<typename FunctionTy, typename... Args>
void on_each_gen(const FunctionTy& fn, std::tuple<Args...> tpl) {
  auto ntpl = std::tuple_cat(tpl, std::make_tuple(loopname{"(NULL)"}));

  on_each_impl(getActiveThreads(), fn, get_by_supertype<loopname_tag>(ntpl).getValue());
}

} //namespace Runtime
} //namespace Galois

#endif
