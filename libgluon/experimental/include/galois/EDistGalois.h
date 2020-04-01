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

// TODO Remove duplicate code and only keep distributed stuff

#ifndef GALOIS_DIST_GALOIS_H
#define GALOIS_DIST_GALOIS_H

#include "galois/runtime/SharedMem.h"
#include "galois/runtime/DistStats.h"

#include <string>
#include <utility>
#include <tuple>

/**
 * Main Galois namespace. All the core Galois functionality will be found in
 * here.
 */
namespace galois {

/**
 * explicit class to initialize the Galois Runtime
 * Runtime is destroyed when this object is destroyed
 */
class DistMemSys : public runtime::SharedMem<runtime::DistStatManager> {

public:
  explicit DistMemSys();

  ~DistMemSys();

  DistMemSys(const DistMemSys&) = delete;
  DistMemSys& operator=(const DistMemSys&) = delete;

  DistMemSys(DistMemSys&&) = delete;
  DistMemSys& operator=(DistMemSys&&) = delete;
};

////////////////////////////////////////////////////////////////////////////////
// Foreach
////////////////////////////////////////////////////////////////////////////////

/**
 * Galois unordered set iterator.
 * Operator should conform to <code>fn(item, UserContext<T>&)</code> where item
 * is a value from the iteration range and T is the type of item.
 *
 * @tparam WLTy Worklist policy {@see galois::worklists}
 * @param b begining of range of initial items
 * @param e end of range of initial items
 * @param fn operator
 * @param args optional arguments to loop, e.g., {@see loopname}, {@see wl}
 */
// TODO: revive
// template<typename IterTy, typename FunctionTy, typename... Args>
// void for_each(const IterTy& b, const IterTy& e, const FunctionTy& fn, const
// Args&... args) { runtime::for_each_gen_dist(runtime::makeStandardRange(b,e),
// fn, std::make_tuple(args...));
// }

/**
 * Galois unordered set iterator.
 * Operator should conform to <code>fn(item, UserContext<T>&)</code> where item
 * is i and T is the type of item.
 *
 * @tparam WLTy Worklist policy {@link galois::worklists}
 * @param i initial item
 * @param fn operator
 * @param args optional arguments to loop
 */
// TODO: revive
// template<typename ItemTy, typename FunctionTy, typename... Args>
// void for_each(const ItemTy& i, const FunctionTy& fn, const Args&... args) {
// ItemTy iwl[1] = {i};
// runtime::for_each_gen_dist(runtime::makeStandardRange(&iwl[0], &iwl[1]), fn,
// std::make_tuple(args...));
// }

} // namespace galois
#endif
