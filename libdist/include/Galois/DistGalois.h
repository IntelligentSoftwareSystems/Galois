/** Galois user interface -*- C++ -*-
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
 * Copyright (C) 2015, The University of Texas at Austin. All rights
 * reserved.
 *
 */

// TODO (amber): copied over from libruntime/include/Galois/Galois.h
// Remove duplicate code and only keep distributed stuff

#ifndef GALOIS_DIST_GALOIS_H
#define GALOIS_DIST_GALOIS_H

#include "Galois/Runtime/DistStatCollector.h"
#include "Galois/Runtime/Init.h"
#include "Galois/Substrate/Init.h"

#include <string>
#include <utility>
#include <tuple>

/**
 * Main Galois namespace. All the core Galois functionality will be found in here.
 */
namespace Galois {

/**
 * explicit class to initialize the Galois Runtime
 * Runtime is destroyed when this object is destroyed
 */
class DistMemSys {
  Runtime::DistStatCollector m_sc;
  bool statsPrinted;

public:
  explicit DistMemSys(const std::string& outfile="", bool statsPrinted = false): m_sc(outfile), statsPrinted(statsPrinted) {
    Substrate::init();
    Runtime::init(&m_sc);
  }

  ~DistMemSys(void) {
    if(!statsPrinted)
      m_sc.printStats();
    Runtime::kill();
    Substrate::kill();
  }

  void printDistStats() {
    m_sc.printStats();
   statsPrinted = true; 
  }
};

////////////////////////////////////////////////////////////////////////////////
// Foreach
////////////////////////////////////////////////////////////////////////////////

/**
 * Galois unordered set iterator.
 * Operator should conform to <code>fn(item, UserContext<T>&)</code> where item is a value from the iteration
 * range and T is the type of item.
 *
 * @tparam WLTy Worklist policy {@see Galois::WorkList}
 * @param b begining of range of initial items
 * @param e end of range of initial items
 * @param fn operator
 * @param args optional arguments to loop, e.g., {@see loopname}, {@see wl}
 */
// TODO: revive
// template<typename IterTy, typename FunctionTy, typename... Args>
// void for_each(const IterTy& b, const IterTy& e, const FunctionTy& fn, const Args&... args) {
  // Runtime::for_each_gen_dist(Runtime::makeStandardRange(b,e), fn, std::make_tuple(args...));
// }

/**
 * Galois unordered set iterator.
 * Operator should conform to <code>fn(item, UserContext<T>&)</code> where item is i and T 
 * is the type of item.
 *
 * @tparam WLTy Worklist policy {@link Galois::WorkList}
 * @param i initial item
 * @param fn operator
 * @param args optional arguments to loop
 */
// TODO: revive
// template<typename ItemTy, typename FunctionTy, typename... Args>
// void for_each(const ItemTy& i, const FunctionTy& fn, const Args&... args) {
  // ItemTy iwl[1] = {i};
  // Runtime::for_each_gen_dist(Runtime::makeStandardRange(&iwl[0], &iwl[1]), fn, std::make_tuple(args...));
// }


} //namespace Galois
#endif
