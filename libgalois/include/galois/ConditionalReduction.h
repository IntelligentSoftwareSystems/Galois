/** Conditional Accumulator type -*- C++ -*-
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
 * License along with Galois. If not, see <http://www.gnu.org/licenses/>.
 *
 * @section Copyright
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights
 * reserved.
 *
 * @author Loc Hoang <l_hoang@utexas.edu>
 */


#ifndef __COND_REDUCTION__
#define __COND_REDUCTION__

#include "galois/Reduction.h"

/**
 *
 *
 */
template<typename Accumulator, bool Active>
class ConditionalAccumulator {
  typename std::conditional<Active, Accumulator, char>::type accumulator;
 public:
  using T = typename Accumulator::AccumType;

  bool isActive() {
    return Active;
  }

  template<bool A = Active, typename std::enable_if<A>::type* = nullptr>
  void reset() {
    accumulator.reset();
  }

  template<bool A = Active, typename std::enable_if<!A>::type* = nullptr>
  void reset() {
    // no-op
  }

  template<bool A = Active, typename std::enable_if<A>::type* = nullptr>
  void update(T newValue) {
    accumulator.update(newValue);
  }

  template<bool A = Active, typename std::enable_if<!A>::type* = nullptr>
  void update(T newValue) {
    // no-op
  }

  template<bool A = Active, typename std::enable_if<A>::type* = nullptr>
  T reduce() {
    return accumulator.reduce();
  }

  template<bool A = Active, typename std::enable_if<!A>::type* = nullptr>
  T reduce() {
    return 0; // TODO choose value that works better regardless of T
  }

  // TODO add the rest of the GSimpleReducible functions?
};

#endif
