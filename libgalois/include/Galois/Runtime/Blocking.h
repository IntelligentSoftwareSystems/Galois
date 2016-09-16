/** Block Ranges -*- C++ -*-
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
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#ifndef GALOIS_RUNTIME_BLOCKING_H
#define GALOIS_RUNTIME_BLOCKING_H

#include <algorithm>
#include <iterator>
#include <utility>

#include <vector>
#include <set>
#include <deque>
#include <list>

namespace Galois {
namespace Runtime {

/**
 * Finds the midpoint of a range.  The first half is always be bigger than
 * the second half if the range has an odd length.
 */
template<typename IterTy>
IterTy split_range(IterTy b, IterTy e) {
  std::advance(b, (std::distance(b,e) + 1) / 2);
  return b;
}

/**
 * Returns a continuous block from the range based on the number of
 * divisions and the id of the block requested
 */
template<typename IterTy,
         typename std::enable_if<!std::is_integral<IterTy>::value>::type* = nullptr>
std::pair<IterTy, IterTy> block_range(IterTy b, IterTy e, unsigned id, unsigned num) {
  unsigned int dist = std::distance(b, e);
  unsigned int numper = std::max((dist + num - 1) / num, 1U); //round up
  unsigned int A = std::min(numper * id, dist);
  unsigned int B = std::min(numper * (id + 1), dist);
  std::advance(b, A);
  if (dist != B) {
    e = b;
    std::advance(e, B - A);
  }
  return std::make_pair(b,e);
}

template<typename IntTy,
         typename std::enable_if<std::is_integral<IntTy>::value>::type* = nullptr>
std::pair<IntTy, IntTy> block_range(IntTy b, IntTy e, unsigned id, unsigned num) {
  unsigned int dist = e - b;
  unsigned int numper = std::max((dist + num - 1) / num, 1U); //round up
  unsigned int A = std::min(numper * id, dist);
  unsigned int B = std::min(numper * (id + 1), dist);
  b += A;
  if (dist != B) {
    e = b;
    e += (B - A);
  }
  return std::make_pair(b,e);
}

} // namespace Runtime
} // namespace Galois
#endif
