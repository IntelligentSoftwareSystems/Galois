/** Numa-aware large Allocators -*- C++ -*-
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
 * Copyright (C) 2017, The University of Texas at Austin. All rights
 * reserved.
 *
 * @section Description
 *
 * Numa interleaved large allocations
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 * @author Loc Hoang <l_hoang@utexas.edu> (largeMalloc Node and Edge)
 */
#ifndef GALOIS_SUBSTRATE_NUMAMEM
#define GALOIS_SUBSTRATE_NUMAMEM

#include <cstddef>
#include <memory>
#include <vector>

namespace Galois {
namespace Substrate {

namespace detail {
struct largeFreer {
  size_t bytes;
  void operator()(void* ptr) const;
};
}//namespace detail

typedef std::unique_ptr<void, detail::largeFreer> LAptr;

LAptr largeMallocLocal(size_t bytes); // fault in locally
LAptr largeMallocFloating(size_t bytes); // leave numa mapping undefined
LAptr largeMallocInterleaved(size_t bytes, unsigned numThreads); // fault in interleaved mapping
LAptr largeMallocBlocked(size_t bytes, unsigned numThreads); // fault in block interleaved mapping

// TODO clean this up into 1 function
LAptr largeMallocSpecifiedNode(size_t bytes, uint32_t numThreads, 
  const uint32_t* threadRanges, size_t elementSize);

LAptr largeMallocSpecifiedEdge(size_t bytes, 
          uint32_t numThreads, const uint32_t* threadRanges, 
          std::vector<uint64_t> edgePrefixSum, size_t elementSize);


} // namespace Substrate
} // namespace Galois

#endif //GALOIS_SUBSTRATE_NUMAMEM
