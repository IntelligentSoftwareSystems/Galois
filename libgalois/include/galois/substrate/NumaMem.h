/** Numa-aware large Allocators -*- C++ -*-
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
 * Copyright (C) 2017, The University of Texas at Austin. All rights
 * reserved.
 *
 * @section Description
 *
 * Numa interleaved large allocations
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 * @author Loc Hoang <l_hoang@utexas.edu> (large malloc specified)
 */
#ifndef GALOIS_SUBSTRATE_NUMAMEM
#define GALOIS_SUBSTRATE_NUMAMEM

#include <cstddef>
#include <memory>
#include <vector>

namespace galois {
namespace substrate {

namespace internal {
struct largeFreer {
  size_t bytes;
  void operator()(void* ptr) const;
};
}//namespace internal

typedef std::unique_ptr<void, internal::largeFreer> LAptr;

LAptr largeMallocLocal(size_t bytes); // fault in locally
LAptr largeMallocFloating(size_t bytes); // leave numa mapping undefined
// fault in interleaved mapping
LAptr largeMallocInterleaved(size_t bytes, unsigned numThreads);
// fault in block interleaved mapping
LAptr largeMallocBlocked(size_t bytes, unsigned numThreads);

// fault in specified regions for each thread (threadRanges)
template<typename RangeArrayTy>
LAptr largeMallocSpecified(size_t bytes, uint32_t numThreads,
                           RangeArrayTy& threadRanges, size_t elementSize);


} // namespace substrate
} // namespace galois

#endif //GALOIS_SUBSTRATE_NUMAMEM
