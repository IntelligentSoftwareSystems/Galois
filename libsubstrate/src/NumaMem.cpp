/** Large Allocatoins -*- C++ -*-
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
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#include "Galois/Substrate/NumaMem.h"
#include "Galois/Substrate/PageAlloc.h"
#include "Galois/Substrate/ThreadPool.h"
#include "Galois/Substrate/gio.h"

#include <cassert>

using namespace Galois::Substrate;

/* Access pages on each thread so each thread has local pages 
 * already loaded */
static void pageIn(void* _ptr, size_t len, size_t pageSize, 
                   unsigned numThreads, bool finegrained) {
  char* ptr = static_cast<char*>(_ptr);

  if (numThreads == 1) {
    for (size_t x = 0; x < len; x += pageSize / 2)
      ptr[x] = 0;
  } else {
    ThreadPool::getThreadPool().run(numThreads, 
      [ptr, len, pageSize, numThreads, finegrained] () 
      {
        auto myID = ThreadPool::getTID();
        if (finegrained) {
          for (size_t x  = pageSize * myID; x < len; x += pageSize * numThreads)
          ptr[x] = 0;
        } else {
          for (size_t x = myID * len / numThreads; 
               x < len && x < (myID + 1) * len / numThreads; 
               x += pageSize)
            ptr[x] = 0;
        }
      }
    );
  }
}

static void largeFree(void* ptr, size_t bytes) {
  freePages(ptr, bytes/allocSize());
}

void Galois::Substrate::detail::largeFreer::operator()(void* ptr) const {
  largeFree(ptr, bytes);
}

//round data to a multiple of mult
static size_t roundup (size_t data, size_t mult) {
  auto rem = data % mult;
  if (!rem)
    return data;
  return data + (mult - rem);
}

LAptr Galois::Substrate::largeMallocInterleaved(size_t bytes, unsigned numThreads) {
  // round up to hugePageSize
  bytes = roundup(bytes, allocSize());

#ifdef GALOIS_USE_NUMA
  // We don't use numa_alloc_interleaved_subset because we really want huge 
  // pages
  // yes this is a comment in a ifdef, but if libnuma improves, this is where 
  // the alloc would go
#endif
  // Get a non-prefaulted allocation
  void* data = allocPages(bytes/allocSize(), false);
  if (data)
    pageIn(data, bytes, allocSize(), numThreads, true);
  return LAptr{data, detail::largeFreer{bytes}};
}

LAptr Galois::Substrate::largeMallocLocal(size_t bytes) {
  // round up to hugePageSize
  bytes = roundup(bytes, allocSize());
  // Get a prefaulted allocation
  return LAptr{allocPages(bytes/allocSize(), true), detail::largeFreer{bytes}};
}

LAptr Galois::Substrate::largeMallocFloating(size_t bytes) {
  //round up to hugePageSize
  bytes = roundup(bytes, allocSize());
  //Get a non-prefaulted allocation
  return LAptr{allocPages(bytes/allocSize(), false), detail::largeFreer{bytes}};
}

LAptr Galois::Substrate::largeMallocBlocked(size_t bytes, unsigned numThreads) {
  // round up to hugePageSize
  bytes = roundup(bytes, allocSize());
  // Get a non-prefaulted allocation
  void* data = allocPages(bytes/allocSize(), false);
  if (data)
    pageIn(data, bytes, allocSize(), numThreads, false);
  return LAptr{data, detail::largeFreer{bytes}};
}

