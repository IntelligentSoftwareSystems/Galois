/** Per Thread Storage -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a gramework to exploit
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

#include "Galois/Substrate/PerThreadStorage.h"

#include "Galois/Runtime/mm/Mem.h"
#include "Galois/Runtime/ll/gio.h"
#include <mutex>

__thread char* Galois::Runtime::ptsBase;

Galois::Runtime::PerBackend& Galois::Runtime::getPTSBackend() {
  static Galois::Runtime::PerBackend b;
  return b;
}

__thread char* Galois::Runtime::ppsBase;

Galois::Runtime::PerBackend& Galois::Runtime::getPPSBackend() {
  static Galois::Runtime::PerBackend b;
  return b;
}

#define MORE_MEM_HACK
#ifdef MORE_MEM_HACK
const size_t allocSize = Galois::Runtime::MM::hugePageSize * 16;
inline void* alloc() {
  return malloc(allocSize);
}

#else
const size_t allocSize = Galois::Runtime::MM::hugePageSize;
inline void* alloc() {
  return Galois::Runtime::MM::pageAlloc();
}
#endif
#undef MORE_MEM_HACK

unsigned Galois::Runtime::PerBackend::nextLog2(unsigned size) {
  unsigned i = MIN_SIZE;
  while ((1U<<i) < size) {
    ++i;
  }
  if (i >= MAX_SIZE) { 
    abort();
  }
  return i;
}

unsigned Galois::Runtime::PerBackend::allocOffset(const unsigned sz) {
  unsigned retval = allocSize;
  unsigned ll = nextLog2(sz);
  unsigned size = (1 << ll);

  if ((nextLoc + size) <= allocSize) {
    // simple path, where we allocate bump ptr style
    retval = __sync_fetch_and_add(&nextLoc, size);
  } else if (!invalid) {
    // find a free offset
    std::lock_guard<Lock> llock(freeOffsetsLock);

    unsigned index = ll;
    if (!freeOffsets[index].empty()) {
      retval = freeOffsets[index].back();
      freeOffsets[index].pop_back();
    } else {
      // find a bigger size 
      for (; (index < MAX_SIZE) && (freeOffsets[index].empty()); ++index)
        ;

      if (index == MAX_SIZE) {
        GALOIS_DIE("PTS out of memory error");
      } else {
        // Found a bigger free offset. Use the first piece equal to required
        // size and produce vending machine change for the rest.
        assert(!freeOffsets[index].empty());
        retval = freeOffsets[index].back();
        freeOffsets[index].pop_back();

        // remaining chunk
        unsigned end = retval + (1 << index);
        unsigned start = retval + size; 
        for (unsigned i = index - 1; start < end; --i) {
          freeOffsets[i].push_back(start);
          start += (1 << i);
        }
      }
    }
  }

  assert(retval != allocSize);

  return retval;
}

void Galois::Runtime::PerBackend::deallocOffset(const unsigned offset, const unsigned sz) {
  unsigned ll = nextLog2(sz);
  unsigned size = (1 << ll);
  if (__sync_bool_compare_and_swap(&nextLoc, offset + size, offset)) {
    ; // allocation was at the end, so recovered some memory
  } else if (!invalid) {
    // allocation not at the end
    std::lock_guard<Lock> llock(freeOffsetsLock);
    freeOffsets[ll].push_back(offset);
  }
}

void* Galois::Runtime::PerBackend::getRemote(unsigned thread, unsigned offset) {
  char* rbase = heads[thread];
  assert(rbase);
  return &rbase[offset];
}

void Galois::Runtime::PerBackend::initCommon() {
  if (!heads) {
    assert(LL::getTID() == 0);
    unsigned n = LL::getMaxThreads();
    heads = new char*[n];
    memset(heads, 0, sizeof(*heads)* n);
  }
}

char* Galois::Runtime::PerBackend::initPerThread() {
  initCommon();
  char* b = heads[LL::getTID()] = (char*) alloc();
  memset(b, 0, allocSize);
  return b;
}

char* Galois::Runtime::PerBackend::initPerPackage() {
  initCommon();
  unsigned id = LL::getTID();
  unsigned leader = LL::getLeaderForThread(id);
  if (id == leader) {
    char* b = heads[id] = (char*) alloc();
    memset(b, 0, allocSize);
    return b;
  } else {
    //wait for leader to fix up package
    while (__sync_bool_compare_and_swap(&heads[leader], 0, 0)) { Substrate::asmPause(); }
    heads[id] = heads[leader];
    return heads[id];
  }
}

void Galois::Runtime::initPTS() {
  if (!ptsBase) {
    //unguarded initialization as initPTS will run in the master thread
    //before any other threads are generated
    ptsBase = getPTSBackend().initPerThread();
  }
  if (!ppsBase) {
    ppsBase = getPPSBackend().initPerPackage();
  }
}

#ifdef GALOIS_USE_EXP
// assumes that per thread storage has been initialized by Galois already
// and just copies over the same pointers to cilk threads
char* Galois::Runtime::PerBackend::initPerThread_cilk() {
  unsigned id = LL::getTID();
  assert(heads[id] != nullptr);

  return heads[id];
}

char* Galois::Runtime::PerBackend::initPerPackage_cilk() {
  unsigned id = LL::getTID();
  assert(heads[id] != nullptr);

  return heads[id];
}

void Galois::Runtime::initPTS_cilk() {
  if (!ptsBase) {
    ptsBase = getPTSBackend().initPerThread_cilk();
  }
  if (!ppsBase) {
    ppsBase = getPPSBackend().initPerPackage_cilk();
  }
  assert (ptsBase != nullptr);
  assert (ppsBase != nullptr);
}
#endif // GALOIS_USE_EXP
