/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
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

#include "galois/substrate/PerThreadStorage.h"

//#include "galois/runtime/Mem.h"
#include "galois/gIO.h"
#include <mutex>

thread_local char* galois::substrate::ptsBase;

galois::substrate::PerBackend& galois::substrate::getPTSBackend() {
  static galois::substrate::PerBackend b;
  return b;
}

thread_local char* galois::substrate::pssBase;

galois::substrate::PerBackend& galois::substrate::getPPSBackend() {
  static galois::substrate::PerBackend b;
  return b;
}

#define MORE_MEM_HACK
#ifdef MORE_MEM_HACK
const size_t allocSize =
    16 * (2 << 20); // galois::runtime::MM::hugePageSize * 16;
inline void* alloc() { return malloc(allocSize); }

#else
const size_t allocSize = galois::runtime::MM::hugePageSize;
inline void* alloc() { return galois::substrate::MM::pageAlloc(); }
#endif
#undef MORE_MEM_HACK

unsigned galois::substrate::PerBackend::nextLog2(unsigned size) {
  unsigned i = MIN_SIZE;
  while ((1U << i) < size) {
    ++i;
  }
  if (i >= MAX_SIZE) {
    abort();
  }
  return i;
}

unsigned galois::substrate::PerBackend::allocOffset(const unsigned sz) {
  unsigned retval = allocSize;
  unsigned ll     = nextLog2(sz);
  unsigned size   = (1 << ll);

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
        unsigned end   = retval + (1 << index);
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

void galois::substrate::PerBackend::deallocOffset(const unsigned offset,
                                                  const unsigned sz) {
  unsigned ll   = nextLog2(sz);
  unsigned size = (1 << ll);
  if (__sync_bool_compare_and_swap(&nextLoc, offset + size, offset)) {
    ; // allocation was at the end, so recovered some memory
  } else if (!invalid) {
    // allocation not at the end
    std::lock_guard<Lock> llock(freeOffsetsLock);
    freeOffsets[ll].push_back(offset);
  }
}

void* galois::substrate::PerBackend::getRemote(unsigned thread,
                                               unsigned offset) {
  char* rbase = heads[thread];
  assert(rbase);
  return &rbase[offset];
}

void galois::substrate::PerBackend::initCommon(unsigned maxT) {
  if (!heads) {
    assert(ThreadPool::getTID() == 0);
    heads = new char*[maxT];
    memset(heads, 0, sizeof(*heads) * maxT);
  }
}

char* galois::substrate::PerBackend::initPerThread(unsigned maxT) {
  initCommon(maxT);
  char* b = heads[ThreadPool::getTID()] = (char*)alloc();
  memset(b, 0, allocSize);
  return b;
}

char* galois::substrate::PerBackend::initPerSocket(unsigned maxT) {
  initCommon(maxT);
  unsigned id     = ThreadPool::getTID();
  unsigned leader = ThreadPool::getLeader();
  if (id == leader) {
    char* b = heads[id] = (char*)alloc();
    memset(b, 0, allocSize);
    return b;
  } else {
    // wait for leader to fix up socket
    while (__sync_bool_compare_and_swap(&heads[leader], 0, 0)) {
      substrate::asmPause();
    }
    heads[id] = heads[leader];
    return heads[id];
  }
}

void galois::substrate::initPTS(unsigned maxT) {
  if (!ptsBase) {
    // unguarded initialization as initPTS will run in the master thread
    // before any other threads are generated
    ptsBase = getPTSBackend().initPerThread(maxT);
  }
  if (!pssBase) {
    pssBase = getPPSBackend().initPerSocket(maxT);
  }
}
