/** Per Thread Storage -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
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
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#include "Galois/Runtime/PerThreadStorage.h"
#include "Galois/Runtime/ll/gio.h"
#include "Galois/Runtime/mm/Mem.h"

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
const size_t allocSize = Galois::Runtime::MM::pageSize * 10;
inline void* alloc() {
  return malloc(allocSize);
}

#else
const size_t allocSize = Galois::Runtime::MM::pageSize;
inline void* alloc() {
  return Galois::Runtime::MM::pageAlloc();
}
#endif
#undef MORE_MEM_HACK

unsigned Galois::Runtime::PerBackend::nextLog2(unsigned size) {
  unsigned i = MIN_SIZE;
  while ((1<<i) < size) {
    ++i;
  }
  if (i >= MAX_SIZE) { 
    GALOIS_DIE ("PTS size too big");
  }
  return i;
}

unsigned Galois::Runtime::PerBackend::allocOffset(const unsigned sz) {
  unsigned retval = allocSize;

  unsigned size = (1 << nextLog2(sz));

  if ((nextLoc + size) <= allocSize) {
    // simple path, where we allocate bump ptr style
    retval = __sync_fetch_and_add (&nextLoc, size);
  } else {
    // find a free offset
    unsigned index = nextLog2(sz);
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
  unsigned size = (1 << nextLog2(sz));
  if (__sync_bool_compare_and_swap(&nextLoc, offset + size, offset)) {
    ; // allocation was at the end , so recovered some memory
  } else {
    // allocation not at the end
    freeOffsets[nextLog2(sz)].push_back(offset);
  }
}

void* Galois::Runtime::PerBackend::getRemote(unsigned thread, unsigned offset) {
  char* rbase = heads[thread];
  assert(rbase);
  return &rbase[offset];
}

void Galois::Runtime::PerBackend::initCommon() {
  if (heads.empty())
    heads.resize(LL::getMaxThreads());
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
    while (__sync_bool_compare_and_swap(&heads[leader], 0, 0)) { LL::asmPause(); }
    heads[id] = heads[leader];
    return heads[id];
  }
}

void Galois::Runtime::initPTS() {
  if (!Galois::Runtime::ptsBase) {
    //unguarded initialization as initPTS will run in the master thread
    //before any other threads are generated
    Galois::Runtime::ptsBase = getPTSBackend().initPerThread();
  }
  if (!Galois::Runtime::ppsBase) {
    Galois::Runtime::ppsBase = getPPSBackend().initPerPackage();
  }
}

#ifdef GALOIS_USE_EXP
char* Galois::Runtime::PerBackend::initPerThread_cilk () {
  assert (heads.size () == LL::getMaxThreads ());
  unsigned id = LL::getTID ();
  assert (heads[id] != nullptr);

  return heads[id];
}

char* Galois::Runtime::PerBackend::initPerPackage_cilk () {
  assert (heads.size () == LL::getMaxThreads ());
  unsigned id = LL::getTID ();
  assert (heads[id] != nullptr);

  return heads[id];
}

void Galois::Runtime::initPTS_cilk () {
  if (!Galois::Runtime::ptsBase) {
    Galois::Runtime::ptsBase = getPTSBackend ().initPerThread_cilk ();
  }
  if (!Galois::Runtime::ppsBase) {
    Galois::Runtime::ppsBase = getPPSBackend ().initPerPackage_cilk ();
  }
}
#endif // GALOIS_USE_EXP
