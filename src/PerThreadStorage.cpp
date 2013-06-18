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

//#define MORE_MEM_HACK
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

unsigned Galois::Runtime::PerBackend::allocOffset(unsigned size) {
  size = (size + 15) & ~15;
  unsigned retval = __sync_fetch_and_add(&nextLoc, size);
  if (retval + size > allocSize) {
    GALOIS_DIE("no more memory");
  }
  return retval;
}

void Galois::Runtime::PerBackend::deallocOffset(unsigned offset, unsigned size) {
  // Simplest way to recover memory; relies on mostly stack-like nature of
  // allocations
  size = (size + 15) & ~15;
  // Should only be executed by main thread but make lock-free for fun
  if (__sync_bool_compare_and_swap(&nextLoc, offset + size, offset)) {
    ; // Recovered some memory
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
