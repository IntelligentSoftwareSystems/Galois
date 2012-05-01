/** Per Thread Storage -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2012, The University of Texas at Austin. All rights reserved.
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
#include "Galois/Runtime/mm/Mem.h"

__thread char* GaloisRuntime::ptsBase;
GaloisRuntime::PerBackend GaloisRuntime::PTSBackend;
__thread char* GaloisRuntime::ppsBase;
GaloisRuntime::PerBackend GaloisRuntime::PPSBackend;


unsigned GaloisRuntime::PerBackend::allocOffset(unsigned size) {
  size = (size + 15) & ~15;
  unsigned retval = __sync_fetch_and_add(&nextLoc, size);
  assert(retval + size < GaloisRuntime::MM::pageSize);
  return retval;
}

void* GaloisRuntime::PerBackend::getRemote(unsigned thread, unsigned offset) {
  char* rbase = heads[thread];
  assert(rbase);
  return &rbase[offset];
}

void GaloisRuntime::PerBackend::initCommon() {
  if (heads.empty())
    heads.resize(LL::getMaxThreads());
}

char* GaloisRuntime::PerBackend::initPerThread() {
  initCommon();
  char* b = heads[LL::getTID()] = (char*)GaloisRuntime::MM::pageAlloc();
  memset(b, 0, GaloisRuntime::MM::pageSize);
  return b;
}

char* GaloisRuntime::PerBackend::initPerPackage() {
  initCommon();
  unsigned id = LL::getTID();
  unsigned leader = LL::getLeaderForThread(id);
  if (id == leader) {
    char* b = heads[id] = (char*)GaloisRuntime::MM::pageAlloc();
    memset(b, 0, GaloisRuntime::MM::pageSize);
    return b;
  } else {
    //wait for leader to fix up package
    while (__sync_bool_compare_and_swap(&heads[leader], 0, 0)) { LL::asmPause(); }
    heads[id] = heads[leader];
    return heads[id];
  }
}

void GaloisRuntime::initPTS() {
  if (!GaloisRuntime::ptsBase) {
    //unguarded initialization as initPTS will run in the master thread
    //before any other threads are generated
    GaloisRuntime::ptsBase = PTSBackend.initPerThread();
  }
  if (!GaloisRuntime::ppsBase) {
    GaloisRuntime::ppsBase = PPSBackend.initPerPackage();
  }
}
