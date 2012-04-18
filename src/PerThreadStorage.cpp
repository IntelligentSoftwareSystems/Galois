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

#include <vector>

__thread void** GaloisRuntime::HIDDEN::base;
static std::vector<void**> heads;
static unsigned int nextLoc; // intially 0

unsigned GaloisRuntime::HIDDEN::allocOffset() {
  unsigned retval = __sync_fetch_and_add(&nextLoc, 1);
  return retval;
}

void* GaloisRuntime::HIDDEN::getRemote(unsigned thread, unsigned offset) {
  void** rbase = heads[thread];
  if (rbase)
    return rbase[offset];
  return 0;
}

void GaloisRuntime::initPTS() {
  if (!GaloisRuntime::HIDDEN::base) {
    //unguarded initialization as initPTS will run in the master thread
    //before any other threads are generated
    if (heads.empty())
      heads.resize(LL::getMaxThreads());
    GaloisRuntime::HIDDEN::base = heads[LL::getTID()] = (void**)GaloisRuntime::MM::pageAlloc();
    memset(GaloisRuntime::HIDDEN::base, 0, GaloisRuntime::MM::pageSize);
  }
}

