/** Galois Per-Topo Storage -*- C++ -*-
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

#include "Galois/Runtime/PerHostStorage.h"


using namespace Galois::Runtime;

PerBackend_v2::PerBackend_v2() :nextID(0) {}

uint64_t PerBackend_v2::allocateOffset() {
  uint64_t off = __sync_add_and_fetch(&nextID, 1);
  off |= (uint64_t)networkHostID << 32;
  return off;
}

void PerBackend_v2::deallocateOffset(uint64_t) {}

void PerBackend_v2::createAt(uint64_t off, void* ptr) {
  lock.lock();
  assert(items.count(off) == 0);
  items[off] = ptr;
  lock.unlock();
}

void* PerBackend_v2::releaseAt_i(uint64_t off) {
  lock.lock();
  assert(items.count(off));
  void* retval = items[off];
  items.erase(off);
  lock.unlock();
  return retval;
}

void* PerBackend_v2::resolve_i(uint64_t off) {
  lock.lock();
  void* retval = items[off];
  lock.unlock();
  return retval;
}

void PerBackend_v2::pBe2ResolveLP(void* ptr, uint32_t srcID, uint64_t off) {
  getPerHostBackend().addRemote(ptr, srcID, off);
}

void PerBackend_v2::pBe2Resolve(uint32_t dest, uint64_t off) {
  void* ptr = getPerHostBackend().resolve_i(off);
  getSystemNetworkInterface().sendAlt(dest, pBe2ResolveLP, ptr, networkHostID, off);
}

void PerBackend_v2::addRemote(void* ptr, uint32_t srcID, uint64_t off) {
  lock.lock();
  remoteCache[std::make_pair(off, srcID)] = ptr;
  lock.unlock();
}

void* PerBackend_v2::resolveRemote_i(uint64_t off, uint32_t hostID) {
  if (hostID == Distributed::networkHostID) {
    return resolve_i(off);
  } else {
    //FIXME: remote message
     lock.lock();
     void* retval = remoteCache[std::make_pair(off, hostID)];
     lock.unlock();
     if (retval)
       return retval;
     getSystemNetworkInterface().sendAlt(hostID, pBe2Resolve, networkHostID, off);
     do {
       if (LL::getTID() == 0)
	 getSystemNetworkInterface().handleReceives();
       lock.lock();
       void* retval = remoteCache[std::make_pair(off, hostID)];
       lock.unlock();
       if (retval)
	 return retval;
     } while (true);
  }
}

Galois::Runtime::PerBackend_v2& Galois::Runtime::getPerHostBackend() {
  static PerBackend_v2 be;
  return be;
}
