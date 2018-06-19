/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
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

#include "galois/runtime/PerHostStorage.h"

#include <mutex>
#include <algorithm>

using namespace galois::runtime;

PerBackend_v2::PerBackend_v2() : nextID(0) {}

uint64_t PerBackend_v2::allocateOffset() {
  uint64_t off = __sync_add_and_fetch(&nextID, 1);
  off |= (uint64_t)NetworkInterface::ID << 32;
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
  getSystemNetworkInterface().sendAlt(dest, pBe2ResolveLP, ptr,
                                      NetworkInterface::ID, off);
}

void PerBackend_v2::addRemote(void* ptr, uint32_t srcID, uint64_t off) {
  lock.lock();
  remoteCache[std::make_pair(off, srcID)] = ptr;
  lock.unlock();
}

void* PerBackend_v2::resolveRemote_i(uint64_t off, uint32_t hostID) {
  if (hostID == NetworkInterface::ID) {
    return resolve_i(off);
  } else {
    // FIXME: remote message
    lock.lock();
    void* retval = remoteCache[std::make_pair(off, hostID)];
    lock.unlock();
    if (retval)
      return retval;
    getSystemNetworkInterface().sendAlt(hostID, pBe2Resolve,
                                        NetworkInterface::ID, off);
    do {
      if (LL::getTID() == 0)
        doNetworkWork();
      lock.lock();
      void* retval = remoteCache[std::make_pair(off, hostID)];
      lock.unlock();
      if (retval)
        return retval;
    } while (true);
  }
}

galois::runtime::PerBackend_v2& galois::runtime::getPerHostBackend() {
  static PerBackend_v2 be;
  return be;
}

////////////////////////////////////////////////////////////////////////////////
// PerThreadDist
////////////////////////////////////////////////////////////////////////////////

uint64_t PerBackend_v3::allocateOffset() {
  if (NetworkInterface::ID == 0) {
    std::lock_guard<LL::SimpleLock> lg(lock);
    auto ii = std::find(freelist.begin(), freelist.end(), false);
    if (ii == freelist.end()) {
      assert(0 && "out of dyn slots");
      abort();
    }
    *ii = true;
    return std::distance(freelist.begin(), ii);
  } else {
    // Send message and await magic
    GALOIS_DIE("not implemented");
  }
}

void PerBackend_v3::deallocateOffset(uint64_t off) {
  if (NetworkInterface::ID == 0) {
    std::lock_guard<LL::SimpleLock> lg(lock);
    assert(freelist[off] && "not allocated");
    freelist[off] = false;
  } else {
    // Send message and await magic
    GALOIS_DIE("not implemented");
  }
}

PerBackend_v3::PerBackend_v3() : freelist(dynSlots) {
  freelist[0] = true; // protect "null"
}

void PerBackend_v3::initThread() {
  lock.lock();
  if (LL::getTID() >= heads.size())
    heads.resize(LL::getTID() + 1);
  heads[LL::getTID()] = space;
  lock.unlock();
}

void* PerBackend_v3::resolveRemote_i(uint64_t offset, uint32_t hostID,
                                     uint32_t threadID) {
  if (hostID == NetworkInterface::ID) {
    return resolveThread<void>(offset, threadID);
  } else {
    // FIXME: remote message
    lock.lock();
    void* retval = remoteCache[std::make_tuple(offset, hostID, threadID)];
    lock.unlock();
    if (retval)
      return retval;
    getSystemNetworkInterface().sendAlt(hostID, pBe2Resolve,
                                        NetworkInterface::ID, offset, threadID);
    do {
      if (LL::getTID() == 0)
        doNetworkWork();
      lock.lock();
      void* retval = remoteCache[std::make_tuple(offset, hostID, threadID)];
      lock.unlock();
      if (retval)
        return retval;
    } while (true);
  }
}

void PerBackend_v3::pBe2ResolveLP(void* ptr, uint32_t srcID, uint64_t off,
                                  uint32_t threadID) {
  getPerThreadDistBackend().addRemote(ptr, srcID, off, threadID);
}

void PerBackend_v3::pBe2Resolve(uint32_t dest, uint64_t off,
                                uint32_t threadID) {
  void* ptr = getPerThreadDistBackend().resolveThread<void>(off, threadID);
  getSystemNetworkInterface().sendAlt(dest, pBe2ResolveLP, ptr,
                                      NetworkInterface::ID, off, threadID);
}

void PerBackend_v3::addRemote(void* ptr, uint32_t srcID, uint64_t off,
                              uint32_t threadID) {
  lock.lock();
  remoteCache[std::make_tuple(off, srcID, threadID)] = ptr;
  lock.unlock();
}

thread_local void* PerBackend_v3::space[dynSlots];

PerBackend_v3& galois::runtime::getPerThreadDistBackend() {
  static PerBackend_v3 be;
  return be;
};
