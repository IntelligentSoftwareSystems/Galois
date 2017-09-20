/** Galois Remote Object Store -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2014, The University of Texas at Austin. All rights reserved.
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

#include "Galois/Runtime/CacheManager.h"

using namespace galois::Runtime;

static __thread ResolveCache* thread_resolve = nullptr;

////////////////////////////////////////////////////////////////////////////////

//ancor vtable
details::remoteObj::~remoteObj() {}

////////////////////////////////////////////////////////////////////////////////

details::remoteObj* CacheManager::resolveIncRef(fatPointer ptr) {
  assert(ptr.getHost() != NetworkInterface::ID);
  std::lock_guard<LL::SimpleLock> lgr(Lock);
  auto ii = remoteObjects.find(ptr);
  if (ii != remoteObjects.end()) {
    assert(ii->second);
    ii->second->incRef();
    return ii->second;
  }
  return nullptr;
}

void* CacheManager::resolve(fatPointer ptr) {
  assert(ptr.getHost() != NetworkInterface::ID);
  std::lock_guard<LL::SimpleLock> lgr(Lock);
  auto ii = remoteObjects.find(ptr);
  if (ii != remoteObjects.end()) {
    assert(ii->second);
    return ii->second->getObj();
  }
  return nullptr;
}

void CacheManager::evict(fatPointer ptr) {
  assert(ptr.getHost() != NetworkInterface::ID);
  std::lock_guard<LL::SimpleLock> lgr(Lock);
  auto R = remoteObjects.find(ptr);
  assert(R != remoteObjects.end() && R->second);
  garbage.push_back(R->second);
  remoteObjects.erase(R);
}

bool CacheManager::isCurrent(fatPointer ptr, void* obj) {
  return obj == resolve(ptr);
}

size_t CacheManager::CM_size(){
  return remoteObjects.size();
}
////////////////////////////////////////////////////////////////////////////////


void* ResolveCache::resolve(fatPointer ptr) {
  void* a = addrs[ptr];
  if (!a) {
    details::remoteObj* r = getCacheManager().resolveIncRef(ptr);
    if (r) {
      a = addrs[ptr] = r->getObj();
      objs.push_back(r);
    }
  }
  return a;
}

void ResolveCache::reset() {
  for (auto* p : objs)
    p->decRef();
  addrs.clear();
  objs.clear();
}

////////////////////////////////////////////////////////////////////////////////


ResolveCache* galois::runtime::getThreadResolve() {
  return thread_resolve;
}

void galois::runtime::setThreadResolve(ResolveCache* rc) {
  thread_resolve = rc;
}

CacheManager& galois::runtime::getCacheManager() {
  static CacheManager CM;
  return CM;
}
