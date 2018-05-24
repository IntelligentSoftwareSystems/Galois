#include "galois/runtime/CacheManager.h"

using namespace galois::runtime;

static thread_local ResolveCache* thread_resolve = nullptr;

////////////////////////////////////////////////////////////////////////////////

//ancor vtable
internal::remoteObj::~remoteObj() {}

////////////////////////////////////////////////////////////////////////////////

internal::remoteObj* CacheManager::resolveIncRef(fatPointer ptr) {
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
    internal::remoteObj* r = getCacheManager().resolveIncRef(ptr);
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
