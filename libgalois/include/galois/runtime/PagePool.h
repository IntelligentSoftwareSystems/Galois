/**
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of XYZ License (a
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

#ifndef GALOIS_RUNTIME_PAGEPOOL_H
#define GALOIS_RUNTIME_PAGEPOOL_H

#include "galois/gIO.h"
#include "galois/substrate/SimpleLock.h"
#include "galois/substrate/PtrLock.h"
#include "galois/substrate/CacheLineStorage.h"
#include "galois/substrate/PageAlloc.h"
#include "galois/substrate/ThreadPool.h"

#include <unordered_map>
#include <vector>
#include <mutex>
#include <numeric>
#include <deque>

#include <cstddef>

namespace galois {
namespace runtime {

//! Low level page pool (individual pages, use largeMalloc for large blocks)

void* pagePoolAlloc();
void pagePoolFree(void*);
void pagePoolPreAlloc(unsigned);

// Size of returned pages
size_t pagePoolSize();

//! Returns total large pages allocated by Galois memory management subsystem
int numPagePoolAllocTotal();
//! Returns total large pages allocated for thread by Galois memory management
//! subsystem
int numPagePoolAllocForThread(unsigned tid);

namespace internal {

struct FreeNode {
  FreeNode* next;
};

typedef galois::substrate::PtrLock<FreeNode> HeadPtr;
typedef galois::substrate::CacheLineStorage<HeadPtr> HeadPtrStorage;

// Tracks pages allocated
template <typename _UNUSED = void>
class PageAllocState {
  std::deque<std::atomic<int>> counts;
  std::vector<HeadPtrStorage> pool;
  std::unordered_map<void*, int> ownerMap;
  galois::substrate::SimpleLock mapLock;

  void* allocFromOS() {
    void* ptr = galois::substrate::allocPages(1, true);
    assert(ptr);
    auto tid = galois::substrate::ThreadPool::getTID();
    counts[tid] += 1;
    std::lock_guard<galois::substrate::SimpleLock> lg(mapLock);
    ownerMap[ptr] = tid;
    return ptr;
  }

public:
  PageAllocState() {
    auto num = galois::substrate::getThreadPool().getMaxThreads();
    counts.resize(num);
    pool.resize(num);
  }

  int count(int tid) const { return counts[tid]; }

  int countAll() const {
    return std::accumulate(counts.begin(), counts.end(), 0);
  }

  void* pageAlloc() {
    auto tid    = galois::substrate::ThreadPool::getTID();
    HeadPtr& hp = pool[tid].data;
    if (hp.getValue()) {
      hp.lock();
      FreeNode* h = hp.getValue();
      if (h) {
        hp.unlock_and_set(h->next);
        return h;
      }
      hp.unlock();
    }
    return allocFromOS();
  }

  void pageFree(void* ptr) {
    assert(ptr);
    mapLock.lock();
    assert(ownerMap.count(ptr));
    int i = ownerMap[ptr];
    mapLock.unlock();
    HeadPtr& hp = pool[i].data;
    hp.lock();
    FreeNode* nh = reinterpret_cast<FreeNode*>(ptr);
    nh->next     = hp.getValue();
    hp.unlock_and_set(nh);
  }

  void pagePreAlloc() { pageFree(allocFromOS()); }
};

//! Initialize PagePool, used by runtime::init();
void setPagePoolState(PageAllocState<>* pa);

} // end namespace internal

} // end namespace runtime
} // end namespace galois

#endif
