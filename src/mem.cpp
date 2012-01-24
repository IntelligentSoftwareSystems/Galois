/** Memory allocator implementation -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
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
 * @section Description
 *
 * Strongly inspired by heap layers:
 *  http://www.heaplayers.org/
 * FSB is modified from:
 *  http://warp.povusers.org/FSBAllocator/
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#include "Galois/Runtime/mem.h"
#include "Galois/Runtime/Support.h"

#include <map>
#ifdef __linux__
#include <linux/mman.h>
#endif
#include <sys/mman.h>

#ifdef GALOIS_NUMA
#include <numa.h>
#endif

#include <iostream>

using namespace GaloisRuntime;
using namespace MM;

// mmap flags
static const int _PROT = PROT_READ | PROT_WRITE;
static const int _MAP_BASE = MAP_ANONYMOUS | MAP_PRIVATE;
#ifdef MAP_POPULATE
static const int _MAP_POP  = MAP_POPULATE | _MAP_BASE;
#endif
#ifdef MAP_HUGETLB
static const int _MAP_HUGE = MAP_HUGETLB | _MAP_POP;
#endif

//Anchor the class
SystemBaseAlloc::SystemBaseAlloc() {}
SystemBaseAlloc::~SystemBaseAlloc() {}

PtrLock<SizedAllocatorFactory*, true> SizedAllocatorFactory::instance;
#ifndef USEMALLOC
SizedAllocatorFactory::SizedAlloc* 
SizedAllocatorFactory::getAllocatorForSize(size_t size) {
  lock.lock();
  SizedAlloc*& retval = allocators[size];
  if (!retval)
    retval = new SizedAlloc;
  lock.unlock();
 
  return retval;
}

SizedAllocatorFactory::~SizedAllocatorFactory() {
  for (AllocatorsTy::iterator it = allocators.begin(), end = allocators.end();
      it != end; ++it) {
    delete it->second;
  }
}
#endif

namespace {
class LowLevelAllocator {
  struct FreeNode {
    FreeNode* next;
  };
  
PtrLock<FreeNode*, true> heads;
#ifdef __linux__
  //serialize mmap in userspace because it prevents
  //linux from sleeping for undefined amounts of time
  SimpleLock<int, true> allocLock;
#else
  SimpleLock<int,false> allocLock;
#endif

  unsigned num;

  void* allocPage() {
#ifdef GALOIS_USEMALLOC
    return malloc(pageSize);
#else

    void* ptr = 0;
    allocLock.lock();
    ++num;
#ifdef MAP_HUGETLB
    //First try huge
    ptr = mmap(0, pageSize, _PROT, _MAP_HUGE, -1, 0);
#endif
    
#ifdef MAP_POPULATE
    //Then try populate
    if (!ptr || ptr == MAP_FAILED)
      ptr = mmap(0, pageSize, _PROT, _MAP_POP, -1, 0);
#endif
    //Then try normal
    if (!ptr || ptr == MAP_FAILED)
      ptr = mmap(0, pageSize, _PROT, _MAP_BASE, -1, 0);
    
    if (!ptr || ptr == MAP_FAILED) {
      allocLock.unlock();
      reportWarning("Memory Allocation Failed");
      assert(0 && "mmap failed");
      abort();
    }
    allocLock.unlock();
    return ptr;
#endif
  }

  void freePage(void* ptr) {
#ifdef GALOIS_USEMALLOC
    free(ptr); // This is broken for multi-page allocs
#else
    munmap(ptr, pageSize);
#endif   
  }

public:
  LowLevelAllocator() :num(0) {
#ifndef MAP_POPULATE
    reportWarning("No MAP_POPULATE");
#endif
#ifndef MAP_HUGETLB
    reportWarning("No MAP_HUGETLB");
#endif
    // for (int x = 0; x < 6200; ++x)
    //   deallocate(allocPage()); //preallocate
  }

  ~LowLevelAllocator() {
    FreeNode* h = heads.getValue();
    heads.setValue(0);
    while (h) {
      FreeNode* n = h->next;
      freePage(h);
      h = n;
    }
    reportInfo("Allocated: ", num);
  }

  void print() {
    reportInfo("Allocated Early ", num);
  }

  void* allocate() {
    //has data
    if (heads.getValue()) {
      heads.lock();
      FreeNode* h = heads.getValue();
      if (h) {
	heads.unlock_and_set(h->next);
	return h;
      }
      heads.unlock();
    }
    return allocPage();
  }

  void deallocate(void* m) {
    FreeNode* nh = reinterpret_cast<FreeNode*>(m);
    heads.lock();
    nh->next = heads.getValue();
    heads.unlock_and_set(nh);
  }
};
}

static LowLevelAllocator SysAlloc;

// void* GaloisRuntime::MM::pageAlloc() {
//   return SysAlloc.allocate();
// }

// void GaloisRuntime::MM::pageFree(void* m) {
//   SysAlloc.deallocate(m);
// }
// void GaloisRuntime::MM::pagePrintInfo() {
//   SysAlloc.print();
// }


void* GaloisRuntime::MM::largeAlloc(size_t len) {
  void* data = 0;
#ifdef GALOIS_NUMA
  bitmask* nm = numa_allocate_nodemask();
  unsigned int num = GaloisRuntime::getSystemThreadPool().getActiveThreads();
  for (unsigned y = 0; y < num; ++y)
    numa_bitmask_setbit(nm, y/4);
  data = numa_alloc_interleaved_subset(len, nm);
  numa_free_nodemask(nm);
#else
  data = malloc(len);
#endif
  if (!data)
    abort();
  return data;
}

void GaloisRuntime::MM::largeFree(void* m, size_t len) {
#ifdef GALOIS_NUMA
  numa_free(m, len);
#else
  free(m);
#endif
}
