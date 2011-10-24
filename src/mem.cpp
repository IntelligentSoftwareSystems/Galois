/** heap building blocks implementation -*- C++ -*-
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

using namespace GaloisRuntime;
using namespace MM;

// Abstract away mmap
static const int _PROT = PROT_READ | PROT_WRITE;
static const int _MAP_BASE = MAP_ANONYMOUS | MAP_PRIVATE;
#ifdef MAP_POPULATE
static const int _MAP_POP  = MAP_POPULATE | _MAP_BASE;
#endif
#ifdef MAP_HUGETLB
static const int _MAP_HUGE = MAP_HUGETLB | _MAP_POP;
#endif

mmapWrapper::mmapWrapper() {
#ifndef MAP_POPULATE
  reportWarning("No MAP_POPULATE");
#endif
#ifndef MAP_HUGETLB
  reportWarning("No MAP_HUGETLB");
#endif
}

void* mmapWrapper::_alloc() {
#ifdef __linux__
  //serialize mmap in userspace because it prevents
  //linux from sleeping for undefined amounts of time
  static SimpleLock<int, true> L;
#else
  static SimpleLock<int,false> L;
#endif
  L.lock();
  void* ptr = 0;

#ifdef MAP_HUGETLB
  //First try huge
  ptr = mmap(0, AllocSize, _PROT, _MAP_HUGE, -1, 0);
#endif

#ifdef MAP_POPULATE
  //Then try populate
  if (!ptr || ptr == MAP_FAILED)
    ptr = mmap(0, AllocSize, _PROT, _MAP_POP, -1, 0);
#endif
  //Then try normal
  if (!ptr || ptr == MAP_FAILED)
    ptr = mmap(0, AllocSize, _PROT, _MAP_BASE, -1, 0);

  if (!ptr || ptr == MAP_FAILED) {
    L.unlock();
    reportWarning("Memory Allocation Failed");
    assert(0 && "mmap failed");
    abort();
  }

  L.unlock();
  return ptr;
}

void mmapWrapper::_free(void* ptr) {
  munmap(ptr, AllocSize);
}

//Anchor the class
SelfLockFreeListHeap<mmapWrapper> SystemBaseAlloc::Source;
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
