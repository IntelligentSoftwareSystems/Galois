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
#include "Galois/Runtime/mm/Mem.h"

#include <map>
#include <mutex>

using namespace Galois::Runtime;
using namespace MM;
using namespace LL;

//Anchor the class
SystemBaseAlloc::SystemBaseAlloc() {}
SystemBaseAlloc::~SystemBaseAlloc() {}

PtrLock<SizedAllocatorFactory, true> SizedAllocatorFactory::instance;
__thread SizedAllocatorFactory::AllocatorsMap* SizedAllocatorFactory::localAllocators = 0;

#ifndef USEMALLOC
SizedAllocatorFactory::SizedAlloc* 
SizedAllocatorFactory::getAllocatorForSize(const size_t size) {
  if (size == 0)
    return 0;
  return getInstance()->getAllocForSize(size);
}

SizedAllocatorFactory::SizedAlloc* 
SizedAllocatorFactory::getAllocForSize(const size_t size) {
  typedef SizedAllocatorFactory::AllocatorsMap AllocMap;

  if (!localAllocators) {
    std::lock_guard<SimpleLock> ll(lock);
    localAllocators = new AllocMap;
    allLocalAllocators.push_front(localAllocators);
  }

  auto& lentry = (*localAllocators)[size];
  if (lentry)
    return lentry;

  {
    std::lock_guard<SimpleLock> ll(lock);
    auto& gentry = allocators[size];
    if (!gentry)
      gentry = new SizedAlloc();
    lentry = gentry;
    return lentry;
  }
}

SizedAllocatorFactory* SizedAllocatorFactory::getInstance() {
  SizedAllocatorFactory* f = instance.getValue();
  if (f)
    return f;
  
  instance.lock();
  f = instance.getValue();
  if (f) {
    instance.unlock();
  } else {
    f = new SizedAllocatorFactory();
    instance.unlock_and_set(f);
  }
  return f;
}

SizedAllocatorFactory::SizedAllocatorFactory() :lock() {}

SizedAllocatorFactory::~SizedAllocatorFactory() {
  // TODO destructor ordering problem: there may be pointers to deleted
  // SizedAlloc when this Factory is destroyed before dependent
  // FixedSizeAllocators.
  for (auto entry : allocators)
    delete entry.second;
  for (auto mptr : allLocalAllocators)
    delete mptr;
}
#endif
