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
//#include "Galois/Runtime/Support.h"

#include <map>

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

  if (!localAllocators)
    localAllocators = new AllocMap;

  auto& lentry = (*localAllocators)[size];
  if (lentry)
    return lentry;

  lock.lock();
  auto& gentry = allocators[size];
  if (!gentry)
    gentry = new SizedAlloc();
  lentry = gentry;
  lock.unlock();
  return  lentry;
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
  for (AllocatorsMap::iterator it = allocators.begin(), end = allocators.end();
      it != end; ++it) {
    delete it->second;
  }
}
#endif
