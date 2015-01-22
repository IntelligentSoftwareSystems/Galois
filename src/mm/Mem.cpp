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
SystemHeap::SystemHeap() {}
SystemHeap::~SystemHeap() {}

#ifndef GALOIS_FORCE_STANDALONE
PtrLock<SizedHeapFactory, true> SizedHeapFactory::instance;
__thread SizedHeapFactory::HeapMap* SizedHeapFactory::localHeaps = 0;
PtrLock<Pow_2_BlockHeap, true>  Pow_2_BlockHeap::instance;

SizedHeapFactory::SizedHeap* 
SizedHeapFactory::getHeapForSize(const size_t size) {
  if (size == 0)
    return 0;
  return getInstance()->getHeap(size);
}

SizedHeapFactory::SizedHeap* 
SizedHeapFactory::getHeap(const size_t size) {
  typedef SizedHeapFactory::HeapMap HeapMap;

  if (!localHeaps) {
    std::lock_guard<SimpleLock> ll(lock);
    localHeaps = new HeapMap;
    allLocalHeaps.push_front(localHeaps);
  }

  auto& lentry = (*localHeaps)[size];
  if (lentry)
    return lentry;

  {
    std::lock_guard<SimpleLock> ll(lock);
    auto& gentry = heaps[size];
    if (!gentry)
      gentry = new SizedHeap();
    lentry = gentry;
    return lentry;
  }
}

SizedHeapFactory* SizedHeapFactory::getInstance() {
  SizedHeapFactory* f = instance.getValue();
  if (f)
    return f;
  
  instance.lock();
  f = instance.getValue();
  if (f) {
    instance.unlock();
  } else {
    f = new SizedHeapFactory();
    instance.unlock_and_set(f);
  }
  return f;
}

Pow_2_BlockHeap::Pow_2_BlockHeap (void) throw (): heapTable () {
  populateTable ();
}

Pow_2_BlockHeap* Pow_2_BlockHeap::getInstance() {
  Pow_2_BlockHeap* f = instance.getValue();
  if (f)
    return f;
  
  instance.lock();
  f = instance.getValue();
  if (f) {
    instance.unlock();
  } else {
    f = new Pow_2_BlockHeap();
    instance.unlock_and_set(f);
  }
  return f;
}

SizedHeapFactory::SizedHeapFactory() :lock() {}

SizedHeapFactory::~SizedHeapFactory() {
  // TODO destructor ordering problem: there may be pointers to deleted
  // SizedHeap when this Factory is destroyed before dependent
  // FixedSizeHeaps.
  for (auto entry : heaps)
    delete entry.second;
  for (auto mptr : allLocalHeaps)
    delete mptr;
}
#endif
