/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
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

#include "galois/runtime/Mem.h"

#include <map>
#include <mutex>

using namespace galois::runtime;

// Anchor the class
SystemHeap::SystemHeap() { assert(AllocSize == runtime::pagePoolSize()); }

SystemHeap::~SystemHeap() {}

#ifndef GALOIS_FORCE_STANDALONE
thread_local SizedHeapFactory::HeapMap* SizedHeapFactory::localHeaps = 0;

SizedHeapFactory::SizedHeap*
SizedHeapFactory::getHeapForSize(const size_t size) {
  if (size == 0)
    return 0;
  return Base::getInstance()->getHeap(size);
}

SizedHeapFactory::SizedHeap* SizedHeapFactory::getHeap(const size_t size) {
  typedef SizedHeapFactory::HeapMap HeapMap;

  if (!localHeaps) {
    std::lock_guard<galois::substrate::SimpleLock> ll(lock);
    localHeaps = new HeapMap;
    allLocalHeaps.push_front(localHeaps);
  }

  auto& lentry = (*localHeaps)[size];
  if (lentry)
    return lentry;

  {
    std::lock_guard<galois::substrate::SimpleLock> ll(lock);
    auto& gentry = heaps[size];
    if (!gentry)
      gentry = new SizedHeap();
    lentry = gentry;
    return lentry;
  }
}

Pow_2_BlockHeap::Pow_2_BlockHeap(void) throw() : heapTable() {
  populateTable();
}

SizedHeapFactory::SizedHeapFactory() : lock() {}

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
