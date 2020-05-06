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

#include "galois/Galois.h"
#include "galois/gIO.h"
#include "galois/runtime/Mem.h"

using namespace galois::runtime;
using namespace galois::substrate;

struct element {
  unsigned val;
  element* next;
  element(int i) : val(i), next(0) {}
};

int main() {
  galois::SharedMemSys Galois_runtime;
  unsigned baseAllocSize = SystemHeap::AllocSize;

  FixedSizeAllocator<element> falloc;
  element* last = nullptr;
  for (unsigned i = 0; i < baseAllocSize; ++i) {
    element* ptr = falloc.allocate(1);
    falloc.construct(ptr, i);
    ptr->next = last;
    last      = ptr;
  }
  for (unsigned i = 0; i < baseAllocSize; ++i) {
    GALOIS_ASSERT(last);
    GALOIS_ASSERT(last->val == baseAllocSize - 1 - i);
    element* next = last->next;
    falloc.destroy(last);
    falloc.deallocate(last, 1);
    last = next;
  }
  GALOIS_ASSERT(!last);

  VariableSizeHeap valloc;
  size_t allocated;
  GALOIS_ASSERT(1 < baseAllocSize);
  valloc.allocate(1, allocated);
  GALOIS_ASSERT(allocated == 1);

  valloc.allocate(baseAllocSize + 1, allocated);
  GALOIS_ASSERT(allocated <= baseAllocSize);

  int toAllocate = baseAllocSize + 1;
  while (toAllocate) {
    valloc.allocate(toAllocate, allocated);
    toAllocate -= allocated;
    GALOIS_ASSERT(allocated);
  }

  return 0;
}
