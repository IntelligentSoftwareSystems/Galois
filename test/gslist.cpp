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
#include "galois/gslist.h"
#include "galois/gIO.h"
#include "galois/runtime/Mem.h"
#include <map>

int main(int argc, char** argv) {
  galois::SharedMemSys Galois_runtime;
  typedef galois::runtime::FixedSizeHeap Heap;
  typedef std::unique_ptr<Heap> HeapPtr;
  typedef galois::substrate::PerThreadStorage<HeapPtr> Heaps;
  typedef galois::concurrent_gslist<int> Collection;
  int numThreads = 2;
  unsigned size  = 100;
  if (argc > 1)
    numThreads = atoi(argv[1]);
  if (size <= 0)
    numThreads = 2;
  if (argc > 2)
    size = atoi(argv[2]);
  if (size <= 0)
    size = 10000;

  galois::setActiveThreads(numThreads);

  Heaps heaps;
  Collection c;

  galois::on_each([&](unsigned id, unsigned total) {
    HeapPtr& hp = *heaps.getLocal();
    hp          = HeapPtr(new Heap(sizeof(Collection::block_type)));
    for (unsigned i = 0; i < size; ++i)
      c.push_front(*hp, i);
  });

  std::map<int, int> counter;
  for (auto i : c) {
    counter[i] += 1;
  }
  for (unsigned i = 0; i < size; ++i) {
    GALOIS_ASSERT(counter[i] == numThreads);
  }
  GALOIS_ASSERT(counter.size() == size);

  galois::on_each([&](unsigned id, unsigned total) {
    while (c.pop_front(Collection::promise_to_dealloc()))
      ;
  });

  return 0;
}
