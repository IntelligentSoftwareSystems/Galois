/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
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

// This example shows how to use galois::runtime::ExternalHeapAllocator
// to wrap up 3rd-party allocators and use the wrapped heap for STL containers.
#include "galois/Galois.h"
#include "galois/runtime/Mem.h"

#include <iostream>

int main() {
  galois::SharedMemSys G;

  //! [heap wrapping example]
  // Our 3rd-party heap
  using RealHeap = galois::runtime::MallocHeap;

  // Wrap RealHeap to conform to STL allocators 
  using WrappedHeap = galois::runtime::ExternalHeapAllocator<int, RealHeap>;

  // Instantiate heaps
  RealHeap externalHeap;
  WrappedHeap heap(&externalHeap);

  // Use the wrapped heap
  std::vector<int, WrappedHeap> v(heap);
  for (int i = 0; i < 5; i++) {
    v.push_back(i);
  }

  std::cout << "Use of a std::vector with a third-party allocator wrapped by galois::runtime::ExternalHeapAllocator.\n";
  for (auto& j: v) {
    std::cout << j << std::endl;
  }
  //! [heap wrapping example]

  return 0;
}
