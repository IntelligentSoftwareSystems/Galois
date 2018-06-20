// This example shows how to use galois::runtime::ExternalHeapAllocator
// to wrap up 3rd-party allocators and use the wrapped heap for STL containers.
#include "galois/Galois.h"
#include "galois/runtime/Mem.h"

#include <iostream>

int main(int argc, char* argv[]) {
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
