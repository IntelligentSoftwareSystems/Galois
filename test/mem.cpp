#include "galois/runtime/Mem.h"
#include "galois/gIO.h"

using namespace galois::runtime;
using namespace galois::substrate;

struct element {
  unsigned val;
  element* next;
  element(int i) : val(i), next(0) {}
};

int main(int argc, char** argv) {
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
