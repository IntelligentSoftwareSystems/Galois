#include "Galois/Runtime/mm/Mem.h"
#include "Galois/Runtime/ll/gio.h"

using namespace Galois::Runtime::MM;
using namespace Galois::Runtime::LL;

struct element {
  unsigned val;
  element* next;
  element(int i): val(i), next(0) { }
};

int main(int argc, char** argv) {
  unsigned baseAllocSize = SystemBaseAlloc::AllocSize;

  FSBGaloisAllocator<element> falloc;
  element* last = nullptr;
  for (unsigned i = 0; i < baseAllocSize; ++i) {
    element* ptr = falloc.allocate(1);
    falloc.construct(ptr, i);
    ptr->next = last;
    last = ptr;
  }
  for (unsigned i = baseAllocSize - 1; i >= 0; --i) {
    GALOIS_ASSERT(last);
    GALOIS_ASSERT(last->val == i);
    element* next = last->next;
    falloc.destroy(last);
    falloc.deallocate(last, 1);
    last = next;
  }
  GALOIS_ASSERT(!last);

  VariableSizeAllocator valloc;
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
