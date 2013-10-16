#include "Galois/Runtime/FatPointer.h"

int main() {
  Galois::Runtime::fatPointer ptr;
  
  Galois::Runtime::Lockable* oldobj = ptr.getObj();
  for (uint32_t h = 0; h < 0x0000FFFF; h += 3) {
    ptr.setHost(h);
    assert(ptr.getHost() == h);
    assert(ptr.getObj() == oldobj);
  }

  return 0;
}
