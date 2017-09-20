#include "Galois/Runtime/RemotePointer.h"
#include "Galois/Runtime/Serialize.h"

struct S {
  int x;
  int *y;
};

struct Ssub: public S {
  int z;
};

int main() {
  galois::Runtime::fatPointer ptr;
  
  galois::Runtime::Lockable* oldobj = reinterpret_cast<galois::Runtime::Lockable*>(ptr.getObj());
  for (uint32_t h = 0; h < 0x0000FFFF; h += 3) {
    ptr.setHost(h);
    assert(ptr.getHost() == h);
    assert(reinterpret_cast<galois::Runtime::Lockable*>(ptr.getObj()) == oldobj);
  }

  static_assert(std::is_trivially_copyable<int>::value, "is_trivially_copyable not well supported");
  static_assert(std::is_trivially_copyable<S>::value, "is_trivially_copyable not well supported");
  static_assert(std::is_trivially_copyable<Ssub>::value, "is_trivially_copyable not well supported");
  //static_assert(std::is_trivially_copyable<galois::Runtime::fatPointer>::value, "fatPointer should be trivially serializable");
  //static_assert(std::is_trivially_copyable<galois::Runtime::gptr<int>>::value, "RemotePointer should be trivially serializable");

  return 0;
}
