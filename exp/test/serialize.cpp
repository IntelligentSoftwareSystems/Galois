#include "Galois/Galois.h"
#include "Galois/Runtime/DistSupport.h"

using namespace Galois::Runtime;

struct DistObj : public Lockable {
  int i;
  DistObj(): i(0) { }
  DistObj(RecvBuffer& buf) { deserialize(buf); }
  
  typedef int tt_has_serialize;
  void deserialize(RecvBuffer& buf) {
    gDeserialize(buf, i);
  }
  void serialize(SendBuffer& buf) const {
    gSerialize(buf, i);
  }
};

int main() {
  static_assert(Galois::Runtime::is_serializable<DistObj>::value, "DistObj not serializable");

  unsigned i1 {0XDEADBEEF}, i2;
  gptr<DistObj> ptr1, ptr2;
  SendBuffer sbuf;
  Galois::Runtime::gSerialize(sbuf, ptr1, i1);

  RecvBuffer rbuf(std::move(sbuf));
  Galois::Runtime::gDeserialize(rbuf, ptr2, i2);

  GALOIS_ASSERT(i1 == i2);

  return 0;
}
