#include "Galois/Galois.h"
#include "Galois/Runtime/Serialize.h"

#include <iostream>

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

  {
    std::vector<double> input(1024*1024, 1.0);
    std::vector<double> output;
    Galois::Timer T;
    T.start();
    for (int i = 0; i < 100; ++i) {
      SendBuffer b;
      Galois::Runtime::gSerialize(b, input);
      RecvBuffer r(std::move(b));
      Galois::Runtime::gDeserialize(r, output);
    }
    T.stop();
    std::cout << "Time: " << T.get() << "\n";
  }

  return 0;
}
