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

#include "galois/Galois.h"
#include "galois/runtime/Serialize.h"

#include <iostream>

using namespace galois::runtime;

struct DistObj : public Lockable {
  int i;
  DistObj() : i(0) {}
  DistObj(RecvBuffer& buf) { deserialize(buf); }

  typedef int tt_has_serialize;
  void deserialize(RecvBuffer& buf) { gDeserialize(buf, i); }
  void serialize(SendBuffer& buf) const { gSerialize(buf, i); }
};

int main() {
  static_assert(galois::runtime::is_serializable<DistObj>::value,
                "DistObj not serializable");

  unsigned i1{0XDEADBEEF}, i2;
  gptr<DistObj> ptr1, ptr2;
  SendBuffer sbuf;
  galois::runtime::gSerialize(sbuf, ptr1, i1);

  RecvBuffer rbuf(std::move(sbuf));
  galois::runtime::gDeserialize(rbuf, ptr2, i2);

  GALOIS_ASSERT(i1 == i2);

  {
    std::vector<double> input(1024 * 1024, 1.0);
    std::vector<double> output;
    galois::Timer T;
    T.start();
    for (int i = 0; i < 100; ++i) {
      SendBuffer b;
      galois::runtime::gSerialize(b, input);
      RecvBuffer r(std::move(b));
      galois::runtime::gDeserialize(r, output);
    }
    T.stop();
    std::cout << "Time: " << T.get() << "\n";
  }

  return 0;
}
