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

#include "galois/runtime/RemotePointer.h"
#include "galois/runtime/Serialize.h"

struct S {
  int x;
  int* y;
};

struct Ssub : public S {
  int z;
};

int main() {
  galois::runtime::fatPointer ptr;

  galois::runtime::Lockable* oldobj =
      reinterpret_cast<galois::runtime::Lockable*>(ptr.getObj());
  for (uint32_t h = 0; h < 0x0000FFFF; h += 3) {
    ptr.setHost(h);
    assert(ptr.getHost() == h);
    assert(reinterpret_cast<galois::runtime::Lockable*>(ptr.getObj()) ==
           oldobj);
  }

  static_assert(std::is_trivially_copyable<int>::value,
                "is_trivially_copyable not well supported");
  static_assert(std::is_trivially_copyable<S>::value,
                "is_trivially_copyable not well supported");
  static_assert(std::is_trivially_copyable<Ssub>::value,
                "is_trivially_copyable not well supported");
  // static_assert(std::is_trivially_copyable<galois::runtime::fatPointer>::value,
  // "fatPointer should be trivially serializable");
  // static_assert(std::is_trivially_copyable<galois::runtime::gptr<int>>::value,
  // "RemotePointer should be trivially serializable");

  return 0;
}
