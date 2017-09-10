/** Test CacheManager and Remote Pointer (Local tests only) -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2012, The University of Texas at Austin. All rights reserved.
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
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#include "Galois/Runtime/CacheManager.h"
#include "Galois/Runtime/RemotePointer.h"

#include <iostream>

using namespace Galois::Runtime;

struct foo: public Lockable {
  int x;
  int y;
  friend std::ostream& operator<<(std::ostream& os, const foo& v) {
    return os << "{" << v.x << "," << v.y << "}";
  }
  foo(int _x, int _y) :x(_x), y(_y) {}
  foo(Galois::Runtime::DeSerializeBuffer& buf) { deserialize(buf); }
  foo() = default;

  typedef int tt_has_serialize;
  void serialize(Galois::Runtime::SerializeBuffer& s) const {
    gSerialize(s, x, y);
  }
  void deserialize(Galois::Runtime::DeSerializeBuffer& s) {
    gDeserialize(s, x, y);
  }
};

void test_CM() {
  auto& cm = getCacheManager();
  fatPointer fp{1, 0x010};

  std::cout << fp << " " << cm.resolve(fp) << "\n";
  cm.create(fp, foo{1,2});
  std::cout << fp << " " << cm.resolve(fp) << "\n";
  cm.evict(fp);
  std::cout << fp << " " << cm.resolve(fp) << "\n";
  cm.create(fp, foo{1,2});
  std::cout << fp << " " << cm.resolve(fp) << "\n";
}

void test_RP() {
  foo lfoo{1,2};
  gptr<foo> glfoo(&lfoo);
  gptr<foo> grfoo(1, reinterpret_cast<foo*>(0x10));
  getCacheManager().create((fatPointer)grfoo, foo{3,4});

  std::cout << "L: " << glfoo << "\n";
  std::cout << "R: " << grfoo << "\n";
  std::cout << "L: " << *glfoo << "\n";
  std::cout << "R: " << *grfoo << "\n";
}


int main() {
  std::cout << "test_CM\n";
  test_CM();
  std::cout << "test_RP\n";
  test_RP();
  return 0;
}
