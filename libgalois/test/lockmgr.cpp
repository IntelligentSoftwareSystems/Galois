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

#include "galois/runtime/Lockable.h"

#include <iostream>

using namespace galois::runtime;

struct simple : public Lockable {
  int foo;
};

char translate(int i) {
  switch (i) {
  case 0:
    return 'F';
  case 1:
    return 'N';
  case 3:
    return 'O';
  default:
    return '?';
  }
}

// FIXME: include ro tests

int main(int argc, char** argv) {
  simple s1, s2;
  LockManagerBase b1, b2;

  std::cout << translate(b1.tryAcquire(&s1, false)) << "\n";
  b1.dump(std::cout);
  b2.dump(std::cout);
  std::cout << "\n";
  std::cout << translate(b1.tryAcquire(&s1, false)) << "\n";
  b1.dump(std::cout);
  b2.dump(std::cout);
  std::cout << "\n";
  std::cout << translate(b1.tryAcquire(&s2, false)) << "\n";
  b1.dump(std::cout);
  b2.dump(std::cout);
  std::cout << "\n";
  std::cout << translate(b2.tryAcquire(&s1, false)) << "\n";
  b1.dump(std::cout);
  b2.dump(std::cout);
  std::cout << "\n";
  std::cout << translate(b2.tryAcquire(&s2, false)) << "\n";
  b1.dump(std::cout);
  b2.dump(std::cout);
  std::cout << "\n";
  auto rb1 = b1.releaseAll();
  std::cout << rb1.first << " " << rb1.second << "\n";
  b1.dump(std::cout);
  b2.dump(std::cout);
  std::cout << "\n";
  std::cout << translate(b2.tryAcquire(&s1, false)) << "\n";
  b1.dump(std::cout);
  b2.dump(std::cout);
  std::cout << "\n";
  std::cout << translate(b2.tryAcquire(&s2, false)) << "\n";
  b1.dump(std::cout);
  b2.dump(std::cout);
  std::cout << "\n";
  // b1.forceAcquire(&s1);
  // b1.dump(std::cout); b2.dump(std::cout); std::cout << "\n";
  // b1.forceAcquire(&s2);
  // b1.dump(std::cout); b2.dump(std::cout); std::cout << "\n";
  std::cout << translate(b2.tryAcquire(&s1, false)) << "\n";
  b1.dump(std::cout);
  b2.dump(std::cout);
  std::cout << "\n";
  std::cout << translate(b2.tryAcquire(&s2, false)) << "\n";
  b1.dump(std::cout);
  b2.dump(std::cout);
  std::cout << "\n";
  auto rb2 = b2.releaseAll();
  std::cout << rb2.first << " " << rb2.second << "\n";
  b1.dump(std::cout);
  b2.dump(std::cout);
  std::cout << "\n";
  rb1 = b1.releaseAll();
  std::cout << rb1.first << " " << rb1.second << "\n";
  b1.dump(std::cout);
  b2.dump(std::cout);
  std::cout << "\n";

  return 0;
}
