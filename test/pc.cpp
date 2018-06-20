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

#include "galois/substrate/PerThreadStorage.h"
#include "galois/Timer.h"
#include "galois/Galois.h"

#include <cstdlib>
#include <iostream>

using namespace galois::substrate;

int num = 1;

template <typename T>
struct testL {
  PerThreadStorage<T>& b;

  testL(PerThreadStorage<T>& B) : b(B) {}
  void operator()(unsigned t, unsigned n) {
    for (int x = 0; x < num; ++x) {
      *b.getLocal() += x;
    }
  }
};

template <typename T>
struct testR {
  PerThreadStorage<T>& b;

  testR(PerThreadStorage<T>& B) : b(B) {}
  void operator()(unsigned t, unsigned n) {
    for (int x = 0; x < num; ++x) {
      *b.getRemote((t + 1) % n) += x;
    }
  }
};

template <typename T>
void testf(const char* str) {
  PerThreadStorage<T> b;
  std::cout << "\nRunning: " << str << " sizeof " << sizeof(PerThreadStorage<T>)
            << "\n";
  galois::Timer tL;
  tL.start();
  testL<T> L(b);
  galois::on_each(L);
  tL.stop();
  galois::Timer tR;
  tR.start();
  testR<T> R(b);
  galois::on_each(R);
  tR.stop();
  std::cout << str << " L: " << tL.get() << " R: " << tR.get() << '\n';
}

int main(int argc, char** argv) {
  if (argc > 1)
    num = atoi(argv[1]);
  if (num <= 0)
    num = 1024 * 1024 * 1024;

  unsigned M = galois::substrate::getThreadPool().getMaxThreads();

  while (M) {
    galois::setActiveThreads(M); // galois::runtime::LL::getMaxThreads());
    std::cout << "Using " << M << " threads\n";

    testf<int>("int");
    testf<double>("double");

    M /= 2;
  }

  return 0;
}
