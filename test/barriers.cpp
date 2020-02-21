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

#include "galois/Timer.h"
#include "galois/Galois.h"
#include "galois/substrate/Barrier.h"

#include <iostream>
#include <cstdlib>
#include <unistd.h>

unsigned iter       = 1;
unsigned numThreads = 2;

char bname[100];

struct emp {
  galois::substrate::Barrier& b;

  void go() {
    for (unsigned i = 0; i < iter; ++i) {
      b.wait();
    }
  }

  template <typename T>
  void operator()(const T& t) {
    go();
  }

  template <typename T, typename C>
  void operator()(const T& t, const C& c) {
    go();
  }
};

void test(std::unique_ptr<galois::substrate::Barrier> b) {
  unsigned M = numThreads;
  if (M > 16)
    M /= 2;
  while (M) {
    galois::setActiveThreads(M); // galois::runtime::LL::getMaxThreads());
    b->reinit(M);
    galois::Timer t;
    t.start();
    emp e{*b.get()};
    galois::on_each(e);
    t.stop();
    std::cout << bname << "," << b->name() << "," << M << "," << t.get()
              << "\n";
    M -= 1;
  }
}

int main(int argc, char** argv) {
  galois::SharedMemSys Galois_runtime;
  if (argc > 1)
    iter = atoi(argv[1]);
  if (!iter)
    iter = 16 * 1024;
  if (argc > 2)
    numThreads = galois::substrate::getThreadPool().getMaxThreads();

  gethostname(bname, sizeof(bname));
  using namespace galois::substrate;
  test(createPthreadBarrier(1));
  test(createCountingBarrier(1));
  test(createMCSBarrier(1));
  test(createTopoBarrier(1));
  test(createDisseminationBarrier(1));
  return 0;
}
