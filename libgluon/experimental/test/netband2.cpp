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

#include <iostream>
#include <sstream>
#include <cstring>
#include <unistd.h>

#include <mpi.h>

#include "galois/runtime/Network.h"
#include "galois/runtime/Barrier.h"
#include "galois/Timer.h"

using namespace galois::runtime;

static std::atomic<int> num;

void func(RecvBuffer&) { ++num; }

volatile int cont = 0;

int main(int argc, char** argv) {
  num        = 0;
  int trials = 1000000;
  if (argc > 1)
    trials = atoi(argv[1]);

  auto& net = getSystemNetworkInterface();
  auto& bar = getSystemBarrier();

  if (net.Num != 2) {
    std::cerr << "Just run with 2 hosts\n";
    return 1;
  }

  // while (!cont) {}

  for (int s = 10; s < trials; s *= 1.1) { // 1069 is from 10. 243470 is also
    std::vector<char> vec(s);
    galois::Timer T1, T2, T3;
    bar.wait();
    T3.start();
    T1.start();
    SendBuffer buf;
    gSerialize(buf, vec);
    T1.stop();
    int oldnum = num;
    bar.wait();
    T2.start();
    if (net.ID == 0) {
      net.send(1, func, buf);
      net.flush();
    } else {
      while (num == oldnum) {
        net.handleReceives();
      }
    }
    T2.stop();
    bar.wait();
    T3.stop();
    bar.wait();
    std::cerr << "H" << net.ID << " size " << s << " T1 " << T1.get() << " T2 "
              << T2.get() << " T3 " << T3.get() << " B "
              << (T3.get() - T1.get() ? s / (T3.get() - T1.get()) : 0) << "\n";
    bar.wait();
  }
  return 0;
}
