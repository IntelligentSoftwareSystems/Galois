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

int main() {

  std::map<int, double> compare;

  std::cout << galois::runtime::gSized(std::vector<int>(100)) << "\n";
  // check by hand that gSized reduces this to a constant
  std::cout << galois::runtime::gSized(1.2, 3.4, '1') << "\n";

  std::cout << "Ser\n\n";

  for (int num = 1; num < 1024; num *= 2) {
    std::vector<double> input(1024 * num, 1.0);
    galois::Timer T;
    T.start();
    for (int i = 0; i < 1000; ++i) {
      SendBuffer b;
      galois::runtime::gSerialize(b, input);
    }
    T.stop();
    auto bytes    = sizeof(double) * 1024 * num * 1000;
    double mbytes = (double)bytes / (1024 * 1024);
    double time   = (double)T.get() / 1000;
    compare[num]  = time;
    std::cout << "Time: " << time << " sec\n";
    std::cout << "Bytes: " << mbytes << " MB\n";
    std::cout << "Throughput: " << mbytes / time << " MB/s\n";
  }

  std::cout << "\n\nSer + DeSer\n\n";

  for (int num = 1; num < 1024; num *= 2) {
    std::vector<double> input(1024 * num, 1.0);
    galois::Timer T;
    T.start();
    for (int i = 0; i < 1000; ++i) {
      SendBuffer b;
      galois::runtime::gSerialize(b, input);
      RecvBuffer r(std::move(b));
      galois::runtime::gDeserialize(r, input);
    }
    T.stop();
    auto bytes    = sizeof(double) * 1024 * num * 1000;
    double mbytes = (double)bytes / (1024 * 1024);
    double time   = (double)T.get() / 1000;
    std::cout << "Time: " << time << " sec\n";
    std::cout << "Bytes: " << mbytes << " MB\n";
    std::cout << "Throughput: " << mbytes / time << " MB/s\n";
    std::cout << "Ratio: " << compare[num] / time << " (ser/(ses+des))\n";
  }

  return 0;
}
