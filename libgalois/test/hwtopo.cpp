/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
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

#include "galois/substrate/HWTopo.h"
#include "galois/gIO.h"

#include <iostream>

void printMyTopo() {
  auto t = galois::substrate::getHWTopo();
  std::cout << "T,C,P,N: " << t.machineTopoInfo.maxThreads << " "
            << t.machineTopoInfo.maxCores << " " << t.machineTopoInfo.maxSockets
            << " " << t.machineTopoInfo.maxNumaNodes << "\n";
  for (unsigned i = 0; i < t.machineTopoInfo.maxThreads; ++i) {
    auto& c = t.threadTopoInfo[i];
    std::cout << "tid: " << c.tid << " leader: " << c.socketLeader
              << " socket: " << c.socket << " numaNode: " << c.numaNode
              << " cumulativeMaxSocket: " << c.cumulativeMaxSocket
              << " osContext: " << c.osContext
              << " osNumaNode: " << c.osNumaNode << "\n";
  }
}

void test(const std::string& name, const std::vector<int>& found,
          const std::vector<int>& expected) {
  if (found != expected) {
    std::cerr << "test " << name << " failed\n";

    std::cerr << "found: ";
    for (auto i : found) {
      std::cerr << i;
    }
    std::cerr << "\n";

    std::cerr << "expected: ";
    for (auto i : expected) {
      std::cerr << i;
    }
    std::cerr << "\n";
    std::abort();
  }
}

int main() {
  printMyTopo();

  using namespace galois::substrate;

  test("parse with spaces", parseCPUList("     0   \n"), std::vector<int>{0});
  test("parse empty", parseCPUList("        \n"), std::vector<int>{});
  test("parse singletons", parseCPUList("     0,1,2   \n"),
       std::vector<int>{0, 1, 2});
  test("parse mix of singletons and ranges", parseCPUList("     0,1,2-4   \n"),
       std::vector<int>{0, 1, 2, 3, 4});
  test("parse multiple ranges", parseCPUList("     0-1,2-4   \n"),
       std::vector<int>{0, 1, 2, 3, 4});
  test("parse range", parseCPUList("     0-4   \n"),
       std::vector<int>{0, 1, 2, 3, 4});

  return 0;
}
