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

#include "galois/runtime/NetworkIO.h"

int main() {
  uint32_t ID, Num;
  std::unique_ptr<galois::runtime::NetworkIO> net;

  std::tie(net, ID, Num) = galois::runtime::makeNetworkIOMPI();

  std::cout << ID << " " << Num << "\n";

  for (int x = 1; x <= 100; ++x) {

    for (int i = 0; i < Num; ++i) {
      galois::runtime::NetworkIO::message m;
      m.len = x;
      m.data.reset(new uint8_t[x]);
      m.host = i;
      for (int y = 0; y < x; ++y)
        m.data[y] = ID;
      net->enqueue(std::move(m));
    }

    for (int i = 0; i < Num; ++i) {
      galois::runtime::NetworkIO::message m;
      do {
        m = net->dequeue();
      } while (!m.len);
      std::cout << ID << ":" << m.len << ":";
      for (int y = 0; y < m.len; ++y)
        std::cout << " " << (char)(m.data[y] + '0');
      std::cout << "\n";
    }
  }

  return 0;
}
