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

#include "galois/Galois.h"
#include "galois/Timer.h"
#include "galois/runtime/Profile.h"

#include <iostream>

template <typename V>
size_t vecSumSerial(V& vec) {
  galois::runtime::profilePapi(
      [&](void) {
        for (size_t i = 0, sz = vec.size(); i < sz; ++i) {
          vec[i] = i;
        }
      },
      "vecInit");

  size_t sum = 0;

  galois::runtime::profilePapi(
      [&](void) {
        for (size_t i = 0, sz = vec.size(); i < sz; ++i) {
          sum += vec[i];
        }
      },
      "vecSum");

  return sum;
}

template <typename V>
size_t vecSumParallel(V& vec) {
  galois::runtime::profilePapi(
      [&](void) {
        galois::do_all(galois::iterate(size_t{0}, vec.size()),
                       [&](size_t i) { vec[i] = i; });
      },
      "vecInit");

  size_t sum = 0;

  galois::runtime::profilePapi(
      [&](void) {
        galois::do_all(galois::iterate(size_t{0}, vec.size()),
                       [&](size_t i) { sum += vec[i]; });
      },
      "vecSum");

  return sum;
}

int main(int argc, char* argv[]) {

  galois::SharedMemSys G;

  unsigned long long numThreads;
  if (argc == 1) {
    numThreads = 1;
  } else if (argc == 2) {
    numThreads = galois::setActiveThreads(std::stoull(argv[1]));
  } else {
    throw std::invalid_argument(
        "Test received too many command line arguments");
  }

  galois::runtime::reportParam("NULL", "Threads", numThreads);

  size_t vecSz = 1024 * 1024;

  std::vector<size_t> vec(vecSz);

  size_t sum = vecSumSerial(vec);

  std::cout << "Array Sum = " << sum << "\n";

  return 0;
}
