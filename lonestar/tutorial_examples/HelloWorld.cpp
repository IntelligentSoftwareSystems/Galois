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
#include <boost/iterator/counting_iterator.hpp>
#include <iostream>

//! [do_all example]
struct HelloWorld {
  void operator()(int i) const { std::cout << "Hello " << i << "\n"; }
};

void helloWorld(int i) { std::cout << "Hello " << i << "\n"; }

int main(int argc, char** argv) {
  galois::SharedMemSys G;

  if (argc < 3) {
    std::cerr << "<num threads> <num of iterations>\n";
    return 1;
  }
  unsigned int numThreads = atoi(argv[1]);
  int n                   = atoi(argv[2]);

  numThreads = galois::setActiveThreads(numThreads);
  std::cout << "Using " << numThreads << " threads and " << n
            << " iterations\n";

  std::cout << "Using a lambda\n";
  galois::do_all(galois::iterate(0, n),
                 [](int i) { std::cout << "Hello " << i << "\n"; });

  std::cout << "Using a function object\n";
  galois::do_all(galois::iterate(0, n), HelloWorld());

  std::cout << "Using a function pointer (discouraged)\n";
  galois::do_all(galois::iterate(0, n), &helloWorld);

  //! [do_all example]

  return 0;
}
