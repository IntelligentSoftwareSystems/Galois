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
#include "galois/ParallelSTL.h"
#include "galois/Timer.h"

#include <iostream>
#include <cstdlib>
#include <numeric>

int RandomNumber() { return (rand() % 1000000); }
bool IsOdd(int i) { return ((i % 2) == 1); }

struct IsOddS {
  bool operator()(int i) const { return ((i % 2) == 1); }
};

int vectorSize = 1;

int do_sort() {

  unsigned M = galois::substrate::getThreadPool().getMaxThreads();
  std::cout << "sort:\n";

  while (M) {

    galois::setActiveThreads(M); // galois::runtime::LL::getMaxThreads());
    std::cout << "Using " << M << " threads\n";

    std::vector<unsigned> V(vectorSize);
    std::generate(V.begin(), V.end(), RandomNumber);
    std::vector<unsigned> C = V;

    galois::Timer t;
    t.start();
    galois::ParallelSTL::sort(V.begin(), V.end());
    t.stop();

    galois::Timer t2;
    t2.start();
    std::sort(C.begin(), C.end());
    t2.stop();

    bool eq = std::equal(C.begin(), C.end(), V.begin());

    std::cout << "Galois: " << t.get() << " STL: " << t2.get()
              << " Equal: " << eq << "\n";

    if (!eq) {
      std::vector<unsigned> R = V;
      std::sort(R.begin(), R.end());
      if (!std::equal(C.begin(), C.end(), R.begin()))
        std::cout << "Cannot be made equal, sort mutated array\n";
      for (size_t x = 0; x < V.size(); ++x) {
        std::cout << x << "\t" << V[x] << "\t" << C[x];
        if (V[x] != C[x])
          std::cout << "\tDiff";
        if (V[x] < C[x])
          std::cout << "\tLT";
        if (V[x] > C[x])
          std::cout << "\tGT";
        std::cout << "\n";
      }
      return 1;
    }

    M >>= 1;
  }

  return 0;
}

int do_count_if() {

  unsigned M = galois::substrate::getThreadPool().getMaxThreads();
  std::cout << "count_if:\n";

  while (M) {

    galois::setActiveThreads(M); // galois::runtime::LL::getMaxThreads());
    std::cout << "Using " << M << " threads\n";

    std::vector<unsigned> V(vectorSize);
    std::generate(V.begin(), V.end(), RandomNumber);

    unsigned x1, x2;

    galois::Timer t;
    t.start();
    x1 = galois::ParallelSTL::count_if(V.begin(), V.end(), IsOddS());
    t.stop();

    galois::Timer t2;
    t2.start();
    x2 = std::count_if(V.begin(), V.end(), IsOddS());
    t2.stop();

    std::cout << "Galois: " << t.get() << " STL: " << t2.get()
              << " Equal: " << (x1 == x2) << "\n";
    M >>= 1;
  }

  return 0;
}

template <typename T>
struct mymax : std::binary_function<T, T, T> {
  T operator()(const T& x, const T& y) const { return std::max(x, y); }
};

int do_accumulate() {

  unsigned M = galois::substrate::getThreadPool().getMaxThreads();
  std::cout << "accumulate:\n";

  while (M) {
    galois::setActiveThreads(M); // galois::runtime::LL::getMaxThreads());
    std::cout << "Using " << M << " threads\n";

    std::vector<unsigned> V(vectorSize);
    std::generate(V.begin(), V.end(), RandomNumber);

    unsigned x1, x2;

    galois::Timer t;
    t.start();
    x1 = galois::ParallelSTL::accumulate(V.begin(), V.end(), 0u,
                                         mymax<unsigned>());
    t.stop();

    galois::Timer t2;
    t2.start();
    x2 = std::accumulate(V.begin(), V.end(), 0u, mymax<unsigned>());
    t2.stop();

    std::cout << "Galois: " << t.get() << " STL: " << t2.get()
              << " Equal: " << (x1 == x2) << "\n";
    if (x1 != x2)
      std::cout << x1 << " " << x2 << "\n";
    M >>= 1;
  }

  return 0;
}

int main(int argc, char** argv) {
  galois::SharedMemSys Galois_runtime;
  if (argc > 1)
    vectorSize = atoi(argv[1]);
  if (vectorSize <= 0)
    vectorSize = 1024 * 1024 * 16;

  int ret = 0;
  //  ret |= do_sort();
  //  ret |= do_count_if();
  ret |= do_accumulate();
  return ret;
}
