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
#include "galois/FlatMap.h"
#ifdef GALOIS_USE_EXP
#include "galois/ConcurrentFlatMap.h"
#endif
#include "galois/Timer.h"

#include <boost/iterator/counting_iterator.hpp>
#include <cstdlib>
#include <iostream>
#include <map>
#include <random>

struct element {
  volatile int val;
  element() : val() {}
  element(int x) : val(x) {}
  operator int() const { return val; }
};

std::ostream& operator<<(std::ostream& out, const element& e) {
  out << e.val;
  return out;
}

template <typename MapTy>
struct Fn1 {
  MapTy* m;
  void operator()(const int& x) const { (*m)[x] = element(x); }
};

template <typename MapTy>
struct Fn2 {
  MapTy* m;
  void operator()(const int& x) const {
    int v = (*m)[x].val;
    GALOIS_ASSERT(v == x || v == 0);
  }
};

template <typename MapTy>
void timeMapParallel(std::string c, const std::vector<int>& keys) {
  MapTy m;
  galois::Timer t1, t2;
  t1.start();
  galois::do_all(galois::iterate(keys), Fn1<MapTy>{&m});
  t1.stop();
  t2.start();
  galois::do_all(galois::iterate(keys), Fn2<MapTy>{&m});
  t2.stop();
  std::cout << c << " " << t1.get() << " " << t2.get() << "\n";
}

template <typename MapTy>
void timeMap(std::string c, const std::vector<int>& keys) {
  MapTy m;
  galois::Timer t1, t2;
  t1.start();
  for (auto& x : keys) {
    m[x] = element(x);
  }
  t1.stop();
  t2.start();
  for (auto& x : keys) {
    int v = m[x].val;
    GALOIS_ASSERT(v == x);
  }
  t2.stop();
  std::cout << c << " " << t1.get() << " " << t2.get() << "\n";
}

template <typename MapTy>
void testMap() {
  MapTy m;
  MapTy m2(m);
  MapTy m3;

  m3.insert(std::make_pair(10, 0));
  m3.insert(std::make_pair(20, 0));

  MapTy m4(m3.begin(), m3.end());

  m2 = m3;
  m3 = std::move(m2);

  m[0] = 0;
  m[1] = 1;
  m[3] = 2;
  m[3] = m[3] + 3;
  m[4] = 4;

  m.insert(std::make_pair(5, 4));
  m.insert(m4.begin(), m4.end());

  std::cout << "10 == " << m.find(10)->first << "\n";

  // m.erase(10);
  // m.erase(1);

  if (m.size() != 7 || m.empty())
    abort();
  std::swap(m, m3);
  if (m.size() != 2 || m.empty())
    abort();
  m.clear();
  if (m.size() != 0 || !m.empty())
    abort();
  std::swap(m, m3);
  if (m.size() != 7 || m.empty())
    abort();

  for (auto ii = m.begin(), ee = m.end(); ii != ee; ++ii)
    std::cout << ii->first << " " << ii->second << " ";
  std::cout << "\n";

  for (auto ii = m.cbegin(), ee = m.cend(); ii != ee; ++ii)
    std::cout << ii->first << " " << ii->second << " ";
  std::cout << "\n";

  for (auto ii = m.rbegin(), ee = m.rend(); ii != ee; ++ii)
    std::cout << ii->first << " " << ii->second << " ";
  std::cout << "\n";

  for (auto ii = m.crbegin(), ee = m.crend(); ii != ee; ++ii)
    std::cout << ii->first << " " << ii->second << " ";
  std::cout << "\n";
}

void timeTests(std::string prefix, const std::vector<int>& keys) {
  for (int i = 0; i < 3; ++i)
    timeMap<std::map<int, element>>(prefix + "std::map", keys);
  for (int i = 0; i < 3; ++i)
    timeMap<galois::flat_map<int, element>>(prefix + "flat_map", keys);
#ifdef GALOIS_USE_EXP
  for (int i = 0; i < 3; ++i)
    timeMap<galois::concurrent_flat_map<int, element>>(
        prefix + "concurrent_flat_map", keys);
  for (int i = 0; i < 3; ++i)
    timeMapParallel<galois::concurrent_flat_map<int, element>>(
        prefix + "concurrent_flat_map (parallel)", keys);
#endif
}

int main(int argc, char** argv) {
  galois::SharedMemSys Galois_runtime;
  testMap<std::map<int, element>>();
  testMap<galois::flat_map<int, element>>();
#ifdef GALOIS_USE_EXP
  testMap<galois::concurrent_flat_map<int, element>>();
#endif
  galois::setActiveThreads(8);

  int size = 100;
  if (argc > 1)
    size = atoi(argv[1]);
  if (size <= 0)
    size = 1000000;

  std::mt19937 mt(0);
  std::uniform_int_distribution<int> dist(0, size);
  std::vector<int> randomKeys;
  std::vector<int> keys;
  for (int i = 0; i < size; ++i) {
    randomKeys.push_back(dist(mt));
    keys.push_back(i);
  }

  timeTests("seq ", keys);
  timeTests("random ", randomKeys);
  return 0;
}
