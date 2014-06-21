#include "Galois/gdeque.h"
#include "Galois/FixedSizeRing.h"
#include "Galois/Bag.h"
#include "Galois/Runtime/ll/gio.h"
#include "Galois/Timer.h"

#include <boost/iterator/counting_iterator.hpp>

#include <iostream>
#include <cassert>
#include <string>
#include <deque>
#include <vector>
#include <random>

template<typename C>
void testBasic(std::string prefix, C&& collection, int N) {
  assert(N > 0);
  C c = std::move(collection);
  for (int i = 0; i < N; ++i)
    c.push_back(i);

  int i = 0;
  for (auto it = c.begin(); it != c.end(); ++it, ++i) {
    GALOIS_ASSERT(*it == i, prefix);
  }

  i = N - 1;
  for (auto it = c.rbegin(); it != c.rend(); ++it, --i) {
    GALOIS_ASSERT(*it == i, prefix);
  }
  
  GALOIS_ASSERT(c.size() == N, prefix);

  GALOIS_ASSERT(c.size() == std::distance(c.begin(), c.end()), prefix);

  i = N - 1;
  for (; !c.empty(); --i, c.pop_back()) {
    GALOIS_ASSERT(c.back() == i, prefix);
  }

  GALOIS_ASSERT(c.size() == 0, prefix);
  GALOIS_ASSERT(c.size() == std::distance(c.begin(), c.end()), prefix);
}

template<typename C>
void testSort(std::string prefix, C&& collection, int N) {
  assert(N > 0);
  C c = std::move(collection);
  std::mt19937 gen;
  std::uniform_int_distribution<int> dist(0, 100);
  for (int i = 0; i < N; ++i)
    c.push_back(dist(gen));

  std::sort(c.begin(), c.end());

  int last = c.front();
  for (auto it = c.begin() + 1; it != c.end(); ++it) {
    GALOIS_ASSERT(last <= *it, prefix);
    last = *it;
  }

  last = c.back();
  c.pop_back();
  for (; !c.empty(); c.pop_back()) {
    GALOIS_ASSERT(last >= c.back(), prefix);
    last = c.back();
  }
}

template<typename T, typename Iterator>
void timeAccess(std::string prefix, T&& x, Iterator first, Iterator last) {
  Galois::Timer t1, t2;
  t1.start();
  while (first != last) {
    x.emplace_back(*first++);
  }
  t1.stop();
  t2.start();
  for (auto ii = x.begin(), ei = x.end(); ii != ei; ++ii) {
    (*ii).val;
  }
  t2.stop();
  std::cout << prefix << " " << t1.get() << " " << t2.get() << "\n";
}

template<typename T>
void timeAccesses(std::string prefix, T&& x, int size) {
  for (int i = 0; i < 3; ++i)
    timeAccess(prefix, std::forward<T>(x), boost::counting_iterator<int>(0), boost::counting_iterator<int>(size));
}

struct element {
  volatile int val;
  element(int x): val(x) { }
};

int main(int argc, char** argv) {
  testBasic("Galois::FixedSizeRing", Galois::FixedSizeRing<int, 32>(), 32);
  testBasic("Galois::gdeque", Galois::gdeque<int>(), 32 * 32);
  testSort("Galois::FixedSizeRing", Galois::FixedSizeRing<int, 32>(), 32);
  //testSort("Galois::gdeque", Galois::gdeque<int>(), 32 * 32);

  int size = 100;
  if (argc > 1)
    size = atoi(argv[1]);
  if (size <= 0)
    size = 1000000;
  timeAccesses("std::deque", std::deque<element>(), size);
  timeAccesses("std::vector", std::vector<element>(), size);
  timeAccesses("Galois::gdeque", Galois::gdeque<element>(), size);
  timeAccesses("Galois::InsertBag", Galois::InsertBag<element>(), size);

  return 0;
}
