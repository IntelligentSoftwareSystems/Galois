#include "galois/Galois.h"
#include "galois/Queue.h"
#include <iostream>
#include <vector>

void check(const galois::optional<int>& r, int exp) {
  if (r && *r == exp)
    ;
  else {
    std::cerr << "Expected " << exp << "\n";
    abort();
  }
}

void testSerial() {
  galois::ConcurrentSkipListMap<int, int> map;
  int v = 0;

  for (int i = 100; i >= 0; --i) {
    map.put(i, &v);
  }

  for (int i = 0; i <= 100; ++i) {
    check(map.pollFirstKey(), i);
  }
}

void testSerial1() {
  galois::ConcurrentSkipListMap<int, int> map;
  std::vector<int> keys;

  for (int i = 0; i < 100; ++i)
    keys.push_back(i);

  std::random_shuffle(keys.begin(), keys.end());

  for (int i = 0; i < 100; ++i)
    map.put(keys[i], &keys[i]);

  for (int i = 0; i < 100; ++i) {
    int* v = map.get(keys[i]);
    if (v != &keys[i]) {
      std::cerr << "Expected " << &keys[i] << " not " << v << " for key " << i
                << "\n";
      abort();
    }
  }
}

struct Process {
  galois::ConcurrentSkipListMap<int, int>* map;
  int dummy;
  Process(galois::ConcurrentSkipListMap<int, int>& m) : map(&m) {}
  Process() = default;

  template <typename Context>
  void operator()(int& item, Context& ctx) {
    map->put(item, &dummy);
  }
};

void testConcurrent() {
  const int top = 1000;
  galois::ConcurrentSkipListMap<int, int> map;
  std::vector<int> range;
  for (int i = top; i >= 0; --i)
    range.push_back(i);

  int numThreads = galois::setActiveThreads(2);
  if (numThreads < 2) {
    assert(0 && "Unable to run with multiple threads");
    abort();
  }

  galois::for_each(range.begin(), range.end(), Process(map));

  for (int i = 0; i <= top; ++i) {
    check(map.pollFirstKey(), i);
  }
}

int main() {
  testSerial();
  testSerial1();
  testConcurrent();
  return 0;
}
