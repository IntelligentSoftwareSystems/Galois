#include "Galois/Accumulator.h"
#include "Galois/Galois.h"
#include "Galois/Queue.h"
#include <iostream>
#include <vector>
#include <algorithm>

void check(const std::pair<bool,int>& r, int exp) {
  if (r.first && r.second == exp)
    ;
  else {
    std::cerr << "Expected " << exp << "got " << r.first << " " << r.second << "\n";
    abort();
  }
}

void testSerial() {
  Galois::PairingHeap<int> heap;

  for (int i = 0; i < 10; ++i) {
    if ((i & 0) == 0) {
      for (int j = 0; j < 10; ++j)
        heap.add(i * 10 + j);
    } else {
      for (int j = 9; j >= 0; --j)
        heap.add(i * 10 + j);
    }
  }

  for (int i = 0; i <= 99; ++i) {
    check(heap.pollMin(), i);
  }
}

void testParallel1() {
  Galois::FCPairingHeap<int> heap;

  for (int i = 0; i < 10; ++i) {
    if ((i & 0) == 0) {
      for (int j = 0; j < 10; ++j)
        heap.add(i * 10 + j);
    } else {
      for (int j = 9; j >= 0; --j)
        heap.add(i * 10 + j);
    }
  }

  for (int i = 0; i <= 99; ++i) {
    check(heap.pollMin(), i);
  }
}

struct Process2 {
  Galois::FCPairingHeap<int>& heap;
  Process2(Galois::FCPairingHeap<int>& h) : heap(h) { }

  template<typename Context>
  void operator()(int& item, Context ctx) {
    heap.add(item);
  }
};

void testParallel2() {
  const int top = 10000;
  std::vector<int> range;
  for (int i = 0; i < top; ++i)
    range.push_back(i);
  std::random_shuffle(range.begin(), range.end());

  int numThreads = Galois::setMaxThreads(2);
  if (numThreads < 2) {
    assert(0 && "Unable to run with multiple threads");
    abort();
  }

  Galois::FCPairingHeap<int> heap;
  Galois::for_each(range.begin(), range.end(), Process2(heap));

  for (int i = 0; i < top; ++i) {
    check(heap.pollMin(), i);
  }
}

struct Process3 {
  Galois::FCPairingHeap<int>& heap;
  Galois::GAccumulator<int>& acc;

  Process3(Galois::FCPairingHeap<int>& h, Galois::GAccumulator<int>& a) : heap(h), acc(a) { }

  template<typename Context>
  void operator()(int& item, Context ctx) {
    std::pair<bool,int> retval = heap.pollMin();
    if (retval.first) {
      acc += 1;
    } else {
      heap.add(item);
      ctx.push(item);
    }
  }
};

void testParallel3() {
  const int top = 10000;
  std::vector<int> range;
  for (int i = 0; i < top; ++i)
    range.push_back(i);
  std::random_shuffle(range.begin(), range.end());

  int numThreads = Galois::setMaxThreads(2);
  if (numThreads < 2) {
    assert(0 && "Unable to run with multiple threads");
    abort();
  }

  Galois::FCPairingHeap<int> heap;
  Galois::GAccumulator<int> acc;
  Galois::for_each(range.begin(), range.end(), Process3(heap, acc));

  if (acc.get() != top) {
    std::cerr << "Expected " << top << "got " << acc.get() << "\n";
    abort();
  }
}

int main() {
  testSerial();
  testParallel1();
  testParallel2();
  testParallel3();
  return 0;
}

