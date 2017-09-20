#include "galois/Galois.h"
#include "galois/Queue.h"

#include <iostream>
#include <vector>
#include <algorithm>

void check(const char* func, bool r) {
  if (!r) {
    std::cerr << func << ": Expected true\n";
    abort();
  }
}

void check(const char* func, const galois::optional<int>& r, int exp) {
  if (r && *r == exp)
    return;
  else if (!r)
    std::cerr << func << ": Expected element\n";
  else 
    std::cerr << func << ": Expected " << exp << " got " << *r << "\n";
  abort();
}


void testPoll() {
  galois::PairingHeap<int> heap;

  for (int i = 0; i < 10; ++i) {
    if (i & 1) {
      for (int j = 0; j < 10; ++j)
        heap.add(i * 10 + j);
    } else {
      for (int j = 9; j >= 0; --j)
        heap.add(i * 10 + j);
    }
  }

  for (int i = 0; i < 100; ++i) {
    check(__func__, heap.pollMin(), i);
  }

  check(__func__, heap.empty());
}

void testDecrease() {
  galois::PairingHeap<int> heap;
  std::vector<galois::PairingHeap<int>::Handle> handles;

  for (int i = 0; i < 10; ++i) {
    if (i & 1) {
      for (int j = 0; j < 10; ++j)
        handles.push_back(heap.add(i * 10 + j));
    } else {
      for (int j = 9; j >= 0; --j)
        handles.push_back(heap.add(i * 10 + j));
    }
  }

  for (int i = 0; i < 100; ++i) {
    heap.decreaseKey(handles[i], heap.value(handles[i]) - 100);
  }

  for (int i = 0; i < 100; ++i) {
    check(__func__, heap.pollMin(), i - 100);
  }

  check(__func__, heap.empty());
}

void testDelete() {
  galois::PairingHeap<int> heap;
  std::vector<galois::PairingHeap<int>::Handle> handles;

  for (int i = 0; i < 100; ++i) {
    handles.push_back(heap.add(i));
  }

  for (int i = 0; i < 10; ++i)
    heap.deleteNode(handles[i * 10 + 5]);

  for (int i = 0; i < 100; ++i) {
    if ((i % 5) == 0 && (i & 1) == 1)
      continue;
    check(__func__, heap.pollMin(), i);
  }

  check(__func__, heap.empty());

}

void testParallel1() {
  galois::FCPairingHeap<int> heap;

  for (int i = 0; i < 10; ++i) {
    if ((i & 0) == 0) {
      for (int j = 0; j < 10; ++j)
        heap.add(i * 10 + j);
    } else {
      for (int j = 9; j >= 0; --j)
        heap.add(i * 10 + j);
    }
  }

  for (int i = 0; i < 100; ++i) {
    check(__func__, heap.pollMin(), i);
  }
}

struct Process2 {
  galois::FCPairingHeap<int>* heap;
  Process2(galois::FCPairingHeap<int>& h) : heap(&h) { }
  Process2() = default;

  template<typename Context>
  void operator()(int& item, Context& ctx) {
    heap->add(item);
  }
};

void testParallel2() {
  const int top = 10000;
  std::vector<int> range;
  for (int i = 0; i < top; ++i)
    range.push_back(i);
  std::random_shuffle(range.begin(), range.end());

  int numThreads = galois::setActiveThreads(2);
  if (numThreads < 2) {
    assert(0 && "Unable to run with multiple threads");
    abort();
  }

  galois::FCPairingHeap<int> heap;
  galois::for_each(range.begin(), range.end(), Process2(heap));

  for (int i = 0; i < top; ++i) {
    check(__func__, heap.pollMin(), i);
  }
}

int main() {
  testPoll();
  testDecrease();
  testDelete();
  testParallel1();
  testParallel2();
  return 0;
}

