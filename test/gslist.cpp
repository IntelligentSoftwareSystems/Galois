#include "galois/Galois.h"
#include "galois/gslist.h"
#include "galois/gIO.h"
#include "galois/runtime/Mem.h"
#include <map>

int main(int argc, char** argv) {
  typedef galois::runtime::FixedSizeHeap Heap;
  typedef std::unique_ptr<Heap> HeapPtr;
  typedef galois::substrate::PerThreadStorage<HeapPtr> Heaps;
  typedef galois::concurrent_gslist<int> Collection;
  int numThreads = 2;
  unsigned size = 100;
  if (argc > 1)
    numThreads = atoi(argv[1]);
  if (size <= 0)
    numThreads = 2;
  if (argc > 2)
    size = atoi(argv[2]);
  if (size <= 0)
    size = 10000;

  galois::setActiveThreads(numThreads);

  Heaps heaps;
  Collection c;

  galois::on_each([&](unsigned id, unsigned total) {
    HeapPtr& hp = *heaps.getLocal();
    hp = std::move(HeapPtr(new Heap(sizeof(Collection::block_type))));
    for (unsigned i = 0; i < size; ++i)
      c.push_front(*hp, i);
  });

  std::map<int, int> counter;
  for (auto i : c) {
    counter[i] += 1;
  }
  for (unsigned i = 0; i < size; ++i) {
    GALOIS_ASSERT(counter[i] == numThreads);
  }
  GALOIS_ASSERT(counter.size() == size);

  galois::on_each([&](unsigned id, unsigned total) {
    while (c.pop_front(Collection::promise_to_dealloc()))
      ;
  });

  return 0;
}
