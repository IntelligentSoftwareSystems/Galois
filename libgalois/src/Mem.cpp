#include "galois/runtime/Mem.h"

#include <map>
#include <mutex>

using namespace galois::runtime;

//Anchor the class
SystemHeap::SystemHeap() {
  assert(AllocSize == runtime::pagePoolSize());
}

SystemHeap::~SystemHeap() {}

#ifndef GALOIS_FORCE_STANDALONE
__thread SizedHeapFactory::HeapMap* SizedHeapFactory::localHeaps = 0;

SizedHeapFactory::SizedHeap* 
SizedHeapFactory::getHeapForSize(const size_t size) {
  if (size == 0)
    return 0;
  return Base::getInstance()->getHeap(size);
}

SizedHeapFactory::SizedHeap* 
SizedHeapFactory::getHeap(const size_t size) {
  typedef SizedHeapFactory::HeapMap HeapMap;

  if (!localHeaps) {
    std::lock_guard<galois::substrate::SimpleLock> ll(lock);
    localHeaps = new HeapMap;
    allLocalHeaps.push_front(localHeaps);
  }

  auto& lentry = (*localHeaps)[size];
  if (lentry)
    return lentry;

  {
    std::lock_guard<galois::substrate::SimpleLock> ll(lock);
    auto& gentry = heaps[size];
    if (!gentry)
      gentry = new SizedHeap();
    lentry = gentry;
    return lentry;
  }
}


Pow_2_BlockHeap::Pow_2_BlockHeap (void) throw (): heapTable () {
  populateTable ();
}


SizedHeapFactory::SizedHeapFactory() :lock() {}

SizedHeapFactory::~SizedHeapFactory() {
  // TODO destructor ordering problem: there may be pointers to deleted
  // SizedHeap when this Factory is destroyed before dependent
  // FixedSizeHeaps.
  for (auto entry : heaps)
    delete entry.second;
  for (auto mptr : allLocalHeaps)
    delete mptr;
}
#endif
