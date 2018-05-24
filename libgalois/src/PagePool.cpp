#define __is_trivial(type)  __has_trivial_constructor(type) && __has_trivial_copy(type)

#include "galois/runtime/PagePool.h"


using namespace galois::runtime;

static galois::runtime::internal::PageAllocState<>* PA;

void galois::runtime::internal::setPagePoolState(PageAllocState<>* pa) {
  GALOIS_ASSERT(!(PA && pa), "PagePool.cpp: Double Initialization of PageAllocState");
  PA = pa;
}

int galois::runtime::numPagePoolAllocTotal() {
  return PA->countAll();
}

int galois::runtime::numPagePoolAllocForThread(unsigned tid) {
  return PA->count(tid);
}

void* galois::runtime::pagePoolAlloc() {
  return PA->pageAlloc();
}

void galois::runtime::pagePoolPreAlloc(unsigned num) {
  while (num--)
    PA->pagePreAlloc();
}

void galois::runtime::pagePoolFree(void* ptr) {
  PA->pageFree(ptr);
}

size_t galois::runtime::pagePoolSize() {
  return substrate::allocSize();
}

