#include "Galois/Runtime/mm/mem.h"
#include "Galois/Runtime/Support.h"

#include <map>

#ifdef __linux__
#include <linux/mman.h>
#endif
#include <sys/mman.h>

using namespace GaloisRuntime;
using namespace MM;

static const int _PROT = PROT_READ | PROT_WRITE;
static const int _MAP_BASE = MAP_ANONYMOUS | MAP_PRIVATE;
#ifdef MAP_POPULATE
static const int _MAP_POP  = MAP_POPULATE | _MAP_BASE;
#endif
#ifdef MAP_HUGETLB
static const int _MAP_HUGE = MAP_HUGETLB | _MAP_POP;
#endif

mmapWrapper::mmapWrapper() {
#ifndef MAP_POPULATE
  reportWarning("No MAP_POPULATE");
#endif
#ifndef MAP_HUGETLB
  reportWarning("No MAP_HUGETLB");
#endif
}

void* mmapWrapper::_alloc() {
  void* ptr = 0;
#ifdef MAP_HUGETLB
  //First try huge
  ptr = mmap(0, AllocSize, _PROT, _MAP_HUGE, -1, 0);
#endif
#ifdef MAP_POPULATE
  //Then try populate
  if (!ptr || ptr == MAP_FAILED)
    ptr = mmap(0, AllocSize, _PROT, _MAP_POP, -1, 0);
#endif
  //Then try normal
  if (!ptr || ptr == MAP_FAILED)
    ptr = mmap(0, AllocSize, _PROT, _MAP_BASE, -1, 0);
  if (!ptr || ptr == MAP_FAILED) {
    reportWarning("Memory Allocation Failed");
    assert(0 && "mmap failed");
    abort();
  }    
  return ptr;
}

void mmapWrapper::_free(void* ptr) {
  munmap(ptr, AllocSize);
}

//Anchor the class
SelfLockFreeListHeap<mmapWrapper> SystemBaseAlloc::Source;
SystemBaseAlloc::SystemBaseAlloc() {}
SystemBaseAlloc::~SystemBaseAlloc() {}

#ifndef USEMALLOC

SizedAlloc* GaloisRuntime::MM::getAllocatorForSize(unsigned int size) {
  static std::map<unsigned int, SizedAlloc*> allocators;
  static SimpleLock<int, true> Lock;

  Lock.lock();
  SizedAlloc*& retval = allocators[size];
  if (!retval)
    retval = new SizedAlloc;
  Lock.unlock();

  return retval;
}

#endif
