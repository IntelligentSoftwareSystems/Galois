/** Page Allocator Implementation -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2014, The University of Texas at Austin. All rights reserved.
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
 *
 * @section Description
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#include "Galois/Runtime/mm/Mem.h"
#include "Galois/Runtime/ll/gio.h"
#include "Galois/Runtime/ll/StaticInstance.h"

#include <unistd.h>
#include <map>
#include <vector>
#include <numeric>
#include <fstream>
#include <sstream>

#ifdef __linux__
#include <linux/mman.h>
#endif
#include <sys/mman.h>

// mmap flags
#if defined(MAP_ANONYMOUS)
static const int _MAP_ANON = MAP_ANONYMOUS;
#elif defined(MAP_ANON)
static const int _MAP_ANON = MAP_ANON;
#else
// fail
#endif
static const int _PROT = PROT_READ | PROT_WRITE;
static const int _MAP_BASE = _MAP_ANON | MAP_PRIVATE;
#ifdef MAP_POPULATE
static const int _MAP_POP  = MAP_POPULATE | _MAP_BASE;
#endif
#ifdef MAP_HUGETLB
static const int _MAP_HUGE_POP = MAP_HUGETLB | _MAP_POP;
static const int _MAP_HUGE = MAP_HUGETLB;
#endif

size_t Galois::Runtime::MM::pageSize;

namespace {
struct FreeNode {
  FreeNode* next;
};
 
typedef Galois::Runtime::LL::PtrLock<FreeNode, true> HeadPtr;
typedef Galois::Runtime::LL::CacheLineStorage<HeadPtr> HeadPtrStorage;

// Tracks pages allocated
struct PAState {
  std::vector<int> counts;
  std::map<void*, HeadPtr*> ownerMap;
  PAState() { 
    counts.resize(Galois::Runtime::LL::getMaxThreads(), 0);
  }
};

static Galois::Runtime::LL::StaticInstance<PAState> PA;

#ifdef __linux__
  static Galois::Runtime::LL::SimpleLock allocLock;
#else
  static Galois::Runtime::LL::DummyLock allocLock;
#endif
static Galois::Runtime::LL::SimpleLock dataLock;
static __thread HeadPtr* head = 0;

void* allocFromOS() {
  //linux mmap can introduce unbounded sleep!
  allocLock.lock();

  void* ptr = 0;
#ifdef MAP_HUGETLB
  //First try huge
  ptr = mmap(0, Galois::Runtime::MM::hugePageSize, _PROT, _MAP_HUGE_POP, -1, 0);
#endif

  //FIXME: improve failure case to ensure hugePageSize alignment
#ifdef MAP_POPULATE
  //Then try populate
  if (!ptr || ptr == MAP_FAILED)
    ptr = mmap(0, Galois::Runtime::MM::hugePageSize, _PROT, _MAP_POP, -1, 0);
#endif
  //Then try normal
  if (!ptr || ptr == MAP_FAILED) {
    ptr = mmap(0, Galois::Runtime::MM::hugePageSize, _PROT, _MAP_BASE, -1, 0);
  }
  
  allocLock.unlock();
  if (!ptr || ptr == MAP_FAILED) {
    GALOIS_SYS_DIE("Out of Memory");
  }

  //protect the tracking structures
  dataLock.lock();
  HeadPtr*& h = head;
  if (!h) { //first allocation
    h = &((new HeadPtrStorage())->data);
  }
  PAState& p = *PA.get();
  p.ownerMap[ptr] = h;
  p.counts[Galois::Runtime::LL::getTID()] += 1;
  dataLock.unlock();
  return ptr;
}

class PageSizeConf {
#ifdef MAP_HUGETLB
  void checkHuge() {
    std::ifstream f("/proc/meminfo");

    if (!f) 
      return;

    char line[2048];
    size_t hugePageSizeKb = 0;
    while (f.getline(line, sizeof(line)/sizeof(*line))) {
      if (strstr(line, "Hugepagesize:") != line)
        continue;
      std::stringstream ss(line + strlen("Hugepagesize:"));
      std::string kb;
      ss >> hugePageSizeKb >> kb;
      if (kb != "kB")
        Galois::Runtime::LL::gWarn("error parsing meminfo");
      break;
    }
    if (hugePageSizeKb * 1024 != Galois::Runtime::MM::hugePageSize)
      Galois::Runtime::LL::gWarn("System HugePageSize does not match compiled HugePageSize");
  }
#else
  void checkHuge() { }
#endif

public:
  PageSizeConf() {
#ifdef _POSIX_PAGESIZE
    Galois::Runtime::MM::pageSize = _POSIX_PAGESIZE;
#else
    Galois::Runtime::MM::pageSize = sysconf(_SC_PAGESIZE);
#endif
    checkHuge();
  }
};

} // end anon namespace

static PageSizeConf pageSizeConf;

void Galois::Runtime::MM::pageInReadOnly(void* buf, size_t len, size_t stride) {
  volatile char* ptr = reinterpret_cast<volatile char*>(buf);
  for (size_t i = 0; i < len; i += stride)
    ptr[i];
}

void Galois::Runtime::MM::pageIn(void* buf, size_t len, size_t stride) {
  volatile char* ptr = reinterpret_cast<volatile char*>(buf);
  for (size_t i = 0; i < len; i += stride)
    ptr[i] = 0;
}

void* Galois::Runtime::MM::pageAlloc() {
  HeadPtr* phead = head;
  if (phead) {
    phead->lock();
    FreeNode* h = phead->getValue();
    if (h) {
      phead->unlock_and_set(h->next);
      return h;
    }
    phead->unlock();
  }
  return allocFromOS();
}

void Galois::Runtime::MM::pageFree(void* m) {
  dataLock.lock();
  HeadPtr* phead = PA.get()->ownerMap[m];
  dataLock.unlock();
  assert(phead);
  phead->lock();
  FreeNode* nh = reinterpret_cast<FreeNode*>(m);
  nh->next = phead->getValue();
  phead->unlock_and_set(nh);
}

void Galois::Runtime::MM::pagePreAlloc(int numPages) {
  while (numPages--)
    Galois::Runtime::MM::pageFree(allocFromOS());
}

int Galois::Runtime::MM::numPageAllocTotal() {
  PAState& p = *PA.get();
  return std::accumulate(p.counts.begin(), p.counts.end(), 0);
}

int Galois::Runtime::MM::numPageAllocForThread(unsigned tid) {
  return PA.get()->counts[tid];
}

void* Galois::Runtime::MM::largeAlloc(size_t len, bool preFault) {
  size_t size = (len + hugePageSize - 1) & ~static_cast<size_t>(hugePageSize - 1);
  void * ptr = 0;

  allocLock.lock();
#ifdef MAP_HUGETLB
  ptr = mmap(0, size, _PROT, preFault ? _MAP_HUGE_POP : _MAP_HUGE, -1, 0);
# ifndef MAP_POPULATE
  if (ptr != MAP_FAILED && ptr && preFault) {
    pageIn(ptr, size, hugePageSize);
  }
# endif
#endif
#ifdef MAP_POPULATE
  if (preFault && (!ptr || ptr == MAP_FAILED))
    ptr = mmap(0, size, _PROT, _MAP_POP, -1, 0);
#endif
  if (!ptr || ptr == MAP_FAILED) {
    ptr = mmap(0, size, _PROT, _MAP_BASE, -1, 0);
    if (ptr != MAP_FAILED && ptr && preFault) {
      pageIn(ptr, size, pageSize);
    }
  }
  allocLock.unlock();

  if (!ptr || ptr == MAP_FAILED)
    GALOIS_SYS_DIE("Out of Memory");
  return ptr;
}

void Galois::Runtime::MM::largeFree(void* m, size_t len) {
  size_t size = (len + hugePageSize - 1) & ~static_cast<size_t>(hugePageSize - 1);
  allocLock.lock();
  munmap(m, size);
  allocLock.unlock();
}
