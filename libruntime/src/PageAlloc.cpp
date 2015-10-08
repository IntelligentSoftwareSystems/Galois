/** Page Allocator Implementation -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galois is a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * Galois is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Galois.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @section Copyright
 *
 * Copyright (C) 2015, The University of Texas at Austin. All rights
 * reserved.
 *
 * @section Description
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#include "Galois/Runtime/Mem.h"
#include "Galois/Substrate/gio.h"
#include "Galois/Substrate/StaticInstance.h"

#include <unistd.h>
#include <map>
#include <vector>
#include <numeric>
#include <fstream>
#include <sstream>
#include <mutex>

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

size_t Galois::Runtime::pageSize;

namespace {
struct FreeNode {
  FreeNode* next;
};
 
typedef Galois::Substrate::PtrLock<FreeNode> HeadPtr;
typedef Galois::Substrate::CacheLineStorage<HeadPtr> HeadPtrStorage;

// Tracks pages allocated
struct PAState {
  std::vector<int> counts;
  std::map<void*, HeadPtr*> ownerMap;
  PAState() { 
    counts.resize(Galois::Substrate::getThreadPool().getMaxThreads(), 0);
  }
};

static Galois::Substrate::StaticInstance<PAState> PA;

#ifdef __linux__
typedef Galois::Substrate::SimpleLock AllocLock;
#else
typedef Galois::Substrate::DummyLock AllocLock;
#endif
static AllocLock allocLock;
static Galois::Substrate::SimpleLock dataLock;
static __thread HeadPtr* head = 0;

void* allocFromOS() {
  //linux mmap can introduce unbounded sleep!
  void* ptr = 0;
  {
    std::lock_guard<AllocLock> ll(allocLock);

#ifdef MAP_HUGETLB
    //First try huge
    ptr = mmap(0, Galois::Runtime::hugePageSize, _PROT, _MAP_HUGE_POP, -1, 0);
#endif

    //FIXME: improve failure case to ensure hugePageSize alignment
#ifdef MAP_POPULATE
    //Then try populate
    if (!ptr || ptr == MAP_FAILED)
      ptr = mmap(0, Galois::Runtime::hugePageSize, _PROT, _MAP_POP, -1, 0);
#endif
    //Then try normal
    if (!ptr || ptr == MAP_FAILED) {
      ptr = mmap(0, Galois::Runtime::hugePageSize, _PROT, _MAP_BASE, -1, 0);
    }
    if (!ptr || ptr == MAP_FAILED) {
      GALOIS_SYS_DIE("Out of Memory");
    }
  }
  
  //protect the tracking structures
  {
    std::lock_guard<Galois::Substrate::SimpleLock> ll(dataLock);
    HeadPtr*& h = head;
    if (!h) { //first allocation
      h = &((new HeadPtrStorage())->data);
    }
    PAState& p = *PA.get();
    p.ownerMap[ptr] = h;
    p.counts[Galois::Substrate::ThreadPool::getTID()] += 1;
    return ptr;
  }
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
        Galois::Substrate::gWarn("error parsing meminfo");
      break;
    }
    if (hugePageSizeKb * 1024 != Galois::Runtime::hugePageSize)
      Galois::Substrate::gWarn("System HugePageSize does not match compiled HugePageSize");
  }
#else
  void checkHuge() { }
#endif

public:
  PageSizeConf() {
#ifdef _POSIX_PAGESIZE
    Galois::Runtime::pageSize = _POSIX_PAGESIZE;
#else
    Galois::Runtime::pageSize = sysconf(_SC_PAGESIZE);
#endif
    checkHuge();
  }
};

} // end anon namespace

static PageSizeConf pageSizeConf;

void Galois::Runtime::pageInReadOnly(void* buf, size_t len, size_t stride) {
  volatile char* ptr = reinterpret_cast<volatile char*>(buf);
  for (size_t i = 0; i < len; i += stride)
    ptr[i];
}

void Galois::Runtime::pageIn(void* buf, size_t len, size_t stride) {
  volatile char* ptr = reinterpret_cast<volatile char*>(buf);
  for (size_t i = 0; i < len; i += stride)
    ptr[i] = 0;
}

void* Galois::Runtime::pageAlloc() {
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

void Galois::Runtime::pageFree(void* m) {
  dataLock.lock();
  HeadPtr* phead = PA.get()->ownerMap[m];
  dataLock.unlock();
  assert(phead);
  phead->lock();
  FreeNode* nh = reinterpret_cast<FreeNode*>(m);
  nh->next = phead->getValue();
  phead->unlock_and_set(nh);
}

void Galois::Runtime::pagePreAlloc(int numPages) {
  while (numPages--)
    Galois::Runtime::pageFree(allocFromOS());
}

int Galois::Runtime::numPageAllocTotal() {
  PAState& p = *PA.get();
  return std::accumulate(p.counts.begin(), p.counts.end(), 0);
}

int Galois::Runtime::numPageAllocForThread(unsigned tid) {
  return PA.get()->counts[tid];
}

void* Galois::Runtime::largeAlloc(size_t len, bool preFault) {
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

void Galois::Runtime::largeFree(void* m, size_t len) {
  size_t size = (len + hugePageSize - 1) & ~static_cast<size_t>(hugePageSize - 1);
  allocLock.lock();
  munmap(m, size);
  allocLock.unlock();
}
