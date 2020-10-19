/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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
 */

#include "galois/substrate/PageAlloc.h"
#include "galois/substrate/SimpleLock.h"
#include "galois/gIO.h"

#include <mutex>

// figure this out dynamically
const size_t hugePageSize = 2 * 1024 * 1024;
// protect mmap, munmap since linux has issues
static galois::substrate::SimpleLock allocLock;

static void* trymmap(size_t size, int flag) {
  std::lock_guard<galois::substrate::SimpleLock> lg(allocLock);
  const int _PROT = PROT_READ | PROT_WRITE;
  void* ptr       = galois::mmap(0, size, _PROT, flag, -1, 0);
  if (ptr == MAP_FAILED)
    ptr = nullptr;
  return ptr;
}

static const int _MAP = _MAP_ANON | MAP_PRIVATE;
#ifdef MAP_POPULATE
static const int _MAP_POP   = MAP_POPULATE | _MAP;
static const bool doHandMap = false;
#else
static const int _MAP_POP      = _MAP;
static const bool doHandMap    = true;
#endif
#ifdef MAP_HUGETLB
static const int _MAP_HUGE_POP = MAP_HUGETLB | _MAP_POP;
static const int _MAP_HUGE     = MAP_HUGETLB | _MAP;
#else
static const int _MAP_HUGE_POP = _MAP_POP;
static const int _MAP_HUGE     = _MAP;
#endif

size_t galois::substrate::allocSize() { return hugePageSize; }

void* galois::substrate::allocPages(unsigned num, bool preFault) {
  if (num > 0) {
    void* ptr =
        trymmap(num * hugePageSize, preFault ? _MAP_HUGE_POP : _MAP_HUGE);
    if (!ptr) {
      ptr = trymmap(num * hugePageSize, preFault ? _MAP_POP : _MAP);
    }

    if (!ptr)
      GALOIS_SYS_DIE("Out of Memory");

    if (preFault && doHandMap)
      for (size_t x = 0; x < num * hugePageSize; x += 4096)
        static_cast<char*>(ptr)[x] = 0;

    return ptr;
  } else {
    return nullptr;
  }
}

void galois::substrate::freePages(void* ptr, unsigned num) {
  std::lock_guard<SimpleLock> lg(allocLock);
  if (munmap(ptr, num * hugePageSize) != 0)
    GALOIS_SYS_DIE("Unmap failed");
}

/*

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
        galois::substrate::gWarn("error parsing meminfo");
      break;
    }
    if (hugePageSizeKb * 1024 != galois::runtime::hugePageSize)
      galois::substrate::gWarn("System HugePageSize does not match compiled
HugePageSize");
  }
#else
  void checkHuge() { }
#endif

public:
  PageSizeConf() {
#ifdef _POSIX_PAGESIZE
    galois::runtime::pageSize = _POSIX_PAGESIZE;
#else
    galois::runtime::pageSize = sysconf(_SC_PAGESIZE);
#endif
    checkHuge();
  }
};
*/
