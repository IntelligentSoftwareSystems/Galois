/** Numa-aware Page Allocators -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a framework to exploit
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
 * Numa small(page) allocations
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#include "galois/Substrate/PageAlloc.h"
#include "galois/Substrate/SimpleLock.h"
#include "galois/gIO.h"

#include <mutex>

#ifdef __linux__
#include <linux/mman.h>
#endif
#include <sys/mman.h>

//figure this out dynamically
const size_t hugePageSize = 2*1024*1024;
//protect mmap, munmap since linux has issues
static galois::substrate::SimpleLock allocLock;


static void* trymmap(size_t size, int flag) {
  std::lock_guard<galois::substrate::SimpleLock> lg(allocLock);
  const int _PROT = PROT_READ | PROT_WRITE;
  void* ptr = mmap(0, size, _PROT, flag, -1, 0);
  if (ptr == MAP_FAILED)
    ptr = nullptr;
  return ptr;
}
  // mmap flags
#if defined(MAP_ANONYMOUS)
static const int _MAP_ANON = MAP_ANONYMOUS;
#elif defined(MAP_ANON)
static const int _MAP_ANON = MAP_ANON;
#else
static_assert(0, "No Anonymous mapping");
#endif

static const int _MAP = _MAP_ANON | MAP_PRIVATE;
#ifdef MAP_POPULATE
static const int _MAP_POP = MAP_POPULATE | _MAP;
static const bool doHandMap = false;
#else
static const int _MAP_POP = _MAP;
static const bool doHandMap = true;
#endif
#ifdef MAP_HUGETLB
static const int _MAP_HUGE_POP = MAP_HUGETLB | _MAP_POP;
static const int _MAP_HUGE = MAP_HUGETLB | _MAP;
#else
static const int _MAP_HUGE_POP = _MAP_POP;
static const int _MAP_HUGE = _MAP;
#endif


size_t galois::substrate::allocSize() {
  return hugePageSize;
}

void* galois::substrate::allocPages(unsigned num, bool preFault) {
  if (num > 0) {
    void* ptr = trymmap(num * hugePageSize, 
                        preFault ? _MAP_HUGE_POP : _MAP_HUGE);
    if (!ptr) {
      gWarn("Huge page alloc failed, falling back");
      ptr = trymmap(num*hugePageSize, preFault ? _MAP_POP : _MAP);
    }

    if (!ptr)
      GALOIS_SYS_DIE("Out of Memory");

    if (preFault && doHandMap)
      for (size_t x = 0; x < num*hugePageSize; x += 4096)
        static_cast<char*>(ptr)[x] = 0;

    return ptr;
  } else {
    return nullptr;
  }
}

void galois::substrate::freePages(void* ptr, unsigned num) {
  std::lock_guard<SimpleLock> lg(allocLock);
  if (munmap(ptr, num*hugePageSize) != 0)
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
      galois::substrate::gWarn("System HugePageSize does not match compiled HugePageSize");
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

