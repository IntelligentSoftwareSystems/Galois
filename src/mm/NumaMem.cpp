/** Memory allocator implementation -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
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
#include "Galois/config.h"
#include "Galois/Runtime/mm/Mem.h"
#include "Galois/Runtime/ll/gio.h"

#include <fstream>

#if defined(GALOIS_USE_NUMA) && !defined(GALOIS_FORCE_NO_NUMA)
#define USE_NUMA
#endif

#ifdef USE_NUMA
#include <numa.h>
#endif

#ifdef USE_NUMA
static int is_numa_available;
#endif

// TODO Remove dependency on USE_NUMA for non-interleaved functionality because
// libnuma dev is not widely available
void Galois::Runtime::MM::printInterleavedStats(int minPages) {
  std::ifstream f("/proc/self/numa_maps");

  if (!f) {
    LL::gInfo("No NUMA support");
    return;
  }

  char line[2048];
  LL::gInfo("INTERLEAVED STATS BEGIN");
  while (f.getline(line, sizeof(line)/sizeof(*line))) { 
    // Chomp \n
    size_t len = strlen(line);
    if (len && line[len-1] == '\n')
      line[len-1] = '\0';

    char* start;
    if (strstr(line, "interleave") != 0) {
      LL::gInfo(line);
    } else if ((start = strstr(line, "anon=")) != 0) {
      int pages;
      if (sscanf(start, "anon=%d", &pages) == 1 && pages >= minPages) {
        LL::gInfo(line);
      }
    } else if ((start = strstr(line, "mapped=")) != 0) {
      int pages;
      if (sscanf(start, "mapped=%d", &pages) == 1 && pages >= minPages) {
        LL::gInfo(line);
      }
    }
  }
  LL::gInfo("INTERLEAVED STATS END");
}

static int num_numa_pages_for(unsigned nodeid) {
  std::ifstream f("/proc/self/numa_maps");
  if (!f) {
    return 0;
  }

  char format[2048];
  char search[2048];
  int written;
  written = snprintf(format, sizeof(format)/sizeof(*format), "N%u=%%d", nodeid);
  assert((unsigned)written < sizeof(format)/sizeof(*format));
  written = snprintf(search, sizeof(search)/sizeof(*search), "N%u=", nodeid);
  assert((unsigned)written < sizeof(search)/sizeof(*search));

  char line[2048];
  int totalPages = 0;
  while (f.getline(line, sizeof(line)/sizeof(*line))) {
    char* start;
    if ((start = strstr(line, search)) != 0) {
      int pages;
      if (sscanf(start, format, &pages) == 1) {
        totalPages += pages;
      }
    }
  }

  return totalPages;
}

static int checkNuma() {
#ifdef USE_NUMA
  if (is_numa_available == 0) {
    is_numa_available = numa_available() == -1 ? -1 : 1;
    if (is_numa_available == -1)
      Galois::Runtime::LL::gWarn("NUMA not available");
  }
  return is_numa_available == 1;
#else
  return false;
#endif
}

int Galois::Runtime::MM::numNumaAllocForNode(unsigned nodeid) {
  return num_numa_pages_for(nodeid);
}

int Galois::Runtime::MM::numNumaNodes() {
  if (!checkNuma())
    return 1;
#ifdef USE_NUMA
  return numa_num_configured_nodes();
#else
  return 1;
#endif
}

#ifdef USE_NUMA
static void *allocInterleavedSubset(size_t len) {
  void* data = 0;
# if defined(GALOIS_USE_NUMA_OLD) && !defined(GALOIS_FORCE_NO_NUMA)
  nodemask_t nm = numa_no_nodes;
  unsigned int num = Galois::Runtime::activeThreads;
  int num_nodes = numa_num_configured_nodes();
  for (unsigned y = 0; y < num; ++y) {
    unsigned proc = Galois::Runtime::LL::getProcessorForThread(y);
    // Assume block distribution from physical processors to numa nodes
    nodemask_set(&nm, proc/num_nodes);
  }
  data = numa_alloc_interleaved_subset(len, &nm);
# elif defined(GALOIS_USE_NUMA) && !defined(GALOIS_FORCE_NO_NUMA)
  bitmask* nm = numa_allocate_nodemask();
  unsigned int num = Galois::Runtime::activeThreads;
  for (unsigned y = 0; y < num; ++y) {
    unsigned proc = Galois::Runtime::LL::getProcessorForThread(y);
    int node = numa_node_of_cpu(proc);
    numa_bitmask_setbit(nm, node);
  }
  data = numa_alloc_interleaved_subset(len, nm);
  numa_free_nodemask(nm);
# else
  data = Galois::Runtime::MM::largeAlloc(len);
# endif
  return data;
}
#endif

#ifdef USE_NUMA
static void countPages(void* data, size_t len) {
  union { void* as_vptr; char* as_cptr; uintptr_t as_uint; } d = { data };
  unsigned pageMask = Galois::Runtime::MM::pageSize - 1;
  size_t numPages = (len + Galois::Runtime::MM::pageSize - 1) / Galois::Runtime::MM::pageSize;

  if (d.as_uint & pageMask)
    GALOIS_DIE("not page aligned");

  std::vector<void*> pages;
  std::vector<int> status;
  pages.resize(numPages);
  status.resize(numPages);
  for (size_t i = 0; i < numPages; ++i)
    pages[i] = d.as_cptr + i * Galois::Runtime::MM::pageSize;

  int s;
  if ((s = numa_move_pages(0, numPages, pages.data(), NULL, status.data(), 0)) != 0)
    GALOIS_DIE("move_pages: ", strerror(s));

  std::array<size_t, 128> hist {};
  int i = 0;
  for (int s : status) {
    if (s < 0) {
      Galois::Runtime::LL::gInfo("unknown status[", i, "]: ", strerror(-s));
      //GALOIS_DIE("unknown status[", i, "]: ", strerror(-s));
    } else if (s >= 128) {
      GALOIS_DIE("numa node out of range[", i , "]: " , s);
    } else {
      hist[s] += 1;
    }
    i += 1;
  }
  for (unsigned i = 0; i < hist.size(); ++i) {
    if (hist[i])
      Galois::Runtime::LL::gInfo(i, ": ", hist[i]);
  }
}
#endif

void* Galois::Runtime::MM::largeInterleavedAlloc(size_t len, bool full) {
  void* data = 0;
#ifdef USE_NUMA
  if (checkNuma()) {
    if (full) 
      data = numa_alloc_interleaved(len);
    else 
      data = allocInterleavedSubset(len);
    // NB(ddn): Some strange bugs when empty interleaved mappings are
    // coalesced. Eagerly fault in interleaved pages to circumvent.
    pageIn(data, len);
    //countPages(data, len);
  } else {
    data = largeAlloc(len);
  }
#else
  data = largeAlloc(len);
#endif
  if (!data)
    abort();
  return data;
}

void Galois::Runtime::MM::largeInterleavedFree(void* m, size_t len) {
  if (!checkNuma())
    largeFree(m, len);
#ifdef USE_NUMA
  numa_free(m, len);
#else
  largeFree(m, len);
#endif
}
