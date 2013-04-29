/** Memory allocator implementation -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2012, The University of Texas at Austin. All rights reserved.
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

#ifdef GALOIS_USE_NUMA
#include <numa.h>
#endif

static int is_numa_available;
static const char* sNumaStat = "/proc/self/numa_maps";

void Galois::Runtime::MM::printInterleavedStats() {
  FILE* f = fopen(sNumaStat, "r");
  if (!f) {
    GALOIS_SYS_DIE("failed opening ", sNumaStat);
  }

  char line[2048];
  LL::gInfo("INTERLEAVED STATS BEGIN");
  while (fgets(line, 2048, f)) {
    // Chomp \n
    size_t len = strlen(line);
    if (len && line[len-1] == '\n')
      line[len-1] = '\0';

    char* start;
    if (strstr(line, "interleave") != 0) {
      LL::gInfo(line);
    } else if ((start = strstr(line, "anon=")) != 0) {
      int pages;
      if (sscanf(start, "anon=%d", &pages) == 1 && pages > 16*1024) {
        LL::gInfo(line);
      }
    } else if ((start = strstr(line, "mapped=")) != 0) {
      int pages;
      if (sscanf(start, "mapped=%d", &pages) == 1 && pages > 16*1024) {
        LL::gInfo(line);
      }
    }
  }
  LL::gInfo("INTERLEAVED STATS END");

  fclose(f);
}

#if defined(GALOIS_USE_NUMA)
static void *alloc_interleaved_subset(size_t len) {
  void* data = 0;
# if defined(GALOIS_USE_NUMA_OLD)
  nodemask_t nm = numa_no_nodes;
  unsigned int num = Galois::Runtime::activeThreads;
  int max_nodes = numa_max_node();
  for (unsigned y = 0; y < num; ++y) {
    unsigned proc = Galois::Runtime::LL::getProcessorForThread(y);
    // Assume block distribution from physical processors to numa nodes
    nodemask_set(&nm, proc/max_nodes);
  }
  data = numa_alloc_interleaved_subset(len, &nm);
# elif defined(GALOIS_USE_NUMA)  
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
#  error "Missing case"
# endif
  return data;
}
#endif

void* Galois::Runtime::MM::largeInterleavedAlloc(size_t len, bool full) {
  void* data = 0;
#if defined(GALOIS_USE_NUMA)
  if (is_numa_available == 0) {
    is_numa_available = numa_available() == -1 ? -1 : 1;
    if (is_numa_available == -1)
      LL::gWarn("NUMA not available");
  }
  if (is_numa_available == 1) {
    if (full) 
      data = numa_alloc_interleaved(len);
    else 
      data = alloc_interleaved_subset(len);
    // NB(ddn): Some strange bugs when empty interleaved mappings are
    // coalesced. Eagerly fault in interleaved pages to circumvent.
    memset(data, 0, len);
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
#if defined(GALOIS_USE_NUMA)
  if (is_numa_available == 1)
    numa_free(m, len);
  else
    largeFree(m, len);
#else
  largeFree(m, len);
#endif
}
