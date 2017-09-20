/** Memory allocator implementation -*- C++ -*-
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
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */

#include "Galois/Runtime/Support.h"
#include "Galois/Runtime/Mem.h"
#include "Galois/gIO.h"

#include <fstream>
#include <limits>
#include <vector>

#ifdef GALOIS_USE_NUMA
static int isNumaAvailable;
#endif

using namespace galois::substrate;

namespace galois {
namespace runtime {
extern unsigned activeThreads;
}
}

//! Define to get version of library that does not depend on Galois thread primatives
//#define GALOIS_FORCE_STANDALONE

void galois::runtime::printInterleavedStats(int minPages) {
  std::ifstream f("/proc/self/numa_maps");

  if (!f) {
    gInfo("No NUMA support");
    return;
  }

  char line[2048];
  gInfo("INTERLEAVED STATS BEGIN");
  while (f.getline(line, sizeof(line)/sizeof(*line))) { 
    // Chomp \n
    size_t len = strlen(line);
    if (len && line[len-1] == '\n')
      line[len-1] = '\0';

    char* start;
    if (strstr(line, "interleave") != 0) {
      gInfo(line);
    } else if ((start = strstr(line, "anon=")) != 0) {
      int pages;
      if (sscanf(start, "anon=%d", &pages) == 1 && pages >= minPages) {
        gInfo(line);
      }
    } else if ((start = strstr(line, "mapped=")) != 0) {
      int pages;
      if (sscanf(start, "mapped=%d", &pages) == 1 && pages >= minPages) {
        gInfo(line);
      }
    }
  }
  gInfo("INTERLEAVED STATS END");
}

static int numNumaPagesFor(unsigned nodeid) {
  std::ifstream f("/proc/self/numa_maps");
  if (!f) {
    return 0;
  }

  char format[2048];
  char search[2048];
  snprintf(format, sizeof(format), "N%u=%%d", nodeid);
  snprintf(search, sizeof(search), "N%u=", nodeid);

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

int galois::runtime::numNumaAllocForNode(unsigned nodeid) {
  return numNumaPagesFor(nodeid);
}

#ifdef GALOIS_USE_NUMA
static void *allocInterleaved(size_t len, unsigned num) {
  auto& tp = galois::substrate::getThreadPool();
  bitmask* nm = numa_allocate_nodemask();
  for (unsigned i = 0; i < num; ++i) {
    numa_bitmask_setbit(nm, tp.getOSNumaNode(i));
  }
  void* data = numa_alloc_interleaved_subset(len, nm);
  numa_free_nodemask(nm);
  // NB(ddn): Some strange bugs when empty interleaved mappings are
  // coalesced. Eagerly fault in interleaved pages to circumvent.
  if (data)
    galois::runtime::pageIn(data, len, galois::runtime::pageSize);
  return data;
}
#endif

#ifdef GALOIS_USE_NUMA
static bool checkIfInterleaved(void* data, size_t len, unsigned total) {
  // Assume small allocations are interleaved properly
  if (len < galois::runtime::hugePageSize * galois::runtime::numNumaNodes())
    return true;

  union { void* as_vptr; char* as_cptr; uintptr_t as_uint; } d = { data };
  size_t pageSize = galois::runtime::pageSize;
  int numNodes = galois::runtime::numNumaNodes();

  std::vector<size_t> hist(numNodes);
  for (size_t i = 0; i < len; i += pageSize) {
    int node;
    char *mem = d.as_cptr + i;
    if (get_mempolicy(&node, NULL, 0, mem, MPOL_F_NODE|MPOL_F_ADDR) < 0) {
      //galois::runtime::LL::gInfo("unknown status[", mem, "]: ", strerror(errno));
    } else {
      hist[node] += 1;
    }
  }

  size_t least = std::numeric_limits<size_t>::max();
  size_t greatest = std::numeric_limits<size_t>::min();
  for (unsigned i = 0; i < total; ++i) {
    int node = getNumaNode(i);
    least = std::min(least, hist[node]);
    greatest = std::max(greatest, hist[node]);
  }

  return !total || least / (double) greatest > 0.5;
}
#endif

#ifndef GALOIS_FORCE_STANDALONE
// Figure out which subset of threads will participate in pageInInterleaved
static void createMapping(std::vector<int>& mapping, unsigned& uniqueNodes) {
  std::vector<bool> hist(galois::runtime::numNumaNodes());
  uniqueNodes = 0;
  for (unsigned i = 0; i < mapping.size(); ++i) {
    int node = getNumaNode(i);
    if (hist[node])
      continue;
    hist[node] = true;
    uniqueNodes += 1;
    mapping[i] = node + 1;
  }
}
#endif

#ifndef GALOIS_FORCE_STANDALONE
static void pageInInterleaved(void* data, size_t len, std::vector<int>& mapping, unsigned numNodes) {
  // XXX Don't know whether memory is backed by hugepages or not, so stick with
  // smaller page size
  size_t blockSize = galois::runtime::pageSize;
#ifdef GALOIS_FORCE_STANDALONE
  unsigned tid = 0;
#else
  unsigned tid = ThreadPool::getTID();
#endif
  int id = mapping[tid] - 1;
  if (id < 0)
    return;
  size_t start = id * blockSize;
  size_t stride = numNodes * blockSize;
  if (len <= start)
    return;
  union { void* as_vptr; char* as_cptr; } d = { data };

  galois::runtime::pageIn(d.as_cptr + start, len - start, stride);
}
#endif

static inline bool isNumaAlloc(void* data, size_t len) {
  union { void* as_vptr; char* as_cptr; } d = { data };
  return d.as_cptr[len-1] != 0;
}

static inline void setNumaAlloc(void* data, size_t len, bool isNuma) {
  union { void* as_vptr; char* as_cptr; } d = { data };
  d.as_cptr[len-1] = isNuma;
}

void* galois::runtime::largeInterleavedAlloc(size_t len, bool full) {
  void* data;
#ifdef GALOIS_FORCE_STANDALONE
  unsigned __attribute__((unused)) total = 1;
  bool inForEach = false;
#else
  unsigned total = full ? getThreadPool().getMaxCores() : activeThreads;
  bool inForEach = substrate::getThreadPool().isRunning();
#endif
  bool numaAlloc = false;

  len += 1; // space for allocation metadata

  if (inForEach) {
    if (checkNuma()) {
#ifdef GALOIS_USE_NUMA
      data = allocInterleaved(len, total);
      numaAlloc = true;
#else
      data = largeAlloc(len, false);
#endif
    } else {
      data = largeAlloc(len, false);
    }
  } else {
    // DDN: Depend on first-touch policy to place memory rather than libnuma
    // calls because numa_alloc_interleaved seems to have issues properly
    // interleaving memory.
    data = largeAlloc(len, false);
#ifndef GALOIS_FORCE_STANDALONE
    unsigned uniqueNodes;
    std::vector<int> mapping(total);
    createMapping(mapping, uniqueNodes);
    substrate::getThreadPool().run(total, std::bind(pageInInterleaved, data, len, std::ref(mapping), uniqueNodes));
#endif
  }

  if (!data)
    abort();

  setNumaAlloc(data, len, numaAlloc);

#ifdef GALOIS_USE_NUMA
  // numa_alloc_interleaved sometimes fails to interleave pages
  if (numaAlloc && !checkIfInterleaved(data, len, total))
    gWarn("NUMA interleaving failed: ", data, " size: ", len);
#endif

  return data;
}

void galois::runtime::largeInterleavedFree(void* data, size_t len) {
  len += 1; // space for allocation metadata

#ifdef GALOIS_USE_NUMA
  if (isNumaAlloc(data, len)) {
    numa_free(data, len);
    return;
  }
#endif
  largeFree(data, len);
}
