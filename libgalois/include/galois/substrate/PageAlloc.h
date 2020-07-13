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

#ifndef GALOIS_SUBSTRATE_PAGEALLOC_H
#define GALOIS_SUBSTRATE_PAGEALLOC_H

#include <cstddef>

#include "galois/config.h"

#ifdef __linux__
#include <linux/mman.h>
#endif
#include <sys/mman.h>

#include <utility>
#ifdef HAVE_MMAP64
namespace galois {
template <typename... Args>
void* mmap(void* addr, Args... args) { // 0 -> nullptr
  return ::mmap64(addr, std::forward<Args>(args)...);
}
} // namespace galois
//! offset type for mmap
typedef off64_t offset_t;
#else
namespace galois {
template <typename... Args>
void* mmap(void* addr, Args... args) { // 0 -> nullptr
  return ::mmap(addr, std::forward<Args>(args)...);
}
} // namespace galois
//! offset type for mmap
typedef off_t offset_t;
#endif

// mmap flags
#if defined(MAP_ANONYMOUS)
static const int _MAP_ANON = MAP_ANONYMOUS;
#elif defined(MAP_ANON)
static const int _MAP_ANON = MAP_ANON;
#else
static_assert(0, "No Anonymous mapping");
#endif

namespace galois {
namespace substrate {

// size of pages
size_t allocSize();

// allocate contiguous pages, optionally faulting them in
void* allocPages(unsigned num, bool preFault);

// free page range
void freePages(void* ptr, unsigned num);

} // namespace substrate
} // namespace galois

#endif // GALOIS_SUBSTRATE_PAGEALLOC_H
