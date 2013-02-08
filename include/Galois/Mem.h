/** User-visible allocators -*- C++ -*-
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
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#ifndef GALOIS_MEM_H
#define GALOIS_MEM_H

#include "Galois/Runtime/mm/Mem.h"

namespace Galois {

//! Base allocator for per-iteration allocator
typedef Galois::Runtime::MM::SimpleBumpPtrWithMallocFallback<Galois::Runtime::MM::FreeListHeap<Galois::Runtime::MM::SystemBaseAlloc> > IterAllocBaseTy;

//! Per-iteration allocator that conforms to STL allocator interface
typedef Galois::Runtime::MM::ExternRefGaloisAllocator<char, IterAllocBaseTy> PerIterAllocTy;

//! Scalable fixed-sized allocator for T that conforms to STL allocator interface but
//! does not support variable sized allocations
template<typename Ty>
struct GFixedAllocator : public Galois::Runtime::MM::FSBGaloisAllocator<Ty> { };

}
#endif
