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

#ifndef GALOIS_MEM_H
#define GALOIS_MEM_H

#include "galois/config.h"
#include "galois/runtime/Mem.h"

namespace galois {

//! [PerIterAllocTy example]
//! Base allocator for per-iteration allocator
typedef galois::runtime::BumpWithMallocHeap<
    galois::runtime::FreeListHeap<galois::runtime::SystemHeap>>
    IterAllocBaseTy;

//! Per-iteration allocator that conforms to STL allocator interface
typedef galois::runtime::ExternalHeapAllocator<char, IterAllocBaseTy>
    PerIterAllocTy;
//! [PerIterAllocTy example]

//! Scalable fixed-sized allocator for T that conforms to STL allocator
//! interface but does not support variable sized allocations
template <typename Ty>
using FixedSizeAllocator = galois::runtime::FixedSizeAllocator<Ty>;

//! Scalable variable-sized allocator for T that allocates blocks of sizes in
//! powers of 2 Useful for small and medium sized allocations, e.g. small or
//! medium vectors, strings, deques
template <typename T>
using Pow_2_VarSizeAlloc = typename runtime::Pow_2_BlockAllocator<T>;

} // namespace galois
#endif
