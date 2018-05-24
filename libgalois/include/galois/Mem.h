#ifndef GALOIS_MEM_H
#define GALOIS_MEM_H

#include "galois/runtime/Mem.h"

namespace galois {

//! [PerIterAllocTy example]
//! Base allocator for per-iteration allocator
typedef galois::runtime::BumpWithMallocHeap<galois::runtime::FreeListHeap<galois::runtime::SystemHeap> > IterAllocBaseTy;

//! Per-iteration allocator that conforms to STL allocator interface
typedef galois::runtime::ExternalHeapAllocator<char, IterAllocBaseTy> PerIterAllocTy;
//! [PerIterAllocTy example]

//! Scalable fixed-sized allocator for T that conforms to STL allocator interface but
//! does not support variable sized allocations
template<typename Ty>
using FixedSizeAllocator = galois::runtime::FixedSizeAllocator<Ty>;

//! Scalable variable-sized allocator for T that allocates blocks of sizes in powers of 2
//! Useful for small and medium sized allocations, e.g. small or medium vectors, strings, deques
template<typename T>
using Pow_2_VarSizeAlloc = typename runtime::Pow_2_BlockAllocator<T>;

}
#endif
