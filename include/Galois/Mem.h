// User-visible allocators -*- C++ -*-
/*
Galois, a framework to exploit amorphous data-parallelism in irregular
programs.

Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS SOFTWARE
AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY, FITNESS FOR ANY
PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF PERFORMANCE, AND ANY
WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF DEALING OR USAGE OF TRADE.
NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH RESPECT TO THE USE OF THE
SOFTWARE OR DOCUMENTATION. Under no circumstances shall University be liable
for incidental, special, indirect, direct or consequential damages or loss of
profits, interruption of business, or related expenses which may arise from use
of Software or Documentation, including but not limited to those resulting from
defects in Software and/or Documentation, or loss or inaccuracy of data of any
kind.
*/

#ifndef GALOIS_MEM_H
#define GALOIS_MEM_H

#include "Galois/Runtime/mem.h"

namespace Galois {

typedef GaloisRuntime::MM::SimpleBumpPtrWithMallocFallback<GaloisRuntime::MM::FreeListHeap<GaloisRuntime::MM::SystemBaseAlloc> > IterAllocBaseTy;

typedef GaloisRuntime::MM::ExternRefGaloisAllocator<char, IterAllocBaseTy> PerIterAllocTy;

template<typename Ty>
class GAllocator : public GaloisRuntime::MM::ExternRefGaloisAllocator<Ty, GaloisRuntime::MM::MallocWrapper> { 
  typedef GaloisRuntime::MM::ExternRefGaloisAllocator<Ty, GaloisRuntime::MM::MallocWrapper> Super;
  GaloisRuntime::MM::MallocWrapper wrapper;
public:
  GAllocator(): Super(&wrapper) { }
};

template<typename Ty>
struct GFixedAllocator : public GaloisRuntime::MM::FSBGaloisAllocator<Ty> { };

}
#endif
