/** User-visible allocators -*- C++ -*-
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
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#ifndef GALOIS_MEM_H
#define GALOIS_MEM_H

#include "Galois/Runtime/Mem.h"

namespace Galois {

//! [PerIterAllocTy example]
//! Base allocator for per-iteration allocator
typedef Galois::Runtime::BumpWithMallocHeap<Galois::Runtime::FreeListHeap<Galois::Runtime::SystemHeap> > IterAllocBaseTy;

//! Per-iteration allocator that conforms to STL allocator interface
typedef Galois::Runtime::ExternalHeapAllocator<char, IterAllocBaseTy> PerIterAllocTy;
//! [PerIterAllocTy example]

//! Scalable fixed-sized allocator for T that conforms to STL allocator interface but
//! does not support variable sized allocations
template<typename Ty>
struct FixedSizeAllocator : public Galois::Runtime::FixedSizeAllocator<Ty> { };

}
#endif
