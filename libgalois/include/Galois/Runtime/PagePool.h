/** heap building blocks -*- C++ -*-
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
 * Strongly inspired by heap layers:
 *  http://www.heaplayers.org/
 * FSB is modified from:
 *  http://warp.povusers.org/FSBAllocator/
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#ifndef GALOIS_RUNTIME_PAGEPOOL_H
#define GALOIS_RUNTIME_PAGEPOOL_H

#include <cstddef>

namespace Galois {
namespace Runtime {

//! Low level page pool (individual pages, use largeMalloc for large blocks)

void* pagePoolAlloc();
void pagePoolFree(void*);
void pagePoolPreAlloc(unsigned);

//Size of returned pages
size_t pagePoolSize();

//! Returns total large pages allocated by Galois memory management subsystem
int numPagePoolAllocTotal();
//! Returns total large pages allocated for thread by Galois memory management subsystem
int numPagePoolAllocForThread(unsigned tid);

namespace internal {
  //! Initialize PagePool, used by Runtime::init();
  void initPagePool(void);

  //! Destroy  PagePool, used by Runtime::finish();
  void killPagePool(void);
} // end namespace internal

} // end namespace Runtime
} // end namespace Galois

#endif
