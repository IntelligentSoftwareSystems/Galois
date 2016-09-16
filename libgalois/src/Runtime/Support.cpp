/** Implementation for various Functions -*- C++ -*-
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
 * Copyright (C) 2016, The University of Texas at Austin. All rights
 * reserved.
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#include "Galois/Runtime/Executor_OnEach.h"
#include "Galois/Runtime/PagePool.h"
#include "Galois/Runtime/GaloisConfig.h"

void preAllocThreads(unsigned num, unsigned activeThreads) {
  using namespace Galois::Runtime;
  unsigned pagesPerThread = (num + activeThreads - 1) / activeThreads;
  on_each_impl(activeThreads, [pagesPerThread] (unsigned,unsigned) {pagePoolPreAlloc(pagesPerThread);}, "prealloc");
}
