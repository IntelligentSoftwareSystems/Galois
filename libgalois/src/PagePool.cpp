/** Page Pool  Implementation -*- C++ -*-
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
 */

#define __is_trivial(type)  __has_trivial_constructor(type) && __has_trivial_copy(type)

#include "Galois/Runtime/PagePool.h"


using namespace galois::Runtime;

static galois::Runtime::internal::PageAllocState<>* PA;

void galois::Runtime::internal::setPagePoolState(PageAllocState<>* pa) {
  GALOIS_ASSERT(!(PA && pa), "PagePool.cpp: Double Initialization of PageAllocState");
  PA = pa;
}

int galois::Runtime::numPagePoolAllocTotal() {
  return PA->countAll();
}

int galois::Runtime::numPagePoolAllocForThread(unsigned tid) {
  return PA->count(tid);
}

void* galois::Runtime::pagePoolAlloc() {
  return PA->pageAlloc();
}

void galois::Runtime::pagePoolPreAlloc(unsigned num) {
  while (num--)
    PA->pagePreAlloc();
}

void galois::Runtime::pagePoolFree(void* ptr) {
  PA->pageFree(ptr);
}

size_t galois::Runtime::pagePoolSize() {
  return Substrate::allocSize();
}

