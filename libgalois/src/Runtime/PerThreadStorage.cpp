/** Per Thread Storage -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galois is a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, version 2.1 of the
 * License.
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
 * @author Andrew Lenharth <andrew@lenharth.org>
 */

#include "Galois/Runtime/PerThreadStorage.h"
#include "Galois/Runtime/ErrorFeedBack.h"

#include <mutex>

thread_local uint64_t* Galois::Runtime::detail::PerThreadStorageBase::storage;

Galois::Runtime::detail::PerBackend Galois::Runtime::detail::PerThreadStorageBase::perBackend;

unsigned Galois::Runtime::detail::PerBackend::alloc(unsigned bytes) {
  std::lock_guard<SimpleLock> lg(lock);
  unsigned num = (bytes + sizeof(uint64_t) - 1) / sizeof(uint64_t);
  unsigned start = 0;
  unsigned count = num;
  for (unsigned x = 0; x < PTSSize && count; ++x) {
    if (mask[x]) { // reset
      count = num;
      start = x + 1;
    } else { // spot
      --count;
    }
  }
  if (count) // no match
    gDie("PerStorage backend exhasted");
  for (unsigned x = start; x < start + num; ++x)
    mask[x] = true;
  return start;
}

void Galois::Runtime::detail::PerBackend::dealloc(unsigned offset, unsigned bytes) {
  std::lock_guard<SimpleLock> lg(lock);
  unsigned b = offset;
  unsigned e = offset + (bytes + sizeof(uint64_t) - 1) / sizeof(uint64_t);
  while (b != e)
    mask[b++] = false;
}

void Galois::Runtime::detail::PerBackend::set(unsigned n, uint64_t* ptr) {
  std::lock_guard<SimpleLock> lg(lock);
  if (heads.size() <= n)
    heads.resize(n+1);
  heads.at(n).reset(ptr);
}

void Galois::Runtime::detail::PerThreadStorageBase::init_inner(unsigned maxThreads) {
  auto tmp =  new uint64_t[PerBackend::PTSSize];
  TRACE("Init PTS ", ThreadPool::getTID(), " ", tmp);
  perBackend.set(ThreadPool::getTID(), tmp);
  storage = tmp;
}

void Galois::Runtime::initPTS(unsigned max) {
  detail::PerThreadStorageBase::init_inner(max);
}

