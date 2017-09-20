/** One element per cache line -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a framework to exploit
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
 * Copyright (C) 2015, The University of Texas at Austin. All rights
 * reserved.
 *
 * @section Description
 *
 * This wrapper ensures the contents occupy its own cache line(s).
 *
 * @author Andrew Lenharth <andrew@lenharth.org>
 */

#ifndef GALOIS_SUBSTRATE_CACHELINESTORAGE_H
#define GALOIS_SUBSTRATE_CACHELINESTORAGE_H

#include "CompilerSpecific.h"

#include <utility>

namespace galois {
namespace substrate {

// Store an item with padding
template<typename T>
struct CacheLineStorage {
  alignas(GALOIS_CACHE_LINE_SIZE) T data;

  char buffer[GALOIS_CACHE_LINE_SIZE - (sizeof(T) % GALOIS_CACHE_LINE_SIZE)];
  //static_assert(sizeof(T) < GALOIS_CACHE_LINE_SIZE, "Too large a type");

  CacheLineStorage() :data() {}
  CacheLineStorage(const T& v) :data(v) {}

  template<typename A>
  explicit CacheLineStorage(A&& v) :data(std::forward<A>(v)) {}

  explicit operator T() { return data; }

  T& get() { return data; }
  template<typename V>
  CacheLineStorage& operator=(const V& v) { data = v; return *this; }
};

} // end namespace substrate
} // end namespace galois

#endif
