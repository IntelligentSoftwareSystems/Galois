/**
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of XYZ License (a copy is located in
 * LICENSE.txt at the top-level directory).
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
