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
