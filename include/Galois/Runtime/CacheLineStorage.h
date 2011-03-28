// one element per cache line storage -*- C++ -*-

#ifndef __CACHE_LINE_STORAGE_H
#define __CACHE_LINE_STORAGE_H

namespace GaloisRuntime {

//xeons have 64 byte cache lines, but will prefetch 2 at a time
#define CACHE_LINE_SIZE 128

// Store an item with padding
template<typename T>
struct cache_line_storage {
  T data __attribute__((aligned(CACHE_LINE_SIZE)));
  char pad[ CACHE_LINE_SIZE % sizeof(T) ?
	    CACHE_LINE_SIZE - (sizeof(T) % CACHE_LINE_SIZE) :
	    0 ];
};

}

#endif
