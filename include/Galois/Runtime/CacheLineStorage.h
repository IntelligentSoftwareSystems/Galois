// one element per cache line storage -*- C++ -*-
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

#ifndef _GALOIS_RUNTIME_CACHELINESTORAGE_H
#define _GALOIS_RUNTIME_CACHELINESTORAGE_H

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
  cache_line_storage() :data() {}
  explicit cache_line_storage(const T& v) :data(v) {}
};

}

#endif //_GALOIS_RUNTIME_CACHELINESTORAGE_H
