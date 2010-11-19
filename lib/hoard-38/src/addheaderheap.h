// -*- C++ -*-

#ifndef _ADDHEADERHEAP_H_
#define _ADDHEADERHEAP_H_

#include "sassert.h"
#include "hldefines.h"

namespace Hoard {

/**
 * @class AddHeaderHeap
 */

template <class SuperblockType,
	  size_t SuperblockSize,
	  class SuperHeap>
class AddHeaderHeap {
private:

  HL::sassert<(((int) SuperHeap::Alignment) % SuperblockSize == 0)> verifySize1;
  HL::sassert<(((int) SuperHeap::Alignment) >= SuperblockSize)> verifySize2;

  SuperHeap theHeap;

public:

  enum { Alignment = 0 };

  MALLOC_FUNCTION INLINE void * malloc (size_t sz) {
    // Allocate extra space for the header,
    // put it at the front of the object,
    // and return a pointer to just past it.
    const size_t headerSize = sizeof(typename SuperblockType::Header);
    void * ptr = theHeap.malloc (sz + headerSize);
    if (ptr == NULL) {
      return NULL;
    }
    typename SuperblockType::Header * p
      = new (ptr) typename SuperblockType::Header (sz, sz);
    assert ((size_t) (p + 1) == (size_t) ptr + headerSize);
    return reinterpret_cast<void *>(p + 1);
  }

  INLINE static size_t getSize (void * ptr) {
    // Find the header (just before the pointer) and return the size
    // value stored there.
    typename SuperblockType::Header * p;
    p = reinterpret_cast<typename SuperblockType::Header *>(ptr);
    return (p - 1)->getSize (ptr);
  }

  INLINE void free (void * ptr) {
    // Find the header (just before the pointer) and free the whole object.
    typename SuperblockType::Header * p;
    p = reinterpret_cast<typename SuperblockType::Header *>(ptr);
    theHeap.free (reinterpret_cast<void *>(p - 1));
  }
};

}

#endif
