/* -*- C++ -*- */

#ifndef _ANSIWRAPPER_H_
#define _ANSIWRAPPER_H_

#include <string.h>

/*
 * @class ANSIWrapper
 * @brief Provide ANSI C behavior for malloc & free.
 *
 * Implements all prescribed ANSI behavior, including zero-sized
 * requests & aligned request sizes to a double word (or long word).
 */


namespace HL {

template <class SuperHeap>
class ANSIWrapper : public SuperHeap {
public:
  
  ANSIWrapper (void)
    {}

  inline void * malloc (size_t sz) {
    if (sz < 2 * sizeof(size_t)) {
      // Make sure it's at least big enough to hold two pointers (and
      // conveniently enough, that's at least the size of a double, as
      // required by ANSI).
      sz = 2 * sizeof(size_t);
    }
    sz = align(sz);
    void * ptr = SuperHeap::malloc (sz);
    return ptr;
  }
 
  inline void free (void * ptr) {
    if (ptr != 0) {
      SuperHeap::free (ptr);
    }
  }

  inline void * calloc (const size_t s1, const size_t s2) {
    char * ptr = (char *) malloc (s1 * s2);
    if (ptr) {
#if 1
      for (int i = 0; i < s1 * s2; i++) {
	ptr[i] = 0;
      }
#else
      #error "Uncomment below."
      //      memset (ptr, 0, s1 * s2);
#endif
    }
    return (void *) ptr;
  }
  
  inline void * realloc (void * ptr, const size_t sz) {
    if (ptr == 0) {
      return malloc (sz);
    }
    if (sz == 0) {
      free (ptr);
      return 0;
    }

    size_t objSize = getSize (ptr);
    if (objSize == sz) {
      return ptr;
    }

    // Allocate a new block of size sz.
    void * buf = malloc (sz);

    // Copy the contents of the original object
    // up to the size of the new block.

    size_t minSize = (objSize < sz) ? objSize : sz;
    if (buf) {
      memcpy (buf, ptr, minSize);
    }

    // Free the old block.
    free (ptr);
    return buf;
  }
  
  inline size_t getSize (void * ptr) {
    if (ptr) {
      return SuperHeap::getSize (ptr);
    } else {
      return 0;
    }
  }

private:
  inline static size_t align (size_t sz) {
    return (sz + (sizeof(double) - 1)) & ~(sizeof(double) - 1);
  }
};

}

#endif
