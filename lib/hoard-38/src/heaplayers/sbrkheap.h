/* -*- C++ -*- */

#ifndef _SBRKHEAP_H_
#define _SBRKHEAP_H_

#ifdef WIN32

// If we're using Windows, we'll need to link in sbrk.c,
// a replacement for sbrk().

extern "C" void * sbrk (size_t sz);

#endif

/*
 * @class SbrkHeap
 * @brief A source heap that is a thin wrapper over sbrk.
 *
 * As it stands, memory cannot be returned to sbrk().
 * This is not a significant limitation, since only memory
 * at the end of the break point can ever be returned anyway.
 */

class SbrkHeap {
public:
  SbrkHeap (void)
  {}

  inline void * malloc (size_t sz) {
    return sbrk(sz);
  }
  
  inline void free (void *) { }
  inline int remove (void *) { return 0; }
};



#endif

