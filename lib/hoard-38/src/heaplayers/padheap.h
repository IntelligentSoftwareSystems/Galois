/* -*- C++ -*- */

#ifndef _PADHEAP_H_
#define _PADHEAP_H_

#include <assert.h>

// Pad object size requests.

template <int CacheLineSize, class SuperHeap>
class PadHeap : public SuperHeap {
public:

  inline void * malloc (size_t sz) {
    return SuperHeap::malloc (roundup(sz));
  }

private:

  inline size_t roundup (size_t sz) {
    // NB: CacheLineSize MUST be a power of two.
	// The assertion below checks this.
	assert ((CacheLineSize & (CacheLineSize-1)) == 0);
    size_t roundup = (sz + CacheLineSize - 1) & ~((int) CacheLineSize-1);
	assert (roundup >= sz);
    return roundup;
  }
};

#endif
