/* -*- C++ -*- */

#ifndef _HYBRIDHEAP_H_
#define _HYBRIDHEAP_H_

#include <assert.h>
#include "sassert.h"
#include "hldefines.h"

/**
 * @class HybridHeap
 * Objects no bigger than BigSize are allocated and freed to SmallHeap.
 * Bigger objects are passed on to the super heap.
 */

namespace HL {

template <int BigSize, class SmallHeap, class BigHeap>
class HybridHeap : public SmallHeap {
public:

  HybridHeap (void)
  {
  }

  enum { Alignment = ((int) SmallHeap::Alignment < (int) BigHeap::Alignment) ?
                      (int) SmallHeap::Alignment :
	              (int) BigHeap::Alignment };

  MALLOC_FUNCTION INLINE void * malloc (size_t sz) {
    if (sz <= BigSize) {
      return SmallHeap::malloc (sz);
    } else {
      return slowPath (sz);
    }
  }
  
  inline void free (void * ptr) {
    if (SmallHeap::getSize(ptr) <= BigSize) {
      SmallHeap::free (ptr);
    } else {
      bm.free (ptr);
    }
  }
  
  inline void clear (void) {
    bm.clear();
    SmallHeap::clear();
  }
  

private:

  MALLOC_FUNCTION NO_INLINE
    void * slowPath (size_t sz) {
    return bm.malloc (sz);
  }


  HL::sassert<(BigSize > 0)> checkBigSizeNonZero;

  BigHeap bm;
};

}

#endif
