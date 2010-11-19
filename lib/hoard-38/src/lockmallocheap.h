#ifndef _LOCKMALLOCHEAP_H_
#define _LOCKMALLOCHEAP_H_

#include "hldefines.h"


// Just lock malloc (unlike LockedHeap, which locks both malloc and
// free). Meant to be combined with something like RedirectFree, which will
// implement free.

namespace Hoard {

template <typename Heap>
class LockMallocHeap : public Heap {
public:
  MALLOC_FUNCTION INLINE void * malloc (size_t sz) {
    HL::Guard<Heap> l (*this);
    return Heap::malloc (sz);
  }
};

}

#endif
