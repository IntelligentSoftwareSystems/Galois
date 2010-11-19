/* -*- C++ -*- */

#ifndef _PROFILEHEAP_H_
#define _PROFILEHEAP_H_


#include <cstdio>

extern int memRequested;
extern int maxMemRequested;

// Maintain & print memory usage info.
// Requires a superheap with the size() method (e.g., SizeHeap).

template <class super, int HeapNumber>
class ProfileHeap : public super {
public:
  
  ProfileHeap (void)
  {
    memRequested = 0;
    maxMemRequested = 0;
  }

  ~ProfileHeap (void)
  {
    if (maxMemRequested > 0) {
      stats();
    }
  }

  inline void * malloc (size_t sz) {
    void * ptr = super::malloc (sz);
    // Notice that we use the size reported by the allocator
    // for the object rather than the requested size.
    memRequested += super::getSize(ptr);
    if (memRequested > maxMemRequested) {
      maxMemRequested = memRequested;
    }
    return ptr;
  }
  
  inline void free (void * ptr) {
    memRequested -= super::getSize (ptr);
    super::free (ptr);
  }
  
private:
  void stats (void) {
    printf ("Heap: %d\n", HeapNumber);
    printf ("Max memory requested = %d\n", maxMemRequested);
    printf ("Memory still in use = %d\n", memRequested);
  }

  int memRequested;
  int maxMemRequested;

};


#endif
