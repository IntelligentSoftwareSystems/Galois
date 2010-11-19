/* -*- C++ -*- */

#ifndef _PREFETCHHEAP_H_
#define _PREFETCHHEAP_H_

#define prefetcht0					  __asm _emit 0x0f __asm _emit 0x18 __asm _emit 0x08


template <class Super>
class PrefetchHeap : public Super {
public:
  
  inline void * malloc (size_t sz) {
    void * Address = Super::malloc (sz);
#ifdef _M_IX86
      // Prefetch this ptr before we return.
      __asm
	{
	  mov   eax,Address    // Load Address.
	  prefetcht0    // Prefetch into the L1.
	}
#endif
    return Address;
  }
  
  inline void free (void * ptr) {
    Super::free (ptr);
  }
};

#endif
