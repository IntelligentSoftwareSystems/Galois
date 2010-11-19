#ifndef _THEAP_H_
#define _THEAP_H_

// A general threshold-based heap.

// Threshold layer.
// Once we have more than Threshold bytes on our freelist,
// return memory to the superheap.

// The class argument FH should be some freelist heap that subclasses NullHeap.

template <class SuperHeap, class FH, int Threshold>
class THeap : public SuperHeap {
public:
  THeap (void)
    : total (0)
  {}

  ~THeap (void)
  {}

  inline void * malloc (size_t sz) {
    void * ptr;
    ptr = fHeap.malloc (sz);
    if (ptr == NULL) {
      // We have no memory on our freelist.
      // Get it from the superheap.
      ptr = SuperHeap::malloc (sz);
    } else {
      total -= size(ptr);
      // printf ("total = %d\n", total);
    }
	assert (size(ptr) >= sz);
    return ptr;
  }

  inline void free (void * ptr) {
    if (total < Threshold) {
      // printf ("FREE TO FREELIST.\n");
      total += size(ptr);
      fHeap.free (ptr);
      //printf ("total = %d\n", total);
    } else {
      // Dump the freelist heap.
      void * p = fHeap.malloc (1);
      while (p != NULL) {
		SuperHeap::free (p);
		p = fHeap.malloc (1);
      }
      SuperHeap::free (ptr);
	  total = 0;
    }
  }

private:
  // Provide a free list that will return NULL if it is out of memory.
  FH fHeap;
  int total;
};


#endif
