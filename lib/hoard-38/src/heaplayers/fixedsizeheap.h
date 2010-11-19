#ifndef _FIXEDSIZEHEAP_H_
#define _FIXEDSIZEHEAP_H_


template <class SH, int Size>
class FixedSizeHeap : public SH {
public:
  inline void * malloc (size_t sz) {
    assert (sz <= Size);
    //printf ("malloc\n");
    return SH::malloc (Size);
  }
  inline void free (void * ptr) {
    //printf ("free\n");
    SH::free (ptr);
  }
protected:
  inline static size_t size (void * p) {
    return Size;
  }
};

#endif
