// -*- C++ -*-

#ifndef _LAZYHEAP_H_
#define _LAZYHEAP_H_

template <class SuperHeap>
class LazyHeap {
public:

  LazyHeap (void)
    : initialized (0)
  {}

  ~LazyHeap (void) {
    if (initialized) {
      delete lazyheap;
    }
  }

  inline void * malloc (size_t sz) {
    return getHeap()->malloc (sz);
  }
  inline void free (void * ptr) {
    getHeap()->free (ptr);
  }
  inline void clear (void) {
    if (initialized) {
      getHeap()->clear();
    }
  }

private:

  SuperHeap * getHeap (void) {
    if (!initialized) {
      lazyheap = new SuperHeap;
      initialized = 1;
    }
    return lazyheap;
  }

  bool initialized;
  SuperHeap * lazyheap;
};


#endif
