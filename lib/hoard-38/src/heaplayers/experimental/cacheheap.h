/* -*- C++ -*- */

#ifndef _CACHEHEAP_H_
#define _CACHEHEAP_H_

template <class Super>
class CacheHeap : public Super {
public:
  
  inline void * malloc (size_t sz) {
    void * ptr = mySuper.malloc (sz);
    return ptr;
  }
  
  inline void free (void * ptr) {
    // Insert checks here!
    mySuper.free (ptr);
  }
  
private:

  Super mySuper;
};


#endif
