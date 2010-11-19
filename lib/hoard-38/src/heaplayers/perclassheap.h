#ifndef _PERCLASSHEAP_H
#define _PERCLASSHEAP_H

#include <new>

/**
 * @class PerClassHeap
 * @brief Enable the use of one heap for all class memory allocation.
 * 
 * This class contains one instance of the SuperHeap argument.  The
 * example below shows how to make a subclass of Foo that uses a
 * FreelistHeap to manage its memory, overloading operators new and
 * delete.
 * 
 * <TT>
 *   class NewFoo : public Foo, PerClassHeap<FreelistHeap<mallocHeap> > {};
 * </TT>
 */

namespace HL {

template <class SuperHeap>
class PerClassHeap {
public:
  inline void * operator new (size_t sz) {
    return getHeap()->malloc (sz);
  }
  inline void operator delete (void * ptr) {
    getHeap()->free (ptr);
  }
  inline void * operator new[] (size_t sz) {
    return getHeap()->malloc (sz);
  }
  inline void operator delete[] (void * ptr) {
	  getHeap()->free (ptr);
  }
  // For some reason, g++ needs placement new to be overridden
  // as well, at least in conjunction with use of the STL.
  // Otherwise, this should be superfluous.
  inline void * operator new (size_t sz, void * p) { return p; }
  inline void * operator new[] (size_t sz, void * p) { return p; }

private:
  inline static SuperHeap * getHeap (void) {
    static SuperHeap theHeap;
    return &theHeap;
  }
};

}

#endif
