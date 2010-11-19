#ifndef _GLOBALHEAP_H_
#define _GLOBALHEAP_H_

#include "hoardsuperblock.h"
#include "processheap.h"

namespace Hoard {

template <size_t SuperblockSize,
	  int EmptinessClasses,
	  class LockType>
class GlobalHeap {
  
  class bogusThresholdFunctionClass {
  public:
    static inline bool function (int, int, size_t) {
      // We *never* cross the threshold for a process heap.
      return false;
    }
  };
  
public:

  GlobalHeap (void) 
    : _theHeap (getHeap())
    {
    }
  
  typedef ProcessHeap<SuperblockSize, EmptinessClasses, LockType, bogusThresholdFunctionClass> SuperHeap;
  typedef HoardSuperblock<LockType, SuperblockSize, GlobalHeap> SuperblockType;

  void put (void * s, size_t sz) {
    assert (s);
    assert (((SuperblockType *) s)->isValidSuperblock());
    _theHeap->put ((typename SuperHeap::SuperblockType *) s,
		  sz);
  }

  SuperblockType * get (size_t sz, void * dest) {
    SuperblockType * s = 
      reinterpret_cast<SuperblockType *>
      (_theHeap->get (sz, reinterpret_cast<SuperHeap *>(dest)));
    if (s) {
      assert (s->isValidSuperblock());
    }
    return s;
  }

private:

  SuperHeap * _theHeap;

  inline static SuperHeap * getHeap (void) {
    static double theHeapBuf[sizeof(SuperHeap) / sizeof(double) + 1];
    static SuperHeap * theHeap = new (&theHeapBuf[0]) SuperHeap;
    return theHeap;
  }

  // Prevent copying.
  GlobalHeap (const GlobalHeap&);
  GlobalHeap& operator=(const GlobalHeap&);

};

}

#endif
