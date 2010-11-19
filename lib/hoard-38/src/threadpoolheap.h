// -*- C++ -*-

#ifndef _THREADPOOLHEAP_H_
#define _THREADPOOLHEAP_H_

#include <cassert>

#include "array.h"
#include "cpuinfo.h"

extern volatile int anyThreadCreated;

namespace Hoard {

  template <int NumThreads,
	    int NumHeaps,
	    class PerThreadHeap_>
  class ThreadPoolHeap : public PerThreadHeap_ {
  public:
    
    typedef PerThreadHeap_ PerThreadHeap;
    
    enum { MaxThreads = NumThreads };
    enum { NumThreadsMask = NumThreads - 1};
    enum { NumHeapsMask = NumHeaps - 1};
    
    HL::sassert<((NumHeaps & NumHeapsMask) == 0)> verifyPowerOfTwoHeaps;
    HL::sassert<((NumThreads & NumThreadsMask) == 0)> verifyPowerOfTwoThreads;
    
    enum { MaxHeaps = NumHeaps };
    
    ThreadPoolHeap (void)
    {
      // Note: The tidmap values should be set externally.
      int j = 0;
      for (int i = 0; i < NumThreads; i++) {
	setTidMap(i, j % NumHeaps);
	j++;
      }
    }
    
    inline PerThreadHeap& getHeap (void) {
      int tid;
      if (anyThreadCreated) {
	tid = HL::CPUInfo::getThreadId();
      } else {
	tid = 0;
      }
      int heapno = _tidMap(tid & NumThreadsMask);
      return _heap(heapno);
    }
    
    inline void * malloc (size_t sz) {
      return getHeap().malloc (sz);
    }
    
    inline void free (void * ptr) {
      getHeap().free (ptr);
    }
    
    inline void clear (void) {
      getHeap().clear();
    }
    
    inline size_t getSize (void * ptr) {
      return PerThreadHeap::getSize (ptr);
    }
    
    void setTidMap (int index, int value) {
      assert ((value >= 0) && (value < MaxHeaps));
      _tidMap(index) = value;
    }
    
    int getTidMap (int index) const {
      return _tidMap(index); 
    }
    
    void setInusemap (int index, int value) {
      _inUseMap(index) = value;
    }
    
    int getInusemap (int index) const {
      return _inUseMap(index);
    }
    
    
  private:
    
    /// Which heap is assigned to which thread, indexed by thread.
    Array<MaxThreads, int> _tidMap;
    
    /// Which heap is in use (a reference count).
    Array<MaxHeaps, int> _inUseMap;
    
    /// The array of heaps we choose from.
    Array<MaxHeaps, PerThreadHeap> _heap;
    
  };
  
}

#endif
