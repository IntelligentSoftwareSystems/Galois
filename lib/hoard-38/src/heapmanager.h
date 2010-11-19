// -*- C++ -*-

#ifndef _HEAPMANAGER_H_
#define _HEAPMANAGER_H_

#include <cstdlib>

#include "guard.h"
#include "cpuinfo.h"

namespace Hoard {

  template <typename LockType,
	    typename HeapType>
  class HeapManager : public HeapType {
  public:

    HeapManager (void)
    {
      HL::Guard<LockType> g (heapLock);
      
      /// Initialize all heap maps (nothing yet assigned).
      int i;
      for (i = 0; i < HeapType::MaxThreads; i++) {
	HeapType::setTidMap (i, 0);
      }
      for (i = 0; i < HeapType::MaxHeaps; i++) {
	HeapType::setInusemap (i, 0);
      }
    }

    /// Set this thread's heap id to 0.
    void chooseZero (void) {
      HL::Guard<LockType> g (heapLock);
      HeapType::setTidMap ((int) HL::CPUInfo::getThreadId() % MaxThreads, 0);
    }

    int findUnusedHeap (void) {

      HL::Guard<LockType> g (heapLock);
      
      unsigned int tid_original = HL::CPUInfo::getThreadId();
      unsigned int tid = tid_original % HeapType::MaxThreads;
      
      int i = 0;
      while ((i < HeapType::MaxHeaps) && (HeapType::getInusemap(i)))
	i++;
      if (i >= HeapType::MaxHeaps) {
	// Every heap is in use: pick heap one.
	i = 0;
      }

      HeapType::setInusemap (i, 1);
      HeapType::setTidMap (tid, i);
      
      return i;
    }

    void releaseHeap (void) {
      // Decrement the ref-count on the current heap.
      
      HL::Guard<LockType> g (heapLock);
      
      // Statically ensure that the number of threads is a power of two.
      enum { VerifyPowerOfTwo = 1 / ((HeapType::MaxThreads & ~(HeapType::MaxThreads-1))) };
      
      int tid = HL::CPUInfo::getThreadId() & (HeapType::MaxThreads - 1);
      int heapIndex = HeapType::getTidMap (tid);
      
      HeapType::setInusemap (heapIndex, 0);
      
      // Prevent underruns (defensive programming).
      
      if (HeapType::getInusemap (heapIndex) < 0) {
	HeapType::setInusemap (heapIndex, 0);
      }
    }
    
    
  private:
    
    // Disable copying.
    
    HeapManager (const HeapManager&);
    HeapManager& operator= (const HeapManager&);
    
    /// The lock, to ensure mutual exclusion.
    LockType heapLock;
  };

}

#endif
