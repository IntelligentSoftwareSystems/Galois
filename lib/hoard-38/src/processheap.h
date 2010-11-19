// -*- C++ -*-

#ifndef _PROCESSHEAP_H_
#define _PROCESSHEAP_H_

#include <cstdlib>

#include "alignedsuperblockheap.h"
#include "conformantheap.h"
#include "emptyhoardmanager.h"
#include "hoardmanager.h"
#include "hoardsuperblock.h"

namespace Hoard {

  template <size_t SuperblockSize,
	    int EmptinessClasses,
	    class LockType,
	    class ThresholdClass>
  class ProcessHeap :
    public ConformantHeap<
    HoardManager<AlignedSuperblockHeap<LockType, SuperblockSize>,
		 EmptyHoardManager<HoardSuperblock<LockType, SuperblockSize, ProcessHeap<SuperblockSize, EmptinessClasses, LockType, ThresholdClass> > >,
		 HoardSuperblock<LockType, SuperblockSize, ProcessHeap<SuperblockSize, EmptinessClasses, LockType, ThresholdClass> >,
		 EmptinessClasses,
		 LockType,
		 ThresholdClass,
		 ProcessHeap<SuperblockSize, EmptinessClasses, LockType, ThresholdClass> > > {
  
public:
  
  ProcessHeap (void) {}
    
  // Disable allocation from this heap.
  inline void * malloc (size_t);

private:

  // Prevent copying or assignment.
  ProcessHeap (const ProcessHeap&);
  ProcessHeap& operator=(const ProcessHeap&);

};

}

#endif
