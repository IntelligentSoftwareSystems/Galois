// -*- C++ -*-

#ifndef _ALIGNEDSUPERBLOCKHEAP_H_
#define _ALIGNEDSUPERBLOCKHEAP_H_

#include "mmapheap.h"
#include "sassert.h"

#include "conformantheap.h"
#include "lockedheap.h"
#include "fixedrequestheap.h"
#include "dllist.h"

// Always requests aligned superblocks.
#include "alignedmmap.h"

namespace Hoard {

template <size_t SuperblockSize,
	  class TheLockType>
class SuperblockStore {
public:

  enum { Alignment = AlignedMmap<SuperblockSize, TheLockType>::Alignment };

  void * malloc (size_t sz) {
    sz = sz; // to avoid warning.
    assert (sz == SuperblockSize);
    if (_freeSuperblocks.isEmpty()) {
      // Get more memory.
      void * ptr = _superblockSource.malloc (ChunksToGrab * SuperblockSize);
      if (!ptr) {
	return NULL;
      }
      char * p = (char *) ptr;
      for (int i = 0; i < ChunksToGrab; i++) {
	_freeSuperblocks.insert ((DLList::Entry *) p);
	p += SuperblockSize;
      }
    }
    return _freeSuperblocks.get();
  }

  void free (void * ptr) {
    _freeSuperblocks.insert ((DLList::Entry *) ptr);
  }

private:

#if defined(__SVR4)
  enum { ChunksToGrab = 1 };

  // We only get 64K chunks from mmap on Solaris, so we need to grab
  // more chunks (and align them to 64K!) for smaller superblock sizes.
  // Right now, we do not handle this case and just assert here that
  // we are getting chunks of 64K.

  HL::sassert<(SuperblockSize == 65536)> PreventMmapFragmentation;
#else
  enum { ChunksToGrab = 1 };
#endif

  AlignedMmap<SuperblockSize, TheLockType> _superblockSource;
  DLList _freeSuperblocks;

};

}


namespace Hoard {

template <class TheLockType,
	  size_t SuperblockSize>
class AlignedSuperblockHeapHelper :
  public ConformantHeap<HL::LockedHeap<TheLockType,
				       FixedRequestHeap<SuperblockSize, 
							SuperblockStore<SuperblockSize, TheLockType> > > > {};


#if 0

template <class TheLockType,
	  size_t SuperblockSize>
class AlignedSuperblockHeap : public AlignedMmap<SuperblockSize,TheLockType> {};


#else

template <class TheLockType,
	  size_t SuperblockSize>
class AlignedSuperblockHeap :
  public AlignedSuperblockHeapHelper<TheLockType, SuperblockSize> {

  HL::sassert<(AlignedSuperblockHeapHelper<TheLockType, SuperblockSize>::Alignment % SuperblockSize == 0)> EnsureProperAlignment;

};
#endif

}

#endif
