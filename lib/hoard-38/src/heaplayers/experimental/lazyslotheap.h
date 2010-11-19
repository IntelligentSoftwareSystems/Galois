/* -*- C++ -*- */

#ifndef _LAZYSLOTHEAP_H_
#define _LAZYSLOTHEAP_H_

/*
  This heap manages memory in units of Chunks.
  malloc returns a slot within a chunk,
  while free returns slots back to a chunk.

  We completely exhaust the first chunk before we ever get another one.
  Once a chunk (except for our current one) is COMPLETELY empty, it is returned to the superheap.
*/

#include <assert.h>
#include <new.h>

#include "chunk.h"


template <int chunkSize, int slotSize, class Super>
class LazySlotHeap : public Super {
public:

  LazySlotHeap (void)
    : myChunk (new (Super::malloc (chunkSize)) Chunk<chunkSize, slotSize>())
  {}

  ~LazySlotHeap (void)
  {
    // Give up our chunk.
    Super::free (myChunk);
  }

  inline void * malloc (size_t sz) {
    assert (sz <= slotSize);
    void * ptr = myChunk->getSlot();
    if (ptr == NULL) {
      myChunk = new (Super::malloc (chunkSize)) Chunk<chunkSize, slotSize>();
      ptr = myChunk->getSlot();
      assert (ptr != NULL);
    }
    return ptr;
  }

  inline void free (void * ptr) {
    /// Return a slot to its chunk.
    Chunk<chunkSize, slotSize> * ch = Chunk<chunkSize, slotSize>::getChunk (ptr);
    ch->putSlot (ptr);
	  	// check if it's completely empty. If so, free it.
	  if (ch != myChunk) {
      // Once the chunk is completely empty, free it.
      if (ch->getNumSlotsAvailable() == ch->getNumSlots()) {
        Super::free (ch);
      }
    }

#if 0
	// If this chunk isn't the one we're currently holding,
		// free it. NB: It is NOT guaranteed to be empty!
		if (ch != myChunk) {
      Super::free (ch);
    }
#endif
#if 0
		template <int chunkSize, int slotSize, class Super>
class StrictSlotHeap : public LazySlotHeap {
public:
	inline void free (void * ptr) {
    /// Return a slot to its chunk.
    Chunk<chunkSize, slotSize> * ch = Chunk<chunkSize, slotSize>::getChunk (ptr);
    ch->putSlot (ptr);
  	// check if it's completely empty. If so, free it.
	  if (ch != myChunk) {
      // Once the chunk is completely empty, free it.
      if (ch->getNumSlotsAvailable() == ch->getNumSlots()) {
        Super::free (ch);
      }
    }
  }

};
#endif

  }

  inline static size_t size (void * ptr)
  {
	  return slotSize;
  }


protected:

  Chunk<chunkSize, slotSize> * myChunk;

};



#endif
