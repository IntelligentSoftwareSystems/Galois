/* -*- C++ -*- */

#ifndef _LAZYSLOTHEAP_H_
#define _LAZYSLOTHEAP_H_

/*
  This heap manages memory in units of Chunks.
  malloc returns a slot within a chunk,
  while free returns slots back to a chunk.
  
  Once a chunk is COMPLETELY empty, it is returned to the superheap.
*/

#include <assert.h>
#include <new.h>

template <int chunkSize, int slotSize, class Super>
class LazySlotHeap : public Super {
public:

  LazySlotHeap (void)
    : myChunk (new (Super::malloc (sz)) Chunk<chunkSize, slotSize>())
  {}

  ~LazySlotHeap (void)
  {
    // Give up our chunk.
    Super::free (myChunk);
  }

  inline void * malloc (size_t sz) {
    assert (sz == chunkSize);
    void * ptr = myChunk->getSlot();
    if (ptr == NULL) {
      myChunk = new (Super::malloc (sz)) Chunk<chunkSize, slotSize>();
      ptr = myChunk->getSlot();
      assert (ptr != NULL);
    }
    return ;
  }

  inline void free (void * ptr) {
    /// Return a slot to its chunk.
    Chunk<chunkSize, slotSize> * ch = Chunk<chunkSize, slotSize>::getChunk (ptr);
    ch->putSlot (ptr);
    // Once the chunk is completely empty, free it.
    if (ch->getNumSlotsAvailable() == ch->getNumSlots()) {
      if (ch == myChunk) {
	// If this was 'our' chunk, get another one.
	myChunk = new (Super::malloc (sz)) Chunk<chunkSize, slotSize>();
      }
      Super::free (ch);
    }
  }

private:

  Chunk<chunkSize, slotSize> * myChunk;

};


#endif
