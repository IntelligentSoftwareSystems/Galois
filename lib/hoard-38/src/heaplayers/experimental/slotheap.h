/* -*- C++ -*- */

#ifndef _SLOTHEAP_H_
#define _SLOTHEAP_H_

// NOTE: All size requests to malloc must be identical!

/*

  A "slot" allocator.

  All allocations come from a fixed-size chunk of memory
  that is carved into a number of pieces.

  The "chunk" class must support the following methods:

  void * getSlot (void); // Returns NULL if there is no slot left.
  void putSlot (void *); // Puts a slot back into its chunk.

*/

#include <assert.h>

#include "chunkheap.h"

/* A "slot" heap.

   This heap reserves exactly one "chunk" that is divided into
   a number of fixed-size slots. When the chunk is used up,
   the heap requests another one. */

template <int chunkSize, int slotSize, class Super>
class SlotInterface;

template <int chunkSize, int slotSize, class Super>
class SlotHeap : public SlotInterface<chunkSize, slotSize, ChunkHeap<chunkSize, slotSize, Super> >{};

template <int chunkSize, int slotSize, class Super>
class SlotInterface : public Super {
public:

  SlotInterface (void)
    : currentChunk (new (Super::malloc(chunkSize)) Chunk<chunkSize, slotSize>)
  {}
  
  inline void * malloc (size_t sz) {
    assert (sz == slotSize);
    // Use up all of the slots in one chunk,
    // and get another chunk if we need one.
    void * ptr = currentChunk->getSlot();
    if (ptr == NULL) {
      // This chunk is empty -- get another one.
      currentChunk = new (Super::malloc(chunkSize)) Chunk<chunkSize, slotSize>;
      ptr = currentChunk->getSlot();
    } 
    assert (ptr != NULL);
    return ptr;
  }
  
  inline void free (void * ptr) {
    // If this object belongs to "our" chunk,
    // free it directly; otherwise, pass it up.
    if (getChunk(ptr) == currentChunk) {
      currentChunk->putSlot (ptr);
    } else {
      Super::free (ptr);
    }
  }

private:

  Chunk<chunkSize, slotSize> * currentChunk;

};

#endif
