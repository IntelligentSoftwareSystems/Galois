/* -*- C++ -*- */

#ifndef _BIGCHUNK_H_
#define _BIGCHUNK_H_

#include <assert.h>

template <int chunkSize, int slotSize, class Super>
class BigChunk {
public:

  inline BigChunk (void);

  // Get and put slots.
  // Get returns NULL if there are no slots available.
  inline void * getSlot (void);
  inline void putSlot (void *);

  // How many slots are there in total?
  inline static int getNumSlots (void);

  // How many slots are available?
  inline int getNumSlotsFree (void);

  // How many slots are available?
  inline int getNumSlotsInUse (void);

  // Find a chunk given its slot.
  inline static BigChunk * getChunk (void *);

  class ChunkBlock {
  public:
    void setChunk (BigChunk * ch) { _myChunk = ch; }
    BigChunk * getChunk (void) { return _myChunk; }
    void setNext (ChunkBlock * n) { _next = n; }
    ChunkBlock * getNext (void) { return _next; }
  private:
    BigChunk * _myChunk;   // If allocated, point to the chunk.
    ChunkBlock * _next; // If not allocated, point to the next chunk block.
  };

  // Find a chunk block for a given pointer.
  static inline ChunkBlock * getChunkBlock (void * ptr) {
	assert (ptr != NULL);
	return (ChunkBlock *) ptr - 1;
  }

  void setHeap (Super * h) {
	heap = h;
  }

  inline Super * getHeap (void) {
	return heap;
  }

  // Add doubly linked-list operations.
  inline BigChunk * getNext (void);
  inline BigChunk * getPrev (void);
  inline void setNext (BigChunk *);
  inline void setPrev (BigChunk *);

private:

  static inline size_t align (size_t sz) {
    return (sz + (sizeof(double) - 1)) & ~(sizeof(double) - 1);
  }

  ChunkBlock * freeSlots;
  int numSlotsAvailable;
  Super * heap;
  BigChunk * prev;
  BigChunk * next;
};


template <int chunkSize, int slotSize, class Super>
BigChunk<chunkSize, slotSize, Super> * BigChunk<chunkSize, slotSize, Super>::getNext (void)
{
  return next;
}


template <int chunkSize, int slotSize, class Super>
BigChunk<chunkSize, slotSize, Super> * BigChunk<chunkSize, slotSize, Super>::getPrev (void)
{
  return prev;
}


template <int chunkSize, int slotSize, class Super>
void BigChunk<chunkSize, slotSize, Super>::setNext (BigChunk<chunkSize, slotSize, Super> * ptr)
{
  next = ptr;
}


template <int chunkSize, int slotSize, class Super>
void BigChunk<chunkSize, slotSize, Super>::setPrev (BigChunk<chunkSize, slotSize, Super> * ptr)
{
  prev = ptr;
}


template <int chunkSize, int slotSize, class Super>
BigChunk<chunkSize, slotSize, Super>::BigChunk (void)
  : freeSlots (NULL),
    numSlotsAvailable (getNumSlots()),
	prev (NULL),
	next (NULL)
{
  //printf ("numSlots = %d\n", numSlots);
  int numSlots = getNumSlots();
  assert (numSlots > 0);
  const int blksize = align(sizeof(ChunkBlock) + slotSize);
  //printf ("blksize = %d\n", blksize);
  // Carve up the chunk into a number of slots.
  ChunkBlock * b = (ChunkBlock *) (this + 1);
  for (int i = 0; i < numSlots; i++) {
	assert ((unsigned long) b < ((unsigned long) (this + 1) + blksize * numSlots));
    new (b) ChunkBlock;
    b->setChunk (this);
	assert (b->getChunk() == this);
    b->setNext (freeSlots);
    freeSlots = b;
    b = (ChunkBlock *) ((char *) b + blksize);
  }
}


template <int chunkSize, int slotSize, class Super>
void * BigChunk<chunkSize, slotSize, Super>::getSlot (void)
{
  if (freeSlots == NULL) {
    assert (numSlotsAvailable == 0);
    return NULL;
  }
  assert (numSlotsAvailable > 0);
  assert (numSlotsAvailable <= getNumSlots());
  ChunkBlock * b = freeSlots;
  freeSlots = freeSlots->getNext();
  numSlotsAvailable--;
  b->setChunk (this); // FIX ME -- this should be unnecessary.
  b->setNext (NULL);
  void * ptr = (void *) (b + 1);
  BigChunk<chunkSize, slotSize, Super> * bch = getChunk(ptr);
  assert (bch == this);
  return (void *) (b + 1);
}


template <int chunkSize, int slotSize, class Super>
void BigChunk<chunkSize, slotSize, Super>::putSlot (void * ptr)
{
  assert (numSlotsAvailable >= 0);
  assert (numSlotsAvailable <= getNumSlots());
  ChunkBlock * b = getChunkBlock (ptr);
  assert (b->getChunk() == this);
  b->setNext (freeSlots);
  freeSlots = b;
  numSlotsAvailable++;
  assert (numSlotsAvailable > 0);
  assert (numSlotsAvailable <= getNumSlots());
}


template <int chunkSize, int slotSize, class Super>
int BigChunk<chunkSize, slotSize, Super>::getNumSlots (void)
{
  return ((chunkSize - sizeof(BigChunk)) / align (sizeof(ChunkBlock) + slotSize));
}


template <int chunkSize, int slotSize, class Super>
int BigChunk<chunkSize, slotSize, Super>::getNumSlotsFree (void)
{
  return numSlotsAvailable;
}


template <int chunkSize, int slotSize, class Super>
int BigChunk<chunkSize, slotSize, Super>::getNumSlotsInUse (void)
{
  return getNumSlots() - numSlotsAvailable;
}


template <int chunkSize, int slotSize, class Super>
BigChunk<chunkSize, slotSize, Super> * BigChunk<chunkSize, slotSize, Super>::getChunk (void * ptr)
{
  ChunkBlock * ch = getChunkBlock (ptr);
  return ch->getChunk();
}

#endif
