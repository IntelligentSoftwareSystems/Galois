/* -*- C++ -*- */

#ifndef _CHUNK_H_
#define _CHUNK_H_

/*

  Heap Layers: An Extensible Memory Allocation Infrastructure
  
  Copyright (C) 2000-2003 by Emery Berger
  http://www.cs.umass.edu/~emery
  emery@cs.umass.edu
  
  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.
  
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  
  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

*/

#include <assert.h>

namespace HL {

template <int chunkSize, int slotSize>
class Chunk {
public:

  inline Chunk (void);

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
  inline static Chunk * getChunk (void *);

  class ChunkBlock {
  public:
    void setChunk (Chunk * ch) { _myChunk = ch; }
    Chunk * getChunk (void) { return _myChunk; }
    void setNext (ChunkBlock * n) { _next = n; }
    ChunkBlock * getNext (void) { return _next; }
  private:
    Chunk * _myChunk;   // If allocated, point to the chunk.
    ChunkBlock * _next; // If not allocated, point to the next chunk block.
  };

  // Find a chunk block for a given pointer.
  static inline ChunkBlock * getChunkBlock (void * ptr) {
	assert (ptr != NULL);
	return (ChunkBlock *) ptr - 1;
  }

private:

  ChunkBlock * freeSlots;
  int numSlotsAvailable;

  static inline size_t align (size_t sz) {
    return (sz + (sizeof(double) - 1)) & ~(sizeof(double) - 1);
  }

};

template <int chunkSize, int slotSize>
Chunk<chunkSize, slotSize>::Chunk (void)
  : freeSlots (NULL),
    numSlotsAvailable (getNumSlots())
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


template <int chunkSize, int slotSize>
void * Chunk<chunkSize, slotSize>::getSlot (void)
{
  if (freeSlots == NULL) {
    assert (numSlotsAvailable == 0);
    return NULL;
  }
  assert (numSlotsAvailable > 0);
  ChunkBlock * b = freeSlots;
  freeSlots = freeSlots->getNext();
  numSlotsAvailable--;
  b->setChunk (this); // FIX ME -- this should be unnecessary.
  b->setNext (NULL);
  void * ptr = (void *) (b + 1);
  Chunk<chunkSize, slotSize> * bch = getChunk(ptr);
  assert (bch == this);
  return (void *) (b + 1);
}


template <int chunkSize, int slotSize>
void Chunk<chunkSize, slotSize>::putSlot (void * ptr)
{
  ChunkBlock * b = getChunkBlock (ptr);
  assert (b->getChunk() == this);
  b->setNext (freeSlots);
  freeSlots = b;
  numSlotsAvailable++;
  assert (numSlotsAvailable <= getNumSlots());
}


template <int chunkSize, int slotSize>
int Chunk<chunkSize, slotSize>::getNumSlots (void)
{
  return ((chunkSize - sizeof(Chunk)) / align (sizeof(ChunkBlock) + slotSize));
}


template <int chunkSize, int slotSize>
int Chunk<chunkSize, slotSize>::getNumSlotsFree (void)
{
  return numSlotsAvailable;
}


template <int chunkSize, int slotSize>
int Chunk<chunkSize, slotSize>::getNumSlotsInUse (void)
{
  return getNumSlots() - numSlotsAvailable;
}


template <int chunkSize, int slotSize>
Chunk<chunkSize, slotSize> * Chunk<chunkSize, slotSize>::getChunk (void * ptr)
{
  ChunkBlock * ch = getChunkBlock (ptr);
  return ch->getChunk();
}

};

#endif
