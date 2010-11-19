/* -*- C++ -*- */

#ifndef _ALIGNEDCHUNK_H_
#define _ALIGNEDCHUNK_H_

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

#include <stdlib.h>
#include <malloc.h>
#include <assert.h>

#include "bitindex.h"

/*
	An aligned chunk is a chunk of memory
	containing a number of fixed-size "slots".
	As long as the chunk is naturally aligned,
	each slot will also be naturally aligned.

	This alignment allows us to use address masking to find
	which chunk a given pointer belongs to.

	All information about the chunk is stored at the end
	of the chunk.

*/

// AlignHeap aligns every memory request to chunkSize.
//
// If you know that memory allocations are always aligned to chunkSize
// from your allocator of choice, don't use AlignHeap because it
// will waste memory.

namespace HL {

template <int chunkSize, class Super>
class AlignHeap {
public:
	inline void * malloc (size_t sz) {
		// Get a piece of memory large enough to be able to guarantee alignment.
		void * buf = ::malloc (sz + chunkSize);
		// Align the buffer.
		void * alignedBuf = (void *) (((unsigned long) buf + sizeof(unsigned long) + chunkSize - 1) & -chunkSize);
		// Record the original buffer's address right behind the aligned part.
		assert ((unsigned long) alignedBuf - (unsigned long) buf > sizeof(unsigned long));
		*((unsigned long *) alignedBuf - 1) = (unsigned long) buf;
		return alignedBuf;
	}

	void free (void * ptr) {
		// Get the original buffer's address and free that.
		::free ((void *) *((unsigned long *) ptr - 1));
	}
};


// An aligned chunk is of size chunkSize and is divided up into 32 pieces
// of size slotSize. The 32nd one will not be available for allocation
// because it is used for 'header' information (albeit stored in the footer).

// Some restrictions: chunkSize MUST BE A POWER OF TWO.
//                    slotSize MUST BE AT LEAST THE SIZE OF A DOUBLE.

// The amount of memory that this approach wastes is small in practice:
//   For one aligned chunk, utilization is 31/32 = 97%.
//   For two nested chunks, utilization is (31/32)^2 = 94%.
//   For three nested chunks, utilization is (31/32)^3 = 91%.
// Note that the smallest possible size of a three-deep aligned chunk
// is 32 * 32 * 32 * 32 = one megabyte.

template <int chunkSize, int slotSize>
class AlignedChunk {
public:

	AlignedChunk (void)
		: prev (NULL),
			next (NULL),
			status (ACQUIRED),
			inUse (0)
	{
		// Make sure there's enough slop to let us store the chunk information!
		assert (getNumSlots() * slotSize + sizeof(AlignedChunk) <= chunkSize);
		// The slot size must be at least large enough to hold a double.
		assert (slotSize == align(slotSize));
		// Initialize the bitmap.
		freeBitmap = 0;
		// Block the last slot.
		BitIndex::set (freeBitmap, 0);
		emptyBitmap = freeBitmap;
	}

	~AlignedChunk (void)
		{}


  // Get and put slots.
	// These are ATOMIC and lock-free.

  // Get returns NULL iff there are no slots available.
  void * getSlot (void) {
RETRY_UNSET:
		unsigned long currBitmap = freeBitmap;
		// If currBitmap is all ones, everything is in use.
		// Just return NULL.
		if (currBitmap == (unsigned long) -1) {
			assert (inUse == getNumSlots());
			return NULL;
		}
		// Find an unset bit.
		// We flip the bits in currBitmap and find the index of a one bit
		// (which corresponds to the index of a zero in currBitmap).
		int bitIndex;
		unsigned long oneBit = (~currBitmap) & (-((signed long) ~currBitmap));
		assert (oneBit != 0);
		bitIndex = BitIndex::msb (oneBit);
		if (bitIndex > getNumSlots()) {
			assert (inUse == getNumSlots());
			return NULL;
		}
		assert (inUse < getNumSlots());
		assert (bitIndex < getNumSlots() + 1);
		// Set it.
		unsigned long oldBitmap = currBitmap;
		BitIndex::set (currBitmap, bitIndex);
		unsigned long updatedBitmap = InterlockedExchange ((long *) &freeBitmap, currBitmap);
		if (updatedBitmap == oldBitmap) {
			// Success.
			// Return a pointer to the appropriate slot.
			char * start = (char *) ((unsigned long) this & -chunkSize);
			inUse++;
			return start + slotSize * (getNumSlots() - bitIndex);
		}
		// Someone changed the bitmap before we were able to write it.
		// Try again.
		goto RETRY_UNSET;
	}


  // Put returns 1 iff the chunk is now completely empty.
  inline int putSlot (void * ptr) {
		assert (inUse >= 1);
		AlignedChunk * ch = getChunk (ptr);
		assert (ch == this);
		// Find the index of this pointer.
		// Since slotSize is known at compile time and is usually a power of two,
		// the divide should be turned into shifts and will be fast.
		char * start = (char *) ((unsigned long) ptr & -chunkSize);
		int bitIndex = getNumSlots() - (((unsigned long) ((char *) ptr - start)) / slotSize);
RETRY_RESET:
		unsigned long currBitmap = freeBitmap;
		unsigned long oldBitmap = currBitmap;
		BitIndex::reset (currBitmap, bitIndex);
		unsigned long updatedBitmap = InterlockedExchange ((long *) &freeBitmap, currBitmap);
		if (updatedBitmap == oldBitmap) {
			// Success.
			inUse--;
			assert ((inUse != 0) || (currBitmap == emptyBitmap));
			assert ((inUse == 0) || (currBitmap != emptyBitmap));
			// Return 1 iff the chunk is now empty.
			if (currBitmap == emptyBitmap) {
				assert (inUse == 0);
				return 1;
			} else {
				assert (inUse > 0);
				return 0;
			}
		}
		// Try again.
		goto RETRY_RESET;
	}


  // How many slots are there in total?
  inline static int getNumSlots (void) {
		return 31;
	}

	inline int isReleased (void) {
		return (status == RELEASED);
	}

	inline void acquire (void) {
		assert (status == RELEASED);
		status = ACQUIRED;
	}

	inline void release (void) {
		assert (status == ACQUIRED);
		status = RELEASED;
	}

  // Find a chunk for a given slot.
  inline static AlignedChunk * getChunk (void * slot) {
		// Here's where the alignment is CRITICAL!!
		// Verify that chunkSize is a power of two.
		assert ((chunkSize & (chunkSize - 1)) == 0);
		// Mask off the slot to find the start of the chunkSize block.
		char * start = (char *) ((unsigned long) slot & -chunkSize);
		// Find the end of the block.
		char * eob = (start + chunkSize);
		// Now locate the 'header' (in this case, it's actually a footer).
		char * headerPos = eob - sizeof(AlignedChunk);
		return (AlignedChunk *) headerPos;
	}


  // Add doubly linked-list operations.
  AlignedChunk * getNext (void) { return next; }
  AlignedChunk * getPrev (void) { return prev; } 
  void setNext (AlignedChunk * p) { next = p; }
  void setPrev (AlignedChunk * p) { prev = p; } 

private:

	enum { RELEASED = 0, ACQUIRED = 1 };

  static inline size_t align (size_t sz) {
    return (sz + (sizeof(double) - 1)) & ~(sizeof(double) - 1);
  }

	int inUse; // For debugging only.
  unsigned long freeBitmap;
	unsigned long emptyBitmap;
	int status;
  AlignedChunk * prev;
  AlignedChunk * next;
};


// AlignedChunkHeap manages a number of chunks.

template <int maxFree, int chunkSize, int slotSize, class Super>
class AlignedChunkHeap : public Super {
public:

	AlignedChunkHeap (void)
		: chunkList (NULL),
		length (0)
	{}

	virtual ~AlignedChunkHeap (void)
	{
		chunkType * ch, * tmp;
		ch = chunkList;
		while (ch != NULL) {
			tmp = ch;
			ch = ch->getNext();
			assert (tmp->isReleased());
			Super::free ((char *) ((unsigned long) tmp & -chunkSize));
		}
	}

	// Malloc a CHUNK. Returns a pointer to the start of the allocated block.
	inline void * malloc (size_t sz)
	{
		assert (sz == chunkSize);
		chunkType * ch;
		// Get a chunk from our chunk list
		// or make one.
		if (chunkList != NULL) {
			ch = chunkList;
			chunkList = chunkList->getNext();
			length--;
			ch->acquire();
		} else {
			// Make a buffer large enough to hold the chunk.
			void * buf = Super::malloc (chunkSize);
			// The buffer MUST BE ALIGNED.
			assert ((unsigned long) buf == ((unsigned long) buf & -chunkSize));
			// Instantiate the chunk "header" (actually the footer)
			// at the end of the chunk.
			ch = new (chunkType::getChunk (buf)) chunkType;
		}
		// Now return the start of the chunk's buffer.
		assert (!ch->isReleased());
		return (void *) ((unsigned long) ch & -chunkSize);
	}

	// Free a CHUNK.
	// The pointer is to the chunk header.
	inline void free (void * ptr)
	{
		chunkType * ch = (chunkType *) AlignedChunk<chunkSize, slotSize>::getChunk(ptr);
		assert (ch->isReleased());
		if (length > maxFree) {
			Super::free ((void *) ((unsigned long) ch & -chunkSize));
		} else {
		  ch->setNext (chunkList);
		  chunkList = ch;
			length++;
		}
	}

	size_t size (void * ptr) {
		return slotSize;
	}

private:

	typedef AlignedChunk<chunkSize, slotSize> chunkType;

	chunkType * chunkList;
	int length;
};


// An AlignedSlotHeap holds at most one chunk.
// When all of the slots are allocated from the chunk,
// the chunk is "released" so that it may be freed back
// to the super heap when all of its slots are freed.

template <int chunkSize, int slotSize, class Super>
class AlignedSlotHeap : public Super {
public:

	AlignedSlotHeap (void)
		: myChunk (NULL)
	{}

	virtual ~AlignedSlotHeap (void) {
		// Note that this is NOT enough to completely clean up after ourselves!!
		if (myChunk != NULL) {
			myChunk->release();
			Super::free ((void *) ((unsigned long) myChunk & -chunkSize));
		}
	}

	// Malloc a SLOT.
	// Use up a chunk, if we've got one.
	// Once it's used up, 'release it' and get another one.
	inline void * malloc (size_t sz) {
		assert (sz <= slotSize);
		void * ptr = NULL;
		while (ptr == NULL) {
			if (myChunk == NULL) {
				myChunk = AlignedChunk<chunkSize, slotSize>::getChunk(Super::malloc (chunkSize));
			}
			ptr = myChunk->getSlot();
			if (ptr == NULL) {
				// This chunk is completely in use.
				// "Release" it.
				myChunk->release();
				myChunk = NULL;
			}
		};
		return ptr;
	}				

	// Free a SLOT.
	// If the chunk is now completely empty and has been 'released',
	// free the whole chunk.
	inline void free (void * ptr)
	{
		// Find out which chunk this slot belongs to.
		AlignedChunk<chunkSize, slotSize> * ch = AlignedChunk<chunkSize, slotSize>::getChunk (ptr);
		// Return it to its chunk.
		int empty = ch->putSlot (ptr);
		if (empty && ch->isReleased()) {
			Super::free ((void *) ((unsigned long) ch & -chunkSize));
		}
	}

private:

	// A one chunk buffer. Emptied when the chunk is completely allocated.
	AlignedChunk<chunkSize, slotSize> * myChunk;

};


template <int maxFree, int chunkSize, int slotSize, class Super>
class AlignedChunkHeapFoo : public AlignedSlotHeap<chunkSize, slotSize, AlignedChunkHeap<maxFree, chunkSize, slotSize, Super> > {};

};

#endif
