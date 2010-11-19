/* -*- C++ -*- */

#ifndef _LOOSESLOTHEAP_H_
#define _LOOSESLOTHEAP_H_

#include <assert.h>

#include "bigchunk.h"

/*
  LooseSlotHeap: A "Loose" slot heap.

  malloc returns objects of size slotSize, allocated from the super heap
  in chunkSize units.

  free returns objects of size slotSize.

  When the slot heap is at least 1/emptyFraction empty, a
  chunk that is at least that empty is freed to the super heap.
  Note that the chunk's memory MAY NOT SAFELY be recycled in toto --
  it must be parceled out using the chunk interface.

  The intent of this heap is to allow mostly-empty chunks to move up
  to a staging area (the super heap) where they may be reused by other
  "loose" slot heaps.

*/

template <int chunkSize, int slotSize, int emptyFraction, class Super>
class LooseSlotHeap {
public:

	inline LooseSlotHeap (void);
	inline ~LooseSlotHeap (void);

  // Get a slotSize object.
  inline void * malloc (size_t sz);

  // Free a slotSize object.
  static inline void free (void * ptr);

private:

	Super theHeap;

	typedef BigChunk<chunkSize, slotSize, LooseSlotHeap> chunkType;

	inline void localFree (void * ptr);

	void checkClassInvariant (void) {
#ifndef NDEBUG
		assert (inUse >= 0);
		assert (inUse <= space);
		assert (lastIndex >= 0);
		assert (lastIndex <= emptyFraction);
		// Now check to make sure that inUse & space are correct.
		int localInUse = 0;
		int localSpace = 0;
		for (int i = 0; i < emptyFraction + 1; i++) {
			chunkType * ch = myChunks[i];
			while (ch != NULL) {
				assert (ch->getHeap() == this);
				localInUse += ch->getNumSlotsInUse();
				localSpace += ch->getNumSlots();
				ch = ch->getNext();
			}
		}
		assert (localSpace == space);
		assert (localInUse == inUse);
#endif
	}

  // How full is this block?
	//   0 == empty to no more than 1/emptyFraction full.
	//   emptyFraction == completely full.
  static inline int getFullness (chunkType * ch);
    
	// Move a chunk, if necessary, into the right list
  // based on its emptiness.
	// Returns the index of which list it is put or left in.
	inline int update (chunkType * ch, int prevIndex);

  // The array of chunks, "radix sorted" by emptiness.
  // completely empty chunks are in index 0.
  // completely full chunks are in index emptyFraction.
  chunkType * myChunks[emptyFraction + 1];

  // The last index (from 0) in the above array that has a chunk.
  int lastIndex;

  // How much space is in use in all of the chunks.
  int inUse;

  // How much space there is (free or not) in all of the chunks.
  int space;
};


template <int chunkSize, int slotSize, int emptyFraction, class Super>
LooseSlotHeap<chunkSize, slotSize, emptyFraction, Super>::LooseSlotHeap (void)
  : space (0),
    inUse (0),
    lastIndex (emptyFraction - 1)
{
	// Initialize the chunk pointers (all initially empty).
	for (int i = 0; i < emptyFraction + 1; i++) {
		myChunks[i] = NULL;
	}
}


template <int chunkSize, int slotSize, int emptyFraction, class Super>
LooseSlotHeap<chunkSize, slotSize, emptyFraction, Super>::~LooseSlotHeap (void)
{
	// Free every chunk.
	chunkType * ch, * tmp;
	for (int i = 0; i < emptyFraction + 1; i++) {
		ch = myChunks[i];
		while (ch != NULL) {
			tmp = ch;
			ch = ch->getNext();
			theHeap.free (tmp);
		}
	}
}


template <int chunkSize, int slotSize, int emptyFraction, class Super>
int LooseSlotHeap<chunkSize, slotSize, emptyFraction, Super>::getFullness (chunkType * ch)
{
  int fullness = (emptyFraction * ch->getNumSlotsInUse())/ch->getNumSlots();
	//printf ("numslots avail = %d, num slots total = %d, fullness = %d\n", ch->getNumSlotsFree(), ch->getNumSlots(), fullness);
	return fullness;
}

	
template <int chunkSize, int slotSize, int emptyFraction, class Super>
void * LooseSlotHeap<chunkSize, slotSize, emptyFraction, Super>::malloc (size_t sz)
{
	checkClassInvariant();
	//printf ("Loose malloc %d (slot size = %d)\n", sz, slotSize);
  assert (sz <= slotSize);
  void * ptr = NULL;
	chunkType * ch = NULL;
  // Always try to allocate from the most-full chunk (resetting
  // lastIndex as we go).
	lastIndex = emptyFraction - 1; // FIX ME!
  while (lastIndex >= 0) {
		ch = myChunks[lastIndex];
    if (ch == NULL) {
      lastIndex--;
    } else {
      // Got one.
      break;
    }
  }
  if (lastIndex < 0) {
		assert (ch == NULL);
    // There were no chunks available.
    assert ((space - inUse) == 0);
    // Make one with memory from the "super" heap.
		printf ("!!!\n");
		ch = (chunkType*) theHeap.malloc (chunkSize);
		ch->setHeap (this);
		assert (ch->getNumSlotsFree() > 0);
		printf ("Super malloc %d\n", chunkSize);
		lastIndex = getFullness(ch);
		register chunkType*& head = myChunks[lastIndex];
		assert (head == NULL);
		ch->setPrev (NULL);
		ch->setNext (NULL);
    head = ch;
    inUse += ch->getNumSlotsInUse();
    space += ch->getNumSlots();
		assert (ch->getNumSlotsFree() <= ch->getNumSlots());
		// FOR NOW!! FIX ME!!!
		assert (ch->getNumSlotsFree() == ch->getNumSlots());
		printf ("Space, in use was %d, %d\n", space, inUse);
		printf ("Space, in use NOW %d, %d\n", space, inUse);
  }
	assert (ch != NULL);
	assert (ch->getNumSlotsFree() > 0);
	int prevFullness = getFullness (ch);
	int prevInUse = ch->getNumSlotsInUse();
	assert (ch->getHeap() == this);
  ptr = ch->getSlot();
  inUse++;
	assert (prevInUse + 1 == ch->getNumSlotsInUse());
	int newFullness = getFullness (ch);
	if (prevFullness != newFullness) {
		int n = update (ch, prevFullness);
		assert (n == newFullness);
	}
  chunkType * bch = (chunkType *) chunkType::getChunk(ptr);
  assert (bch == ch);
  assert (ptr != NULL);
	checkClassInvariant();
  return ptr;
}


template <int chunkSize, int slotSize, int emptyFraction, class Super>
void LooseSlotHeap<chunkSize, slotSize, emptyFraction, Super>::free (void * ptr) {
  chunkType * ch
    = (chunkType *) chunkType::getChunk (ptr);
	assert (ch != NULL);
	(ch->getHeap())->localFree (ptr);
}


template <int chunkSize, int slotSize, int emptyFraction, class Super>
void LooseSlotHeap<chunkSize, slotSize, emptyFraction, Super>::localFree (void * ptr) {
	checkClassInvariant();
	printf ("free! (in use = %d, space = %d)\n", inUse, space);
  // Mark the slot in the chunk as free.
  chunkType * ch
    = (chunkType *) chunkType::getChunk (ptr);
	assert (ch != NULL);
	assert (ch->getHeap() == this);
	int prevFullness = getFullness (ch);
	int prevInUse = ch->getNumSlotsInUse();
  ch->putSlot (ptr);
	inUse--;
	assert (prevInUse - 1 == ch->getNumSlotsInUse());
	checkClassInvariant();
	int newIndex = getFullness (ch);
	if (prevFullness != newIndex) {
		int n = update (ch, prevFullness);
		assert (n == newIndex);
	}
  // If we are more than 1/emptyFraction empty,
  // return a mostly-empty chunk to the super heap.
  if ((space - inUse > 2 * chunkSize/slotSize)
		&& (emptyFraction * (space - inUse) > space)) {
		printf ("RETURNING A CHUNK!\n");
		// Find an empty chunk.
		int emptyIndex = 0;
		ch = NULL;
		while (emptyIndex < emptyFraction) {
			ch = myChunks[emptyIndex];
			if (ch != NULL)
				break;
			emptyIndex++;
		}
		assert (ch != NULL);
		checkClassInvariant();
		ch->setHeap (NULL);
    // Remove a chunk and give it to the super heap.
		myChunks[emptyIndex] = myChunks[emptyIndex]->getNext();
		if (myChunks[emptyIndex] != NULL) {
			myChunks[emptyIndex]->setPrev (NULL);
		}
		ch->setPrev (NULL);
		ch->setNext (NULL);
		// A sanity check on the chunk we're about to return.
		assert (ch->getNumSlotsFree() >= 0);
		assert (ch->getNumSlots() >= ch->getNumSlotsFree());
		printf ("Updating space & in use: was %d, %d (slots in use = %d)\n",
			space, inUse, ch->getNumSlotsInUse());
		space -= ch->getNumSlots();
		inUse -= ch->getNumSlotsInUse();
		printf ("Updating space & in use: NOW %d, %d\n", space, inUse);
		theHeap.free (ch);
		checkClassInvariant();
  }
	checkClassInvariant();
}


// Move a chunk, if necessary, into the right list
// based on its emptiness.
template <int chunkSize, int slotSize, int emptyFraction, class Super>
int LooseSlotHeap<chunkSize, slotSize, emptyFraction, Super>::update (chunkType * ch, int prevIndex)
{
  // Move this chunk if necessary.
#ifndef NDEBUG
	chunkType * bch = myChunks[prevIndex];
	while (bch != NULL) {
		if (bch == ch)
			break;
		bch = bch->getNext();
	}
	assert (bch == ch);
#endif
  int newIndex = getFullness(ch);
  // printf ("newIndex = %d\n", newIndex);
	// Reset last index (for simplicity). // FIX ME??
  // Remove the chunk from its current list.
	if (ch == myChunks[prevIndex]) {
		myChunks[prevIndex] = myChunks[prevIndex]->getNext();
	}
	/////	lastIndex = emptyFraction - 1;
  if (ch->getPrev() != NULL) {
		ch->getPrev()->setNext(ch->getNext());
  }
	if (ch->getNext() != NULL) {
	  ch->getNext()->setPrev(ch->getPrev());
  }
  // Add it to the newIndex list.
  chunkType*& head = myChunks[newIndex];
  ch->setPrev (NULL);
  ch->setNext (head);
  if (head != NULL) {
    head->setPrev(ch);
  }
	head = ch;
	// Verify that the chunk is in the right list.
#ifndef NDEBUG
	bch = myChunks[newIndex];
	while (bch != NULL) {
		if (bch == ch)
			break;
		bch = bch->getNext();
	}
	assert (bch == ch);
#endif
	return newIndex;
}



#endif
