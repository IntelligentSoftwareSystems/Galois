/* -*- C++ -*- */

#ifndef _TOPSLOTHEAP_H_
#define _TOPSLOTHEAP_H_

#include <assert.h>

#include "bigchunk.h"

/*
  TopSlotHeap.

  malloc returns objects of size chunkSize.

  free returns objects of size chunkSize.

*/

template <int chunkSize, int slotSize, class Super>
class TopSlotHeap : public Super {
public:

  TopSlotHeap (void)
    : myChunks (NULL)
  {}

  // Get a chunkSize object.
  inline void * malloc (size_t sz);

  // Free a chunkSize object.
  inline void free (void * ptr);

protected:

	virtual inline void localFree (void * ptr);

	
private:

  BigChunk<chunkSize, slotSize, TopSlotHeap> * myChunks;

};


template <int chunkSize, int slotSize, class Super>
void * TopSlotHeap<chunkSize, slotSize, Super>::malloc (size_t sz)
{
  assert (sz <= chunkSize);
  if (myChunks == NULL) {
	return new (Super::malloc (chunkSize)) BigChunk<chunkSize, slotSize, TopSlotHeap>;
  } else {
	printf ("Recycled a chunk.\n");
	BigChunk<chunkSize, slotSize, TopSlotHeap> * ch = myChunks;
	myChunks = myChunks->getNext();
	ch->setNext (NULL);
	ch->setHeap (NULL);
	return ch;
  }
}


template <int chunkSize, int slotSize, class Super>
void TopSlotHeap<chunkSize, slotSize, Super>::free (void * ptr) {
  printf ("Freed a chunk.\n");
  BigChunk<chunkSize, slotSize, TopSlotHeap> * ch = (BigChunk<chunkSize, slotSize, TopSlotHeap> *) ptr;
  ch->setNext (myChunks);
  ch->setHeap (this);
  myChunks = ch;
}


template <int chunkSize, int slotSize, class Super>
void TopSlotHeap<chunkSize, slotSize, Super>::localFree (void * ptr) {
	free (ptr);
}


#endif
