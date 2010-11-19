/* -*- C++ -*- */

#ifndef _AOFHEAP_H_
#define _AOFHEAP_H_

#include <new> // for placement new
using namespace std;

/*
  
  A very simple address-oriented first-fit heap
  with immediate coalescing.

  Because the heap is stored as a sorted doubly-linked list,
  it's potentially quite expensive:

  cost of a malloc = O(f) {# of non-contiguous freed objects}
  cost of a free   = O(f) {# of non-contiguous freed objects}

  Worst-case performance could be improved by using a balanced O(log n) dictionary
  (like an RB-tree). Also, the free list is threaded through the objects themselves,
  something that can cause O(f) page faults; keeping the free list data structures
  separate from the data (and in a contiguous region of memory) would avoid this problem.

*/

// Freed objects of at least FreeThreshold size are freed to the super heap.

template <class SuperHeap, int FreeThreshold>
class AOFFHeap : public SuperHeap {
public:

  AOFFHeap (void)
  {
    freeList.setSize (0);
    freeList.setNext (&freeList);
    freeList.setPrev (&freeList);
  }

  ~AOFFHeap (void) {
    return; // FIX ME FIX ME FIX ME!!!
    // Free everything on the free list.
    freedObject * ptr = freeList.getNext();
    while (ptr != &freeList) {
      freedObject * prev = ptr;
      ptr = ptr->getNext();
      SuperHeap::free (prev);
    }
  }

  void * malloc (size_t sz) {
    // printf ("aoffheap malloc sz = %d\n", sz);
    assert (isValid());
    assert (sz >= sizeof(freedObject) - sizeof(allocatedObject));
    // Find the first object in the free list that fits.
    freedObject * ptr = freeList.getNext();
    while ((ptr != &freeList) && (ptr->getSize() < sz + sizeof(freedObject))) {
      ptr = ptr->getNext();
    }
    // If there wasn't a big-enough object available on the free list,
    // make a new one.
    if (ptr == &freeList) {

      // printf ("AOFF super malloc(%d)\n", sz + sizeof(allocatedObject));

      void * buf = SuperHeap::malloc (sz + sizeof(allocatedObject));
      if (buf == NULL) {
	assert (isValid());
	return NULL;
      }
      freedObject * newptr 
	= new (buf) freedObject (sz, NULL, NULL);
      assert (size(newptr->getStart()) == sz);
      assert (isValid());
      return (void *) newptr->getStart();
    } else {
      assert (ptr->getSize() >= sz);
      // Remove it from the free list.
      freedObject * prev = ptr->getPrev();
      freedObject * next = ptr->getNext();
      assert (prev->isValid());
      assert (next->isValid());
      // If it's bigger than the request size,
      // splice it up.
      if (ptr->getSize() - sz >= sizeof(freedObject)) {
				// There's room for another object.
				// Splice it off.
	size_t oldSize = ptr->getSize();
	ptr->setSize (sz);
	freedObject * splicedObj = new (ptr->getSuccessor())
	  freedObject (oldSize - sz - sizeof(freedObject),
		       prev,
		       next);
	prev->setNext (splicedObj);
	next->setPrev (splicedObj);
	assert (splicedObj->isValid());
	assert (isValid());
	assert (size(ptr->getStart()) == sz);
	return (void *) ptr->getStart();
      } else {
	assert (0);
	abort();
				// It's just right.
				// Just remove it.
	prev->setNext (next);
	next->setPrev (prev);
	assert (isValid());
	assert (size(ptr->getStart()) == sz);
	return (void *) ptr->getStart();
      }
    }
  }

  void free (void * p) {
    assert (isValid());
    // Put the object back in sorted order (by address).
    freedObject * thisObject = (freedObject *) ((char *) p - sizeof(allocatedObject));
    assert (thisObject->getStart() == p);
    freedObject * prev = &freeList;
    freedObject * next = freeList.getNext();
    while ((next != &freeList) && ((char *) thisObject > (char *) next)) {
      prev = next;
      next = next->getNext();
    }
    // Check if this object is already on the free list
    // (i.e., it was already deleted).
    if (thisObject == next) {
      // printf ("Bad call.\n");
      // Ignore the bad programmer and just walk away...
      return;
    }
    // Now insert this object.
    prev->setNext (thisObject);
    next->setPrev (thisObject);
    thisObject = new (thisObject) freedObject (thisObject->getSize(), prev, next);
    // See if we can coalesce.
    if (prev->getSuccessor() == thisObject) {
      assert (prev != &freeList);
      coalesce (prev, thisObject);
      thisObject = prev;
      assert (thisObject->isValid());
      assert (thisObject->getSize() > 0);
      // printf ("coalesced prev with this, new size = %d\n", thisObject->getSize());
    }
    if (thisObject->getSuccessor() == next) {
      coalesce (thisObject, next);
      assert (thisObject->isValid());
      //printf ("coalesced this with next, new size = %d\n", thisObject->getSize());
    }
    // If this object is big enough, free it.
    if (thisObject->getSize() >= FreeThreshold) {
      // printf ("freed this (size = %d)\n", thisObject->getSize());
      freedObject * prev = thisObject->getPrev();
      freedObject * next = thisObject->getNext();
      prev->setNext (next);
      next->setPrev (prev);
      assert (thisObject->isValid());
      SuperHeap::free (thisObject);
    }
    assert (isValid());
  }

  //protected:
  inline static size_t size (void * p)
  {
    allocatedObject * thisObject = (allocatedObject *) p - 1;
    assert (thisObject->isValid());
    return thisObject->getSize();
  }


private:

  int isValid (void) {
    // March through the whole list and check for validity.
    freedObject * ptr = freeList.getNext();
    while (ptr != &freeList) {
      freedObject * prev = ptr;
      assert (prev->getNext()->getPrev() == prev);
      assert (prev->isValid());
      ptr = ptr->getNext();
    }
    return 1;
  }

  class freedObject;

  inline void coalesce (freedObject * curr, freedObject * succ) {
    // printf ("Coalesce %d with %d\n", curr->getSize(), succ->getSize());
    assert (curr->getSuccessor() == succ);
    assert (succ->getPrev() == curr);
    assert (curr->getNext() == succ);
    curr->setNext (succ->getNext());
    succ->getNext()->setPrev(curr);
    curr->setSize (curr->getSize() + succ->getSize() + sizeof(allocatedObject));
    assert (curr->isValid());
  }
  inline static size_t align (int sz) {
    return (sz + (sizeof(double) - 1)) & ~(sizeof(double) - 1);
  }

  class allocatedObject {
  public:
    allocatedObject (void)
    {
      size = 0;
#ifndef NDEBUG
      magic = 0xfacefade;
#endif
    }
    int isValid (void) const {
      return (size >= 0) && (magic == 0xfacefade);
    }
    void setSize (size_t sz) { size = sz; assert (isValid()); }
    int getSize (void) const { assert (isValid()); return size; }

    // Return the start of the next object.
    allocatedObject * getSuccessor (void) const {
      assert (isValid());
      return (allocatedObject *) ((char *) (this + 1) + size);
    }

    // Return the start of the actual object (beyond the header).
    char * getStart (void) const {
      assert (isValid());
      return ((char *) (this + 1));
    }
  private:
    int size;
    int magic;
  };

  class freedObject : public allocatedObject {
  public:
    freedObject (void)
      : prev ((freedObject *) 0xdeadbeef),
	next ((freedObject *) 0xdeadbeef)
    {}
    freedObject (size_t sz,
		 freedObject * p,
		 freedObject * n)
      :	prev (p),
	next (n)
    {
      allocatedObject::setSize (sz);
    }
    int isValid (void) const {
      return (allocatedObject::isValid());
    }
    freedObject * getPrev (void) const { assert (isValid()); return prev; }
    freedObject * getNext (void) const { assert (isValid()); return next; }
    void setPrev (freedObject * p) { assert (isValid()); prev = p; }
    void setNext (freedObject * p) { assert (isValid()); next = p; }
  private:
    freedObject * prev;
    freedObject * next;
  };


  freedObject freeList;
  double _dummy; // Just to make sure that the freeList node won't ever be coalesced!
};

#endif
