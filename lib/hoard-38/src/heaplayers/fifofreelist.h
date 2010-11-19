/* -*- C++ -*- */

#ifndef _FIFOFREELIST_H_
#define _FIFOFREELIST_H_

#include <assert.h>

#ifdef NDEBUG
#define IFDEBUG(x)
#else
#define IFDEBUG(x) x
#endif


template <class SuperHeap>
class FIFOFreelistHeap : public SuperHeap {
public:
  
  FIFOFreelistHeap (void)
	  IFDEBUG(:	nObjects (0))
  {
	  head.next = &tail;
	  tail.next = &tail;
	  assert (isEmpty());
  }

  ~FIFOFreelistHeap (void)
  {
	// printf ("Adios free list.\n");
    // Delete everything on the free list.
    freeObject * ptr = head.next;
    while (ptr != &tail) {
      void * oldptr = ptr;
      ptr = ptr->next;
      SuperHeap::free (oldptr);
    }
  }

  inline void * malloc (size_t sz) {
	assert (isValid());
    // Check the free list first.
    freeObject * ptr = head.next;
    if (ptr == &tail) {
	  assert (nObjects == 0);
	  assert (isEmpty());
      ptr = (freeObject *) SuperHeap::malloc (sz);
    } else {
	  assert (nObjects > 0);
	  head.next = ptr->next;
	  if (head.next == &tail) {
		  tail.next = &tail;
	  }
	  IFDEBUG (nObjects--);
    }
	assert (isValid());
    return (void *) ptr;
  }
  
  inline void free (void * ptr) {
	assert (isValid());
    // Add this object to the free list.
	assert (ptr != NULL);
	IFDEBUG (nObjects++);
	freeObject * p = (freeObject *) ptr;
    p->next = &tail;
	tail.next->next = p;
	tail.next = p;
	if (head.next == &tail) {
		head.next = p;
	}
	assert (!isEmpty());
	assert (isValid());
  }

private:

	int isValid (void) {
		// Make sure every object is the right size.
		freeObject * ptr = head.next;
		if (ptr != &tail) {
			size_t sz = SuperHeap::getSize(ptr);
			while (ptr != &tail) {
			  void * oldptr = ptr;
			  ptr = ptr->next;
			  assert (SuperHeap::getSize(oldptr) >= sz);
			}
		}
		return 1;
	}

  int isEmpty (void) {
	return (head.next == &tail);
  }

  class freeObject {
  public:
	freeObject * prev;
    freeObject * next;
  };

  freeObject head, tail;
  IFDEBUG (int nObjects);
};

#endif
