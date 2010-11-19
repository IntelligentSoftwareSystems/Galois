/* -*- C++ -*- */

#ifndef _FIFODLFREELIST_H_
#define _FIFODLFREELIST_H_

#include <assert.h>

template <class Super>
class FIFODLFreelistHeap : public Super {
public:
  
  FIFODLFreelistHeap (void)
    {
      head.prev = &head;
      head.next = &tail;
      tail.prev = &head;
      tail.next = &tail;
      assert (isEmpty());
    }

  ~FIFODLFreelistHeap (void)
    {
      // Delete everything on the free list.
      freeObject * ptr = head.next;
      while (ptr != &tail) {
	void * oldptr = ptr;
	ptr = ptr->next;
	Super::free (oldptr);
      }
    }

  inline void * malloc (size_t sz) {
    //printf ("flist malloc %d\n", sz);
    // Check the free list first.
    freeObject * ptr = tail.prev;
    if (ptr == &head) {
      assert (isEmpty());
      ptr = (freeObject *) Super::malloc (sz);
    } else {
      ptr->prev->next = &tail;
      tail.prev = ptr->prev;
#if 0
      ptr->prev = NULL;
      ptr->next = NULL;
#endif
    }
    assert (getSize(ptr) >= sz);
    assert (getSize(ptr) >= sizeof(freeObject));
    return (void *) ptr;
  }
  
  inline void free (void * ptr) {
    // Add this object to the free list.
    assert (ptr != NULL);
    freeObject * p = (freeObject *) ptr;
    p->next = head.next;
    p->next->prev = p;
    p->prev = &head;
    head.next = p;
    assert (!isEmpty());
  }

#if 0
  // Returns the entire linked list of freed objects.
  inline void * multimalloc (size_t sz) {
    freeObject * ptr = head.next;
    ptr->prev = NULL;
    tail.prev->next = NULL;
    head.next = &tail;
    tail.prev = &head;
    return ptr;
  }
#endif

  inline static void remove (void * rptr)
    {
      freeObject * p = (freeObject *) rptr;
      assert (p->next != NULL);
      assert (p->prev != NULL);
      p->prev->next = p->next;
      p->next->prev = p->prev;
      p->prev = p->next = NULL;
    }


private:

  int isEmpty (void) {
    return (head.next == &tail);
  }

  class freeObject {
  public:
    freeObject * prev;
    freeObject * next;
  };

  freeObject head, tail;
};

#endif
