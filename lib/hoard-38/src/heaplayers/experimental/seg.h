// -*- C++ -*-

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

#ifndef _SEG_H_
#define _SEG_H_

#include <assert.h>

#include "sassert.h"
#include "reaplet.h"

namespace HL {

template <int ReapletSize,
	  class SizeClassComputer,
	  class TopHeap>
class Seg : public TopHeap {
public:

  enum { Alignment = sizeof(double) };

private:

  Seg (void)
  {
    sassert <(sizeof(Reaplet<ReapletSize>) == ReapletSize)> verifyReapletSize;
  }
  
  class SanityChecker {
  public:
    SanityChecker (Seg * s)
      : _s (s)
      {
	_s->sanityCheck();
      }
    ~SanityChecker (void) {
      _s->sanityCheck();
    }
  private:
    Seg * _s;
  };


#if 1
  enum { TopAlignment = TopHeap::Alignment };

  class VerifyAlignment :
    public sassert <TopAlignment % ReapletSize == 0> {};
#endif

public:

  __forceinline Seg (void) {
    for (int i = 0; i < SizeClassComputer::NUM_BINS; i++) {
      available[i] = 0;
      full[i] = 0;
    }
  }

  ~Seg (void) {
    clear();
  }

  __forceinline void * malloc (size_t sz) {
    SanityChecker c (this);
    assert (sz <= SizeClassComputer::BIG_OBJECT);
    const int SizeClass = SizeClassComputer::getSizeClass (sz);
    // Try to use one of our reaplets to satisfy the request.
    Reaplet<ReapletSize> * const xs = available[SizeClass];
    if (xs) {
      //      assert (!xs->isFull());
      // We've got one. Now try to get some memory from it.
      void * const ptr = xs->malloc (SizeClassComputer::bins[SizeClass]);
      //      printf ("allocated: %x\n", ptr);
      if (ptr) {
	// Success! We're out of here.
	return ptr;
      } else {
	// It's full.
	//	printf ("Full - adios.\n");
	removeFromAvailable (xs, SizeClass);
      }
    }
    //    printf ("slow!\n");
    // We have to go get more memory.
    return slowPathGetMoreMemory (sz);
  }


  __forceinline static size_t getSize (void * ptr) {
    return (enclosingReaplet (ptr))->getSize();
  }

  __forceinline void free (void * ptr) {
    SanityChecker c (this);
    // printf ("seg free: r = %x\n", r);
    Reaplet<ReapletSize> * r = enclosingReaplet (ptr);
    //    printf ("freeing %x to %x\n", ptr, r);
    //    if (sz <= SizeClassComputer::BIG_OBJECT) {
    // It's a small object. Free it.
    bool wasFull = r->isFull(); //  || !r->getAmountAllocated()) {
    r->free (ptr);
    if (wasFull) {
      moveFromFullToAvailable (r);
    }
  }

  void clear (void) {
    SanityChecker c (this);
    for (int i = 0; i < SizeClassComputer::NUM_BINS; i++) {
      clearOut (full[i]);
      clearOut (available[i]);
    }
  }

private:

  __forceinline void * slowPathGetMoreMemory (const size_t sz) {
    //    printf ("slowpathgetmorememory\n");
    SanityChecker c (this);
    if (sz <= SizeClassComputer::BIG_OBJECT) {
      const int SizeClass = SizeClassComputer::getSizeClass (sz);
      do {
	// printf ("looking for more memory.\n");
	Reaplet<ReapletSize> * xs = available[SizeClass];
	// Look for an available reaplet.
	if (xs != 0) {
	  assert (!xs->isFull());
	  // We've got one. Now try to get some memory from it.
	  void * ptr = xs->malloc (SizeClassComputer::bins[SizeClass]);
	  if (ptr) {
	    // Success!
	    //	    printf ("amt free = %d\n", enclosingReaplet(ptr)->getAmountFree());
	    // printf ("success: found more memory.\n");
	    return ptr;
	  } else {
	    //	    printf ("Adios!\n");
	    // This reaplet is out of memory.
	    // Remove it from the available list.
	    removeFromAvailable (xs, SizeClass);
	    // Put it on the full list.
	    putOnFullList (xs, SizeClass);
	  }
	} else {
	  // The current bin is empty - make another reaplet.
	  int r = insertNewReaplet (SizeClass);
	  if (!r) {
	    return 0;
	  }
	}
      } while (1);
    } else {
      return makeBigObject (sz);
    }
  }

  __forceinline void sanityCheck (void) {
#if !defined(NDEBUG)
    // FIX ME
    //#error "whoa dude. slowing things down."
    Reaplet<ReapletSize> * q;
    for (int i = 0; i < SizeClassComputer::NUM_BINS; i++) {
      // Visit the available list.
      q = available[i];
      if (q) {
	assert (q->getPrev() == 0);
      }
      while (q) {
	if (q->isFull()) {
	  // printf ("allocated = %d\n", q->getAmountAllocated());
	}
	assert (!q->isFull());
	q = (Reaplet<ReapletSize> *) q->getNext();
      }
      // Now visit the full list.
      q = full[i];
      if (q) {
	assert (q->getPrev() == 0);
      }
      while (q) {
	assert (q->isFull());
	q = (Reaplet<ReapletSize> *) q->getNext();
      }
    }
#endif
  }

  __forceinline
  void putOnFullList (Reaplet<ReapletSize> * xs,
		      const int SizeClass) {
    // printf ("putOnFullList %d\n", SizeClass);
    SanityChecker c (this);
    //    assert (!xs->isFull());
    xs->setNext (full[SizeClass]);
    xs->setPrev (0);
    if (full[SizeClass]) {
      full[SizeClass]->setPrev (xs);
    }
    full[SizeClass] = xs;
    //    xs->markFull();
  }

  __forceinline void clearOut (Reaplet<ReapletSize> *& list) {
    SanityChecker c (this);
    Reaplet<ReapletSize> * q;
    q = list;
    while (q) {
      Reaplet<ReapletSize> * c = q;
      q = (Reaplet<ReapletSize> *) q->getNext();
      TopHeap::free (c);
    }
    list = 0;
  }

  __forceinline void removeFromFullList (Reaplet<ReapletSize> * r,
					 const size_t sz) {
    //    printf ("removeFromFullList\n");
    // prev <-> r <-> next =>
    // prev <-> next
    SanityChecker c (this);
    //    assert (r->isFull());
    // Remove from full list.
    Reaplet<ReapletSize> * prev, * next;
    prev = (Reaplet<ReapletSize> *) r->getPrev();
    next = (Reaplet<ReapletSize> *) r->getNext();
    assert (prev != r);
    assert (next != r);
    if (prev) {
      prev->setNext (next);
    }
    if (next) {
      next->setPrev (prev);
    }
    const int SizeClass = SizeClassComputer::getSizeClass (sz);
    if (r == full[SizeClass]) {
      assert (prev == 0);
      full[SizeClass] = next;
    }
    assert (full[SizeClass] != r);
    // r->setPrev (0);
    // r->setNext (0);
    //    r->markNotFull();
  }

  __forceinline
  void moveFromFullToAvailable (Reaplet<ReapletSize> * r) {
    SanityChecker c (this);
    const size_t sz = r->getSize();
    removeFromFullList (r, sz);
    // Put it on the available list.
    addToAvailableList (r, sz);
    assert (r->getPrev() == 0);
  }

  __forceinline
  void removeAndFreeReaplet (Reaplet<ReapletSize> * r)
  {
    SanityChecker c (this);
    assert (!r->isFull());
    const size_t sz = r->getSize();
    Reaplet<ReapletSize> * n
      = (Reaplet<ReapletSize> *) r->getNext();
    Reaplet<ReapletSize> * p
      = (Reaplet<ReapletSize> *) r->getPrev();
    if (n) {
      n->setPrev (p);
    }
    if (p) {
      p->setNext (n);
    }
    int sizeclass = SizeClassComputer::getSizeClass (sz);
    if (available[sizeclass] == r) {
      available[sizeclass] = n;
    }
    //r->setPrev ((Reaplet<ReapletSize> *) 1);
    //r->setNext ((Reaplet<ReapletSize> *) 1);
    //r->clear();
    TopHeap::free (r);
    //	printf ("reset %d\n", sz);
  }

  __forceinline
  int insertNewReaplet (int SizeClass)
  {
    // printf ("insertNewReaplet\n");
    SanityChecker c (this);
    void * m = TopHeap::malloc (ReapletSize);
    // NOTE: TopHeap *must* add a reaplet prefix.
    // Ensure m is aligned properly.
    assert (((size_t) m & (ReapletSize - 1)) == 0);
    if (m == 0) {
      // printf ("NO MORE MEMORY!\n");
      // No more memory.
      return 0;
    }
    // Make a new Reaplet for this size class.
    size_t ssz = SizeClassComputer::getClassSize (SizeClass);
    
    // printf ("new reaplet!: ssz = %d\n", ssz);
    available[SizeClass] = new (m) Reaplet<ReapletSize> (ssz);
    available[SizeClass]->clear();
    return 1;
  }

protected:

  __forceinline
  void addToAvailableList (Reaplet<ReapletSize> * r,
			   size_t sz)
  {
    SanityChecker c (this);
    //    assert (!r->isFull());
    const int SizeClass = SizeClassComputer::getSizeClass (sz);
    Reaplet<ReapletSize> * head = available[SizeClass];

    assert (head != r);

    // r
    // => r <-> head
    
    r->setPrev (0);
    r->setNext (head);
    if (head) {
      head->setPrev (r);
    }
    available[SizeClass] = r;
  }

private:

  NO_INLINE void removeFromAvailable (Reaplet<ReapletSize> * xs,
					  const int SizeClass) {
    //    printf ("removefromavailable %d.\n", SizeClass);
    // The reaplet xs is out of memory.
    // Remove it from the linked list.
    // Note that it is the head of the linked list.
    //	printf ("Removing REAPLET %x\n", xs);
    assert (xs->isFull());
    assert (xs->getPrev() == 0);
    Reaplet<ReapletSize> * n
      = (Reaplet<ReapletSize> *) xs->getNext();
    if (n) {
      // printf ("setprev.\n");
      n->setPrev (0);
      assert (!n->isFull());
    }
    available[SizeClass] = n;
    SanityChecker c (this);
    // xs->setPrev ((Reaplet<ReapletSize> *) 1);
    // xs->setNext ((Reaplet<ReapletSize> *) 1);
    assert (n == 0 || (n->getPrev() == 0));
  }

  __forceinline static Reaplet<ReapletSize> * enclosingReaplet (void * ptr) {
    // Find the enclosing reaplet for this pointer.
    Reaplet<ReapletSize> * r
      = (Reaplet<ReapletSize> *) (((size_t) ptr) & ~(ReapletSize - 1));
    return r;
  }

#if 1
  NO_INLINE
  void * makeBigObject (size_t sz) {
    //    printf ("makeBigObject\n");
    SanityChecker c (this);
    //    printf ("make big object: %d\n", sz);
    // Allocate big objects from the top heap. These must be
    // ReapletSize aligned.
    assert (sz > SizeClassComputer::BIG_OBJECT);
    ReapletBase * m
      = (ReapletBase *) TopHeap::malloc (sz + sizeof(ReapletBase));
    // printf ("m = %x, size = %d\n", m, ReapletSize);
    assert (((size_t) m & (ReapletSize - 1)) == 0);
    // Put a reaplet base object as a header in the beginning. We
    // need this for size information.
    if (m == 0) {
      return 0;
    }
    new (m) ReapletBase (sz, (char *) (m + 1)); 
    return (void *) (m + 1);
  }
#endif

  Reaplet<ReapletSize> * available[SizeClassComputer::NUM_BINS];
  Reaplet<ReapletSize> * full[SizeClassComputer::NUM_BINS];
};

};

#endif
