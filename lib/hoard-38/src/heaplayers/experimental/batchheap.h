/* -*- C++ -*- */

#ifndef _BATCHHEAP_H_
#define _BATCHHEAP_H_

#include <assert.h>


template <class SuperHeap>
class MultiMalloc : public SuperHeap {
public:
	// Sets ptr to a list of up to num objects of size sz
	// and returns how many free objects ptr points to.
	int multimalloc (int num, size_t sz, void *& ptr)
	{
		int i = 0;
		ptr = (freeObject *) SuperHeap::malloc (sz);
		freeObject * p = (freeObject *) ptr;
		if (ptr != NULL) {
			for (i = 1; i < num; i++) {
				p->next = (freeObject *) SuperHeap::malloc (sz);
				if (p->next == NULL)
					break;
				p = p->next;
			}
			p->next = NULL;
		}
		return i;
	}

	// Frees all num items pointed to by ptr
	// and sets ptr to NULL.
	void multifree (int num, void *& ptr)
	{
		freeObject * p;
		freeObject * prev = (freeObject *) ptr;
		for (int i = 0; i < num; i++) {
			p = prev->next;
			SuperHeap::free (prev);
			prev = p;
		}
		ptr = NULL;
	}

private:

  class freeObject {
  public:
    freeObject * next;
  };


};


template <int BatchNumber, class SuperHeap>
class BatchHeap : public SuperHeap {
public:

	BatchHeap (void)
		: nObjects (0)
	{
		freeList[0] = NULL;
		freeList[1] = NULL;
	}


	~BatchHeap (void) {
		if (nObjects <= BatchNumber) {
			SuperHeap::multifree (nObjects, (void *&) freeList[0]);
		} else {
			SuperHeap::multifree (BatchNumber, (void *&) freeList[0]);
			SuperHeap::multifree (nObjects - BatchNumber, (void *&) freeList[1]);
		}
	}

	inline void * malloc (size_t sz) {
		if (nObjects == 0) {
			// Obtain BatchNumber objects if we're out.
			nObjects = SuperHeap::multimalloc (BatchNumber, sz, (void *&) freeList[0]);
		}
		assert (nObjects >= 1);
		freeObject * ptr;
		if (nObjects > BatchNumber) {
			freeObject *& head = freeList[1];
			ptr = head;
			nObjects--;
			head = head->next;
			return (void *) ptr;
		} else {
			freeObject *& head = freeList[0];
			ptr = head;
			nObjects--;
			head = head->next;
			return (void *) ptr;
		}
	}

	inline void free (void * ptr) {
		if (nObjects <= BatchNumber) {
			freeObject *& head = freeList[0];
			((freeObject *) ptr)->next = head;
			head = (freeObject *) ptr;
		} else {
			freeObject *& head = freeList[1];
			((freeObject *) ptr)->next = head;
			head = (freeObject *) ptr;
		}
		nObjects++;
		if (nObjects == 2 * BatchNumber) {
			// Free half of them.
			assert (freeList[1] != NULL);
			SuperHeap::multifree (BatchNumber, (void *&) freeList[1]);
		}
	}

private:

  class freeObject {
  public:
    freeObject * next;
  };

  int nObjects;

  // The first free list holds the first BatchNumber objects,
  // while the second free list holds the rest.
  freeObject * freeList[2];
};

#endif
