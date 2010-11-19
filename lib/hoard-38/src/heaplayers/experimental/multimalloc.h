#ifndef _MULTIMALLOC_H_
#define _MULTIMALLOC_H_

#include <assert.h>

#include <fstream.h>


#include "dynarray.h"
#include "stack.h"


template <class SuperHeap>
class MultiMalloc : public SuperHeap {
public:


	MultiMalloc (void) {
		//f.open ("multimalloc.log");
	}

	~MultiMalloc (void) {
		//f.close ();
	}

	// Sets ptr to a list of up to num objects of size sz
	// and returns how many free objects ptr points to.
	int multimalloc (int num, size_t sz, void *& ptr)
	{
		//f << "multimalloc from " << this << endl << flush;
		// printf ("multimalloc\n");
		assert (num > 0);
		if (stk.empty()) {
#if 0
			//f << "malloc " << num << ", " << sz << "\n" << flush;
			ptr = SuperHeap::malloc (sz);
			assert (size(ptr) >= sz);
			assert (sz >= sizeof(FreeObject *));
			FreeObject * p = (FreeObject *) ptr;
			for (int i = 1; i < num; i++) {
				p->next = (FreeObject *) SuperHeap::malloc (sz);
				p = p->next;
				assert (size(p) >= sz);
			}
			p->next = NULL;
#else
			size_t sz1 = align(sz + sizeof(double));
			ptr = SuperHeap::malloc (num * sz1); // Allow room for size & thread info.
			ptr = (double *) ptr + 1;
			FreeObject * p = (FreeObject *) ptr;
			for (int i = 0; i < num - 1; i++) {
				size(p) = sz;
				FreeObject * next = (FreeObject *) (((unsigned long) p) + sz1);
				p->next = next;
				p = p->next;
			}
			size(p) = sz;
			p->next = NULL;
			assert (size(p) + ((unsigned long) p) <= ((unsigned long) ptr + num * sz1));
#endif

#ifndef NDEBUG
			p = (FreeObject *) ptr;
			int c = 0;
			while (p != NULL) {
				c++;
				p = p->next;
			}
			assert (c == num);
#endif
			return num;
		} else {
			// Pop off some memory from the stack.
			assert (!stk.empty());
			np v = stk.pop();
			// JUST FOR CHECKING --
			assert (v.num == num);
			ptr = v.ptr;
			assert (v.num > 0);
			assert (size(ptr) >= sz);
			//f << "multimalloc " << v.num << ", " << sz << "\n" << flush;
			return v.num;
		}
	}

	// Frees all num items pointed to by ptr.
	void multifree (int num, void * ptr)
	{
		// printf ("multifree\n");
		np v;
		//f << "multifree " << num << ", size = " << size(ptr) << "\n" << flush;
		v.num = num;
		v.ptr = ptr;
		stk.push (v);
	}

private:

	MultiMalloc (const MultiMalloc&);
	MultiMalloc& operator=(const MultiMalloc&);

	class np {
	public:
		int num;
		void * ptr;
	};

	class FreeObject {
	public:
		FreeObject * next;
	};

	void * malloc (size_t);
	void free (void *);

	Stack<np, DynArray<np> > stk;
	//ofstream f;

	static inline size_t align (size_t sz) {
		return (sz + (sizeof(double) - 1)) & ~(sizeof(double) - 1);
	}

};


#endif
