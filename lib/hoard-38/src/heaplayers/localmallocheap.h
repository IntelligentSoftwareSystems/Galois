#ifndef _LOCALMALLOCHEAP_H_
#define _LOCALMALLOCHEAP_H_

#include <dlfcn.h>

#include "ansiwrapper.h"
#include "sizeheap.h"
#include "staticheap.h"

extern "C" {
  size_t malloc_usable_size (void *);
  
  typedef void * mallocFunction (size_t);
  typedef void freeFunction (void *);
  typedef size_t msizeFunction (void *);
  
  typedef void exitFunction (int);
  exitFunction * trueExitFunction;
}

namespace HL {

  class LocalMallocHeap {
  public:

    LocalMallocHeap (void)
      : freefn (NULL),
      msizefn (NULL),
      mallocfn (NULL),
      firsttime (true)
      {}

    inline void * malloc (size_t sz) {
      if (firsttime) {

	firsttime = false;

	// We haven't initialized anything yet.
	// Initialize all of the malloc shim functions.

	freefn = (freeFunction *) dlsym (RTLD_NEXT, "free");
	msizefn = (msizeFunction *) dlsym (RTLD_NEXT, "malloc_usable_size");
	trueExitFunction = (exitFunction *) dlsym (RTLD_NEXT, "exit");
	mallocfn = (mallocFunction *) dlsym (RTLD_NEXT, "malloc");

	if (!(freefn && msizefn && trueExitFunction && mallocfn)) {
	  fprintf (stderr, "Serious problem!\n");
	  abort();
	}

	assert (freefn);
	assert (msizefn);
	assert (trueExitFunction);
	assert (mallocfn);

	// Go get some memory from malloc!
	return (*mallocfn)(sz);
      }

      // Now, once we have mallocfn resolved, we can use it.
      // Otherwise, we're still in dlsym-land, and have to use our local heap.

      if (mallocfn) {
	return (*mallocfn)(sz);
      } else {
	void * ptr = localHeap.malloc (sz);
	assert (ptr);
	return ptr;
      }
    }

    inline void free (void * ptr) {
      if (mallocfn) {
	if (localHeap.isValid (ptr)) {
	  // We got a pointer to the temporary allocation buffer.
	  localHeap.free (ptr);
	} else {
	  (*freefn)(ptr);
	}
      }
    }

    inline size_t getSize (void * ptr) {
      if (localHeap.isValid (ptr)) {
	return localHeap.getSize (ptr);
      } else if (mallocfn) {
	return (*msizefn)(ptr);
      } else {
	// This should never happen.
	return 0;
      }
    }

  private:

    bool firsttime;   /// True iff we haven't initialized the shim functions.

    // Shim functions below.

    freeFunction *   freefn;
    msizeFunction *  msizefn;
    mallocFunction * mallocfn;

    /// The local heap (for use while we are in dlsym, installing the
    /// shim functions). Hopefully 64K is enough...

    ANSIWrapper<SizeHeap<StaticHeap<65536> > > localHeap;

  };

}

#endif
