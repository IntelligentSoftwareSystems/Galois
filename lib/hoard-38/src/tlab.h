// -*- C++ -*-

/**
 *
 * @class  ThreadLocalAllocationBuffer
 * @author Emery Berger <http://www.cs.umass.edu/~emery>
 * @brief  An allocator, meant to be used for thread-local allocation.
 */

#include "dllist.h"
#include "array.h"

namespace Hoard {

template <int NumBins,
	  int (*getSizeClass) (size_t),
	  size_t (*getClassSize) (const int),
	  int LargestObject,
	  int LocalHeapThreshold,
	  class SuperblockType,
	  int SuperblockSize,
	  class ParentHeap>

class ThreadLocalAllocationBuffer {

public:

  ThreadLocalAllocationBuffer (ParentHeap * parent)
    : _parentHeap (parent),
      _localHeapBytes (0)
  {
  }

  ~ThreadLocalAllocationBuffer (void) {
    clear();
  }

  inline static size_t getSize (void * ptr) {
    return getSuperblock(ptr)->getSize (ptr);
  }

  inline void * malloc (size_t sz) {
#if 1
    if (sz < 2 * sizeof(size_t)) {
      sz = 2 * sizeof(size_t);
    }
    sz = align (sz);
#endif
    // Get memory from the local heap,
    // and deduct that amount from the local heap bytes counter.
    if (sz <= LargestObject) {
      int c = getSizeClass (sz);
      void * ptr = _localHeap(c).get();
      if (ptr) {
	assert (_localHeapBytes >= sz);
	_localHeapBytes -= sz; 
	assert (getSize(ptr) >= sz);
	return ptr;
      }
    }

    // No more local memory (for this size, at least).
    // Now get the memory from our parent.
    void * ptr = _parentHeap->malloc (sz);
    return ptr;
  }


  inline void free (void * ptr) {
    if (!ptr) {
      return;
    }
    const SuperblockType * s = getSuperblock (ptr);
    // If this isn't a valid superblock, just return.

    if (s->isValidSuperblock()) {

      ptr = s->normalize (ptr);
      const size_t sz = s->getObjectSize ();

      if ((sz <= LargestObject) && (sz + _localHeapBytes <= LocalHeapThreshold)) {
	// Free small objects locally, unless we are out of space.

	assert (getSize(ptr) >= sizeof(HL::DLList::Entry *));
	int c = getSizeClass (sz);

	_localHeap(c).insert ((HL::DLList::Entry *) ptr);
	_localHeapBytes += sz;
	  
      } else {

	// Free it to the parent.
	_parentHeap->free (ptr);
      }

    } else {
      // Illegal pointer.
    }
  }

  void clear (void) {
    // Free every object to the 'parent' heap.
    int i = NumBins - 1;
    while ((_localHeapBytes > 0) && (i >= 0)) {
      const size_t sz = getClassSize (i);
      while (!_localHeap(i).isEmpty()) {
	HL::DLList::Entry * e = _localHeap(i).get();
	_parentHeap->free (e);
	_localHeapBytes -= sz;
      }
      i--;
    }
  }

  static inline SuperblockType * getSuperblock (void * ptr) {
    return SuperblockType::getSuperblock (ptr);
  }

private:

  inline static size_t align (size_t sz) {
    return (sz + (sizeof(double) - 1)) & ~(sizeof(double) - 1);
  }
  

  // Disable assignment and copying.

  ThreadLocalAllocationBuffer (const ThreadLocalAllocationBuffer&);
  ThreadLocalAllocationBuffer& operator=(const ThreadLocalAllocationBuffer&);

  /// This heap's 'parent' (where to go for more memory).
  ParentHeap * _parentHeap;

  /// The number of bytes we currently have on this thread.
  size_t _localHeapBytes;

  /// The local heap itself.
  Array<NumBins, HL::DLList> _localHeap;

};

}
