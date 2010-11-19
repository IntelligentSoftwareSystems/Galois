#ifndef _SCALABLELEAHEAP_H_
#define _SCALABLELEAHEAP_H_

#include "heaplayers.h"

namespace ScalableHeapNS {

	template <int NumHeaps, class SuperHeap>
	class GlobalHeapWrapper : public SuperHeap {
	public:
		inline void * malloc (size_t sz) {
			void * ptr = SuperHeap::malloc (sz);
			if (ptr != NULL) {
				assert (!isFree(ptr));
			}
			return ptr;
		}
		inline void free (void * ptr) {
			// Set this object's heap to an unassigned heap (NumHeaps),
			// then free it.
			setHeap (ptr, NumHeaps); // This should be an unassigned heap number.
			assert (getHeap(ptr) == NumHeaps);
			setPrevHeap(getNext(ptr), NumHeaps);
			assert (getPrevHeap(getNext(ptr)) == NumHeaps);
			SuperHeap::free (ptr);
		}
	private:
		inline int remove (void *);
	};


  template <int Threshold, class Heap1, class Heap2>
  class TryHeap : public Heap2 {
  public:
	  TryHeap (void)
		  : reserved (0)
	  {}

    inline void * malloc (size_t sz) {
      void * ptr = heap1.malloc (sz);
      if (ptr == NULL) {
#if 1
		// Get a big chunk.
		size_t chunkSize = (Threshold / 2) > sz ? (Threshold / 2) : sz;
		ptr = Heap2::malloc (chunkSize);
		if (ptr == NULL) {
			return NULL;
		}
		// Split it.
		void * splitPiece = CoalesceHeap<Heap2>::split (ptr, sz);
		assert (splitPiece != ptr);
		assert (!isFree(ptr));
		// Put the split piece on heap 1.
		if (splitPiece != NULL) {
			reserved += getSize(splitPiece);
			heap1.free (splitPiece);
		}
#else
		ptr = Heap2::malloc (sz);
#endif
	  } else {
		  reserved -= getSize(ptr);
	  }
//	  assert (getHeap(ptr) == tid);
      return ptr;
    }
    inline void free (void * ptr) {
	  reserved += getSize(ptr);
      heap1.free (ptr);
      if (reserved > Threshold) {
		// We've crossed the threshold.
		// Free objects from heap 1 and give them to heap 2.
		// Start big.
		size_t sz = Threshold / 2;
		while ((sz > sizeof(double)) && (reserved > Threshold / 2)) {
		  void * p = NULL;
		  while ((p == NULL) && (sz >= sizeof(double))) {
			p = heap1.malloc (sz);
			if (p == NULL) {
			  sz >>= 1;
			}
		  }
		  if (p != NULL) {
			  reserved -= getSize(p);
			  Heap2::free (p);
		  }
		}
      }
    }

	
  private:
	inline int remove (void * ptr);
#if 0
	{
		assert (0);
		abort();
	}
#endif

    Heap1 heap1;
	int reserved;
  };


  template <int NumHeaps, int MmapThreshold, class BaseNullHeap, class BaseHeap>
  class SmallHeap : public
	  ScalableHeapNS::TryHeap<MmapThreshold,
        MarkThreadHeap<NumHeaps, BaseNullHeap>,
        MarkThreadHeap<NumHeaps, GlobalHeapWrapper<NumHeaps, BaseHeap> > > {};

  template <int NumHeaps, int MmapThreshold, class BaseNullHeap, class BaseHeap>
  class MTHeap :
    public PHOThreadHeap<NumHeaps,
		     LockedHeap<SmallHeap<NumHeaps, MmapThreshold, BaseNullHeap, BaseHeap> > > {};

};

  
template <int NumHeaps, class BaseNullHeap, class BaseHeap, class Mmap>
class ScalableHeap :
public SelectMmapHeap<128 * 1024,
         ScalableHeapNS::MTHeap<NumHeaps, 128 * 1024, BaseNullHeap, BaseHeap>,
		    LockedHeap<Mmap> > {};
#endif
