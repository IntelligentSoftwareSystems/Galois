#ifndef _SPLITHEAP_H_
#define _SPLITHEAP_H_

template <class SuperHeap>
class SplitHeap : public SuperHeap {
public:
  inline void * malloc (const size_t sz)
  {
    void * ptr = SuperHeap::malloc (sz);
    if (ptr != NULL) {
		markInUse (ptr);
		size_t oldSize = getSize(ptr);
		if (oldSize <= sz + sizeof(double)) {
			return ptr;
		} else {
			void * splitPiece = split (ptr, sz);
			if (splitPiece != NULL) {
				// printf ("split %d into %d and %d\n", oldSize, getSize(ptr), getSize(splitPiece));
				markFree (splitPiece);
				SuperHeap::free (splitPiece);
			}
		}
	}
    return ptr;
  }
};

#endif
