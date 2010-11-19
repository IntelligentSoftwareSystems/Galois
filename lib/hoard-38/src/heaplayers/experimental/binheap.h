/* -*- C++ -*- */

#ifndef _BINHEAP_H_
#define _BINHEAP_H_

#include <stdio.h>


template <int Bins[], int NumBins, class Super>
class BinHeap {
public:
  
  inline void * malloc (size_t sz) {
    // Find bin.
    int bin = findBin (sz);
    void * ptr = myHeaps[bin].malloc (sz);
    return ptr;
  }
  
  inline void free (void * ptr) {
    size_t sz = Super::size (ptr);
    int bin = findBin (sz);
    myHeaps[bin].free (ptr);
  }

private:

  inline int findBin (size_t sz) {
    int i;
    for (i = 0; i < NumBins; i++) {
      if (Bins[i] >= sz) {
	break;
      }
    }
    return i;
  }

  Super myHeaps[NumBins + 1];

};

#endif
