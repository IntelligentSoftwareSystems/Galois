#ifndef _KINGSMODHEAP_H_
#define _KINGSMODHEAP_H_

#include "segheap.h"

/* KingsMod (segregated fits) allocator */

namespace KingsMod {

  inline size_t class2Size (int i);

  inline int pow2Class (size_t sz) {
    static   size_t sizeTable[] = {8UL, 16UL, 24UL, 32UL, 40UL, 48UL, 56UL, 72UL, 80UL, 96UL, 120UL, 144UL, 168UL, 200UL, 240UL, 288UL, 344UL, 416UL, 496UL, 592UL, 712UL, 856UL, 1024UL, 1232UL, 1472UL, 1768UL, 2120UL, 2544UL, 3048UL, 3664UL};
	int c = 0;
	while (c < 30 && sz < sizeTable[c]) {
	  c++;
	}
	return c;
  }

  inline size_t class2Size (int i) {
    assert (i >= 0);
    assert (i < 30);
    static size_t sizeTable[] = {8UL, 16UL, 24UL, 32UL, 40UL, 48UL, 56UL, 72UL, 80UL, 96UL, 120UL, 144UL, 168UL, 200UL, 240UL, 288UL, 344UL, 416UL, 496UL, 592UL, 712UL, 856UL, 1024UL, 1232UL, 1472UL, 1768UL, 2120UL, 2544UL, 3048UL, 3664UL};
    return sizeTable[i];
  }

};


template <class PerClassHeap>
class KingsModHeap : public SegHeap<29, KingsMod::pow2Class, KingsMod::class2Size, PerClassHeap, PerClassHeap> {};

#endif
