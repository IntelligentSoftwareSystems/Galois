// -*- C++ -*-

#if !defined(_BINS64K_H_)
#define _BINS64K_H_

#include <cstdlib>
#include <assert.h>

#include "bins.h"
#include "sassert.h"

namespace HL {

template <class Header>
class bins<Header, 65536> {
public:

  bins (void)
  {
#ifndef NDEBUG
    for (int i = sizeof(double); i < BIG_OBJECT; i++) {
      int sc = getSizeClass(i);
      assert (getClassSize(sc) >= i);
      assert (getClassSize(sc-1) < i);
      assert (getSizeClass(getClassSize(sc)) == sc);
    }
#endif
  }

  enum { BIG_OBJECT = 65536 / 2 - sizeof(Header) };
  
  enum { NUM_BINS = 55,
	 NUM_LOOKUP = 508 };

  static const size_t _bins[NUM_BINS];
  static const int    _sizeclasses[NUM_LOOKUP];

  static inline int getSizeClass (size_t sz) {
    assert (sz <= _bins[NUM_BINS-1]);
    sz = (sz < sizeof(double)) ? sizeof(double) : sz;
    if (sz <= 80) {
      return (int) ((sz - 1) >> 3);
    } else {
      return slowGetSizeClass (sz);
    }
  }

  static inline size_t getClassSize (const int i) {
    assert (i >= 0);
    assert (i < NUM_BINS);
    assert (getSizeClass(_bins[i]) == i);
    return _bins[i];
  }

private:

  static inline int slowGetSizeClass (size_t sz) {
    int ind = 0;
    while (sz > _bins[ind]) {
      ind++;
      assert (ind < NUM_BINS);
    }
    return ind;
  }

  sassert<(BIG_OBJECT > 0)> verifyHeaderSize;
};

}


template <class Header>
const size_t HL::bins<Header, 65536>::_bins[NUM_BINS] =
  { 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 96, 112, 128, 152, 176, 208, 248, 296, 352, 416, 496, 592, 704, 840, 1008, 1208, 1448, 1736, 2080, 2496, 2992, 3584, 4096 - sizeof(Header), 4912, 5888, 7064, 8192 - sizeof(Header), 9824, 11784, 12288 - sizeof(Header), 14744, 16384 - sizeof(Header), 19656, 20480 - sizeof(Header), 24576 - sizeof(Header), 28672 - sizeof(Header), 32768 - sizeof(Header), 36864 - sizeof(Header), 40960 - sizeof(Header), 45056 - sizeof(Header), 49152 - sizeof(Header), 53248 - sizeof(Header), 57344 - sizeof(Header), 61440 - sizeof(Header), 65536 - sizeof(Header) };

template <class Header>
const int HL::bins<Header, 65536>::_sizeclasses[NUM_LOOKUP] =
  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 11, 11, 12, 12, 13, 13, 13, 14, 14, 14, 15, 15, 15, 15, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32 };

#endif
