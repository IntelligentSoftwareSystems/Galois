/* -*- C++ -*- */

/**
 * @file bitindex.h
 *
 * Bit-access operations, including msb and lsb.
 * @author Emery Berger (emery@cs.umass.edu)
*/

// lsb is due to Leiserson, Prokop, Randall (MIT).
// msb is also a fast implementation of floor(lg_2).

#ifndef _BITINDEX_H_
#define _BITINDEX_H_

#include <assert.h>

namespace HL {

class BitIndex {
private:
  
  enum { debruijn32 = 0x077CB531UL };
  static int index32[32];
  static unsigned long lgtable[16];
  static unsigned long on[32];
  static unsigned long off[32];

  void setup (void);

public:

  BitIndex (void);
  ~BitIndex (void) {}

  // Set bit_index in b (to 1).
  static void set (unsigned long &b, int index)
    {
      assert (index >= 0);
      assert (index < 32);
      b |= on[index];
    }

  // Reset bit_index in b (set it to 0).
  static void reset (unsigned long &b, int index)
    {
      assert (index >= 0);
      assert (index < 32);
      b &= off[index];
    }

  // Find the least-significant bit.
  static int lsb (unsigned long b)
    {
      //#if 0
#if 0 // i386
      // Intel x86 code.
      register unsigned long index = 0;
      if (b > 0) {
	asm ("bsfl %1, %0" : "=r" (index) : "r" (b));
      }
      return index;
#else
      b &= (unsigned long) -((signed long) b);
      b *= debruijn32;
      b >>= 27;
      return index32[b];
#endif
    }

  // Find the most-significant bit.
  static int msb (unsigned long b)
    {
#if 0 // i386
      // Intel x86 code.
      register unsigned long index = 0;
      if (b > 0) {
	asm ("bsrl %1, %0" : "=r" (index) : "r" (b));
      }
      return index;
#else
      int l = 0;
      // b < 2^32
      if (b >= 65536) {
	l += 16;
	b >>= 16;
      }
      // b < 2^16
      if (b >= 256) {
	l += 8;
	b >>= 8;
      }
      // b < 2^8
      if (b >= 16) {
	l += 4;
	b >>= 4;
      }
      return l + lgtable[b];
#endif
    }

  
};

};

#endif
