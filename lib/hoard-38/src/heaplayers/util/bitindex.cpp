#include "bitindex.h"

// Definitions of static members of BitIndex.

int BitIndex::index32[32];
unsigned long BitIndex::on[32];
unsigned long BitIndex::off[32];
unsigned long BitIndex::lgtable[16];

// Implementations.

BitIndex::BitIndex (void)
{
  setup();
}


void BitIndex::setup (void)
{
  int t[] = { 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3};
  int i;
  for (i = 0; i < 32; i++) {
    index32[ (debruijn32 << i) >> 27 ] = i;
  }
  for (i = 0; i < 16; i++) {
    lgtable[i] = t[i];
  }
  for (i = 0; i < 32; i++) {
    on[i] = 1 << i;
    off[i] = ~on[i];
  }
}



// One instance of BitIndex so everything will work.

static BitIndex _hiddenBitIndex;
