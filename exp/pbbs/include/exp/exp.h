
#ifndef EXP_H
#define EXP_H

#include <cstdlib>

namespace Exp {

extern __thread unsigned TID;
extern unsigned nextID;

// NB(ddn): Not "DRF" for DMP but this is okay if we don't interpret the value
// itself, i.e., only use this as a identifier for thread-local data.
static inline unsigned get_tid() {
  unsigned x = TID;
  if (x & 1)
    return x >> 1;
  x = __sync_fetch_and_add(&nextID, 1);
  TID = (x << 1) | 1;
  return x;
}

unsigned get_num_threads();
int get_num_rounds();

}

#endif
