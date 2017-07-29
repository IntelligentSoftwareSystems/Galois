#include "Galois/Runtime/sync_structures.h"

GALOIS_SYNC_STRUCTURE_BROADCAST(comp_current, unsigned int);
GALOIS_SYNC_STRUCTURE_REDUCE_SET(comp_current, unsigned int);
GALOIS_SYNC_STRUCTURE_REDUCE_MIN(comp_current, unsigned int);
GALOIS_SYNC_STRUCTURE_BITSET(comp_current);
