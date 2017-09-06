#include "Galois/Runtime/sync_structures.h"

GALOIS_SYNC_STRUCTURE_BROADCAST(dist_current, unsigned int);
GALOIS_SYNC_STRUCTURE_REDUCE_SET(dist_current, unsigned int);
GALOIS_SYNC_STRUCTURE_REDUCE_MIN(dist_current, unsigned int);
GALOIS_SYNC_STRUCTURE_BITSET(dist_current);
