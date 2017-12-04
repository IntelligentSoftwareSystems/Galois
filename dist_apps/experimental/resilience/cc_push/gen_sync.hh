#include "galois/runtime/SyncStructures.h"

GALOIS_SYNC_STRUCTURE_BROADCAST(comp_current, uint32_t);
GALOIS_SYNC_STRUCTURE_REDUCE_SET(comp_current, uint32_t);
GALOIS_SYNC_STRUCTURE_REDUCE_MIN(comp_current, uint32_t);
GALOIS_SYNC_STRUCTURE_BITSET(comp_current);
