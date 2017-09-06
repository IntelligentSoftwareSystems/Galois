#include "Galois/Runtime/sync_structures.h"

////////////////////////////////////////////////////////////////////////////////
// current_degree
////////////////////////////////////////////////////////////////////////////////

GALOIS_SYNC_STRUCTURE_REDUCE_ADD(current_degree, uint32_t);
GALOIS_SYNC_STRUCTURE_REDUCE_SET(current_degree, uint32_t);
GALOIS_SYNC_STRUCTURE_REDUCE_MIN(current_degree, uint32_t);
GALOIS_SYNC_STRUCTURE_BROADCAST(current_degree, uint32_t);

////////////////////////////////////////////////////////////////////////////////
// trim
////////////////////////////////////////////////////////////////////////////////

GALOIS_SYNC_STRUCTURE_REDUCE_ADD(trim, uint32_t);
GALOIS_SYNC_STRUCTURE_REDUCE_MIN(trim, uint32_t);
GALOIS_SYNC_STRUCTURE_BROADCAST(trim, uint32_t);

// this is included for initialization
GALOIS_SYNC_STRUCTURE_BITSET(current_degree);
#if __OPT_VERSION__ >= 3
GALOIS_SYNC_STRUCTURE_BITSET(trim);
#endif

#if __OPT_VERSION__ == 5
// dummy sync structs that won't even be used
//GALOIS_SYNC_STRUCTURE_REDUCE_SET(flag, uint8_t);
//GALOIS_SYNC_STRUCTURE_BROADCAST(flag, uint8_t);
//FieldFlags Flags_flag;

FieldFlags Flags_current_degree;
FieldFlags Flags_trim;
#endif
