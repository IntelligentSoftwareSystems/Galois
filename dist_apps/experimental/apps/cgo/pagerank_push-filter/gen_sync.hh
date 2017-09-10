#include "Galois/Runtime/sync_structures.h"

GALOIS_SYNC_STRUCTURE_REDUCE_ADD(nout, unsigned int);
GALOIS_SYNC_STRUCTURE_BROADCAST(nout, unsigned int);
GALOIS_SYNC_STRUCTURE_BITSET(nout);

GALOIS_SYNC_STRUCTURE_REDUCE_SET(residual, float);
GALOIS_SYNC_STRUCTURE_REDUCE_ADD(residual, float);
GALOIS_SYNC_STRUCTURE_BROADCAST(residual, float);

#if __OPT_VERSION__ >= 3
GALOIS_SYNC_STRUCTURE_BITSET(residual);
#endif
#if __OPT_VERSION__ >= 5
FieldFlags Flags_nout;
FieldFlags Flags_residual;
#endif
