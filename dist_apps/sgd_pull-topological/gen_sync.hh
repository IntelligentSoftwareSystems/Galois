#include "Galois/Runtime/sync_structures.h"

#define LATENT_VECTOR_SIZE 20

typedef std::array<double, LATENT_VECTOR_SIZE> ArrTy;
GALOIS_SYNC_STRUCTURE_REDUCE_SET(updates, unsigned int);
GALOIS_SYNC_STRUCTURE_REDUCE_SET(edge_offset, unsigned int);
GALOIS_SYNC_STRUCTURE_REDUCE_SET(latent_vector, ArrTy);
GALOIS_SYNC_STRUCTURE_REDUCE_PAIR_WISE_AVG_ARRAY(latent_vector, ArrTy);

GALOIS_SYNC_STRUCTURE_BROADCAST(updates, unsigned int);
GALOIS_SYNC_STRUCTURE_BROADCAST(edge_offset, unsigned int);
GALOIS_SYNC_STRUCTURE_BROADCAST(latent_vector, ArrTy);
