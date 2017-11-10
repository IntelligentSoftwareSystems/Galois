#include "galois/runtime/SyncStructures.h"
#include "galois/AtomicWrapper.h"
#include "galois/ArrayWrapper.h"

#define LATENT_VECTOR_SIZE 20

typedef galois::CopyableArray<double, LATENT_VECTOR_SIZE> ArrTy;
typedef galois::CopyableArray<galois::CopyableAtomic<double>, LATENT_VECTOR_SIZE> ArrAtomicTy;
//GALOIS_SYNC_STRUCTURE_REDUCE_SET(updates, unsigned int);
//GALOIS_SYNC_STRUCTURE_REDUCE_SET(edge_offset, unsigned int);
GALOIS_SYNC_STRUCTURE_REDUCE_SET(residual_latent_vector, ArrAtomicTy);
GALOIS_SYNC_STRUCTURE_REDUCE_PAIR_WISE_ADD_ARRAY(residual_latent_vector, ArrAtomicTy);
//GALOIS_SYNC_STRUCTURE_REDUCE_PAIR_WISE_AVG_ARRAY(residual_latent_vector, ArrAtomicTy);
GALOIS_SYNC_STRUCTURE_REDUCE_SET(latent_vector, ArrTy);
GALOIS_SYNC_STRUCTURE_REDUCE_PAIR_WISE_AVG_ARRAY(latent_vector, ArrTy);

//GALOIS_SYNC_STRUCTURE_BROADCAST(updates, unsigned int);
//GALOIS_SYNC_STRUCTURE_BROADCAST(edge_offset, unsigned int);
GALOIS_SYNC_STRUCTURE_BROADCAST(residual_latent_vector, ArrAtomicTy);
GALOIS_SYNC_STRUCTURE_BROADCAST(latent_vector, ArrTy);
