#include "galois/runtime/SyncStructures.h"
#include "galois/AtomicWrapper.h"
#include "galois/ArrayWrapper.h"

#define LATENT_VECTOR_SIZE 20

typedef galois::CopyableArray<double, LATENT_VECTOR_SIZE> ArrTy;
typedef galois::CopyableArray<galois::CopyableAtomic<double>, LATENT_VECTOR_SIZE> ArrAtomicTy;
typedef std::vector<galois::CopyableAtomic<double>> VecAtomicTy;
typedef std::vector<double> VecTy;

//GALOIS_SYNC_STRUCTURE_REDUCE_SET(residual_latent_vector, ArrAtomicTy);
//GALOIS_SYNC_STRUCTURE_REDUCE_PAIR_WISE_ADD_ARRAY(residual_latent_vector, ArrAtomicTy);

//New vector type
GALOIS_SYNC_STRUCTURE_REDUCE_SET(residual_latent_vector, VecAtomicTy);
GALOIS_SYNC_STRUCTURE_REDUCE_PAIR_WISE_ADD_ARRAY(residual_latent_vector, VecAtomicTy);
GALOIS_SYNC_STRUCTURE_BITSET(residual_latent_vector);


//GALOIS_SYNC_STRUCTURE_REDUCE_PAIR_WISE_AVG_ARRAY(residual_latent_vector, ArrAtomicTy);
//GALOIS_SYNC_STRUCTURE_REDUCE_SET(latent_vector, ArrTy);
//GALOIS_SYNC_STRUCTURE_REDUCE_PAIR_WISE_AVG_ARRAY(latent_vector, ArrTy);

//New vector type
GALOIS_SYNC_STRUCTURE_REDUCE_SET(latent_vector, VecTy);
GALOIS_SYNC_STRUCTURE_REDUCE_PAIR_WISE_AVG_ARRAY(latent_vector, VecTy);
GALOIS_SYNC_STRUCTURE_BITSET(latent_vector);

//Old arrays
//GALOIS_SYNC_STRUCTURE_BROADCAST(residual_latent_vector, ArrAtomicTy);
//GALOIS_SYNC_STRUCTURE_BROADCAST(latent_vector, ArrTy);

//Vector
GALOIS_SYNC_STRUCTURE_BROADCAST(residual_latent_vector, VecAtomicTy);
GALOIS_SYNC_STRUCTURE_BROADCAST(latent_vector, VecTy);

