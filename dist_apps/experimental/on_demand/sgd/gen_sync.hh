#include "galois/runtime/SyncStructures.h"
#include "galois/AtomicWrapper.h"
#include "galois/ArrayWrapper.h"

#define LATENT_VECTOR_SIZE 20

typedef galois::CopyableArray<double, LATENT_VECTOR_SIZE> ArrTy;
typedef galois::CopyableArray<galois::CopyableAtomic<double>, LATENT_VECTOR_SIZE> ArrAtomicTy;
typedef std::vector<galois::CopyableAtomic<double>> VecAtomicTy;
typedef std::vector<double> VecTy;

//New vector type
GALOIS_SYNC_STRUCTURE_REDUCE_SET(residual_latent_vector, VecAtomicTy);
GALOIS_SYNC_STRUCTURE_REDUCE_PAIR_WISE_ADD_ARRAY(residual_latent_vector, VecAtomicTy);


//New vector type
GALOIS_SYNC_STRUCTURE_REDUCE_SET(latent_vector, VecTy);
GALOIS_SYNC_STRUCTURE_REDUCE_PAIR_WISE_AVG_ARRAY(latent_vector, VecTy);

//Vector
GALOIS_SYNC_STRUCTURE_BROADCAST(residual_latent_vector, VecAtomicTy);
GALOIS_SYNC_STRUCTURE_BROADCAST(latent_vector, VecTy);

#if __OPT_VERSION__ == 5
galois::runtime::FieldFlags Flags_residual_latent_vector;
#endif
