/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 */

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
