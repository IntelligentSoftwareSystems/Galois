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

////////////////////////////////////////////////////////////////////////////
// ToAdd
////////////////////////////////////////////////////////////////////////////

GALOIS_SYNC_STRUCTURE_REDUCE_ADD(to_add, uint32_t);
GALOIS_SYNC_STRUCTURE_REDUCE_SET(to_add, uint32_t);
GALOIS_SYNC_STRUCTURE_BROADCAST(to_add, uint32_t);

////////////////////////////////////////////////////////////////////////////
// ToAddFloat
////////////////////////////////////////////////////////////////////////////

GALOIS_SYNC_STRUCTURE_REDUCE_ADD(to_add_float, float);
GALOIS_SYNC_STRUCTURE_REDUCE_SET(to_add_float, float);
GALOIS_SYNC_STRUCTURE_BROADCAST(to_add_float, float);

////////////////////////////////////////////////////////////////////////////
// # short paths
////////////////////////////////////////////////////////////////////////////

GALOIS_SYNC_STRUCTURE_REDUCE_SET(num_shortest_paths, uint32_t);
GALOIS_SYNC_STRUCTURE_BROADCAST(num_shortest_paths, uint32_t);

////////////////////////////////////////////////////////////////////////////
// Succ
////////////////////////////////////////////////////////////////////////////

GALOIS_SYNC_STRUCTURE_REDUCE_ADD(num_successors, uint32_t);
GALOIS_SYNC_STRUCTURE_REDUCE_SET(num_successors, uint32_t);
GALOIS_SYNC_STRUCTURE_BROADCAST(num_successors, uint32_t);

////////////////////////////////////////////////////////////////////////////
// Pred
////////////////////////////////////////////////////////////////////////////

GALOIS_SYNC_STRUCTURE_REDUCE_ADD(num_predecessors, uint32_t);
GALOIS_SYNC_STRUCTURE_REDUCE_SET(num_predecessors, uint32_t);
GALOIS_SYNC_STRUCTURE_BROADCAST(num_predecessors, uint32_t);

////////////////////////////////////////////////////////////////////////////
// Trim
////////////////////////////////////////////////////////////////////////////

GALOIS_SYNC_STRUCTURE_REDUCE_ADD(trim, uint32_t);
GALOIS_SYNC_STRUCTURE_REDUCE_SET(trim, uint32_t);
GALOIS_SYNC_STRUCTURE_BROADCAST(trim, uint32_t);

GALOIS_SYNC_STRUCTURE_REDUCE_ADD(trim2, uint32_t);
GALOIS_SYNC_STRUCTURE_REDUCE_SET(trim2, uint32_t);
GALOIS_SYNC_STRUCTURE_BROADCAST(trim2, uint32_t);

////////////////////////////////////////////////////////////////////////////
// Current Lengths
////////////////////////////////////////////////////////////////////////////

GALOIS_SYNC_STRUCTURE_REDUCE_MIN(current_length, uint32_t);
GALOIS_SYNC_STRUCTURE_REDUCE_SET(current_length, uint32_t);
GALOIS_SYNC_STRUCTURE_BROADCAST(current_length, uint32_t);

////////////////////////////////////////////////////////////////////////////
// Flag
////////////////////////////////////////////////////////////////////////////

GALOIS_SYNC_STRUCTURE_REDUCE_SET(propogation_flag, uint8_t);
GALOIS_SYNC_STRUCTURE_BROADCAST(propogation_flag, uint8_t);

////////////////////////////////////////////////////////////////////////////
// Dependency
////////////////////////////////////////////////////////////////////////////

GALOIS_SYNC_STRUCTURE_REDUCE_SET(dependency, float);
GALOIS_SYNC_STRUCTURE_BROADCAST(dependency, float);

////////////////////////////////////////////////////////////////////////////////
// Bitsets
////////////////////////////////////////////////////////////////////////////////

#if __OPT_VERSION__ >= 3
GALOIS_SYNC_STRUCTURE_BITSET(current_length);
GALOIS_SYNC_STRUCTURE_BITSET(to_add);
GALOIS_SYNC_STRUCTURE_BITSET(to_add_float);
GALOIS_SYNC_STRUCTURE_BITSET(num_successors);
GALOIS_SYNC_STRUCTURE_BITSET(num_predecessors);
GALOIS_SYNC_STRUCTURE_BITSET(trim);
GALOIS_SYNC_STRUCTURE_BITSET(trim2);
#endif

#if __OPT_VERSION__ == 5
FieldFlags Flags_current_length;
FieldFlags Flags_num_successors;
FieldFlags Flags_num_predecessors;
FieldFlags Flags_trim;
FieldFlags Flags_trim2;
FieldFlags Flags_to_add;
FieldFlags Flags_to_add_float;
#endif
