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

GALOIS_SYNC_STRUCTURE_REDUCE_ADD(to_add, uint64_t);
GALOIS_SYNC_STRUCTURE_REDUCE_SET(to_add, uint64_t);
GALOIS_SYNC_STRUCTURE_BROADCAST(to_add, uint64_t);

////////////////////////////////////////////////////////////////////////////
// ToAddFloat
////////////////////////////////////////////////////////////////////////////

GALOIS_SYNC_STRUCTURE_REDUCE_ADD(to_add_float, float);
GALOIS_SYNC_STRUCTURE_REDUCE_SET(to_add_float, float);
GALOIS_SYNC_STRUCTURE_BROADCAST(to_add_float, float);

////////////////////////////////////////////////////////////////////////////
// # short paths
////////////////////////////////////////////////////////////////////////////

GALOIS_SYNC_STRUCTURE_REDUCE_SET(num_shortest_paths, uint64_t);
GALOIS_SYNC_STRUCTURE_REDUCE_ADD(num_shortest_paths, uint64_t);
GALOIS_SYNC_STRUCTURE_BROADCAST(num_shortest_paths, uint64_t);

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

////////////////////////////////////////////////////////////////////////////
// Current Lengths
////////////////////////////////////////////////////////////////////////////

GALOIS_SYNC_STRUCTURE_REDUCE_MIN(current_length, uint32_t);
GALOIS_SYNC_STRUCTURE_REDUCE_SET(current_length, uint32_t);
GALOIS_SYNC_STRUCTURE_BROADCAST(current_length, uint32_t);

////////////////////////////////////////////////////////////////////////////////
// Old length
////////////////////////////////////////////////////////////////////////////////

GALOIS_SYNC_STRUCTURE_REDUCE_SET(old_length, uint32_t);
GALOIS_SYNC_STRUCTURE_BROADCAST(old_length, uint32_t);

////////////////////////////////////////////////////////////////////////////
// Flag
////////////////////////////////////////////////////////////////////////////

GALOIS_SYNC_STRUCTURE_REDUCE_SET(propagation_flag, uint8_t);
GALOIS_SYNC_STRUCTURE_BROADCAST(propagation_flag, uint8_t);

////////////////////////////////////////////////////////////////////////////
// Dependency
////////////////////////////////////////////////////////////////////////////

GALOIS_SYNC_STRUCTURE_REDUCE_SET(dependency, float);
GALOIS_SYNC_STRUCTURE_REDUCE_ADD(dependency, float);
GALOIS_SYNC_STRUCTURE_BROADCAST(dependency, float);

////////////////////////////////////////////////////////////////////////////////
#if __OPT_VERSION__ >= 3
GALOIS_SYNC_STRUCTURE_BITSET(to_add);
GALOIS_SYNC_STRUCTURE_BITSET(to_add_float);
GALOIS_SYNC_STRUCTURE_BITSET(num_shortest_paths);
GALOIS_SYNC_STRUCTURE_BITSET(num_successors);
GALOIS_SYNC_STRUCTURE_BITSET(num_predecessors);
GALOIS_SYNC_STRUCTURE_BITSET(trim);
GALOIS_SYNC_STRUCTURE_BITSET(current_length);
GALOIS_SYNC_STRUCTURE_BITSET(propagation_flag);
GALOIS_SYNC_STRUCTURE_BITSET(dependency);
#endif


#if __OPT_VERSION__ == 5
galois::runtime::FieldFlags Flags_to_add;
galois::runtime::FieldFlags Flags_to_add_float;
galois::runtime::FieldFlags Flags_num_shortest_paths;
galois::runtime::FieldFlags Flags_num_successors;
galois::runtime::FieldFlags Flags_num_predecessors;
galois::runtime::FieldFlags Flags_trim;
galois::runtime::FieldFlags Flags_current_length;
galois::runtime::FieldFlags Flags_propagation_flag;
galois::runtime::FieldFlags Flags_dependency;
galois::runtime::FieldFlags Flags_old_length;
galois::runtime::FieldFlags Flags_betweeness_centrality;
#endif
