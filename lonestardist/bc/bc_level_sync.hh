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
// # short paths
////////////////////////////////////////////////////////////////////////////

struct Reduce_add_num_shortest_paths {
  using ValTy = ShortPathType;

  static ValTy extract(uint32_t node_id, struct NodeData& node) {
    // only send in round before use; else send 0
    if (node.current_length == globalRoundNumber + 1) {
      return node.num_shortest_paths;
    } else {
      return (ValTy)0;
    }
  }

  static bool extract_batch(unsigned, uint8_t*, size_t*,
                                  DataCommMode*) { return false; }

  static bool extract_batch(unsigned, uint8_t*) { return false; }

  static bool extract_reset_batch(unsigned, uint8_t*, size_t*,
                                  DataCommMode*) { return false; }

  static bool extract_reset_batch(unsigned, uint8_t*) { return false; }

  static bool reduce(uint32_t node_id, struct NodeData& node, ValTy y) {
    galois::add(node.num_shortest_paths, y);
    return true;
  }

  static bool reduce_batch(unsigned, uint8_t*, DataCommMode) { return false; }

  static bool reduce_mirror_batch(unsigned, uint8_t*, DataCommMode) { return false; }

  // reset the number of shortest paths (the master will now have it)
  static void reset(uint32_t node_id, struct NodeData &node) {
    node.num_shortest_paths = (ValTy)0;
  }

  static void setVal(uint32_t node_id, struct NodeData& node, ValTy y) {
    galois::set(node.num_shortest_paths, y);
  }

  static bool setVal_batch(unsigned, uint8_t*, DataCommMode) { return false; }
};
// used for middle sync only
GALOIS_SYNC_STRUCTURE_REDUCE_SET(num_shortest_paths, ShortPathType);
GALOIS_SYNC_STRUCTURE_BITSET(num_shortest_paths);

////////////////////////////////////////////////////////////////////////////
// Current Lengths
////////////////////////////////////////////////////////////////////////////

GALOIS_SYNC_STRUCTURE_REDUCE_MIN(current_length, uint32_t);
GALOIS_SYNC_STRUCTURE_BITSET(current_length);

////////////////////////////////////////////////////////////////////////////
// Dependency
////////////////////////////////////////////////////////////////////////////

struct Reduce_add_dependency {
  using ValTy = float;

  static ValTy extract(uint32_t node_id, struct NodeData& node) {
    // only send in round before use; else send 0
    if (node.current_length == backRoundCount) {
      return node.dependency;
    } else {
      return (ValTy)0;
    }
  }

  static bool extract_batch(unsigned, uint8_t*, size_t*,
                                  DataCommMode*) { return false; }

  static bool extract_batch(unsigned, uint8_t*) { return false; }

  static bool extract_reset_batch(unsigned, uint8_t*, size_t*,
                                  DataCommMode*) { return false; }

  static bool extract_reset_batch(unsigned, uint8_t*) { return false; }

  static bool reduce(uint32_t node_id, struct NodeData& node, ValTy y) {
    galois::add(node.dependency, y);
    return true;
  }

  static bool reduce_batch(unsigned, uint8_t*, DataCommMode) { return false; }

  static bool reduce_mirror_batch(unsigned, uint8_t*, DataCommMode) { return false; }

  // reset the number of shortest paths (the master will now have it)
  static void reset(uint32_t node_id, struct NodeData &node) {
    node.dependency = (ValTy)0;
  }

  static void setVal(uint32_t node_id, struct NodeData& node, ValTy y) {
    galois::set(node.dependency, y);
  }

  static bool setVal_batch(unsigned, uint8_t*, DataCommMode) { return false; }
};
GALOIS_SYNC_STRUCTURE_BITSET(dependency);
