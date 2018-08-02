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

////////////////////////////////////////////////////////////////////////////////
// Forward Phase
////////////////////////////////////////////////////////////////////////////////

struct ReduceAPSP {
  using ValTy = std::pair<uint32_t, ShortPathType>;

  static ValTy extract(uint32_t node_id, const struct NodeData& node,
                       unsigned vecIndex) {
    return ValTy(node.minDistances[vecIndex], node.pathAccumulator[vecIndex]);
  }


  static bool extract_reset_batch(unsigned, unsigned long int*,
                                  unsigned int*, ValTy*, size_t*,
                                  DataCommMode*) {
    return false;
  }

  static bool extract_reset_batch(unsigned, ValTy*) {
    return false;
  }

  static bool reduce(uint32_t node_id, struct NodeData& node, ValTy y,
                     unsigned vecIndex) {
    bool returnVar = false;
    auto& myDistances = node.minDistances;
    uint32_t oldDist = galois::min(myDistances[vecIndex], y.first);
    // if there's a change, reset the shortestPathsAdd var +
    if (oldDist > myDistances[vecIndex]) {
      node.dTree.setDistance(vecIndex, oldDist, y.first);
      node.shortestPathNumbers[vecIndex] = 0;
      node.pathAccumulator[vecIndex] = y.second;
      returnVar = true;
    } else if (oldDist == myDistances[vecIndex]) {
      // no change to distance => add to path accumulator
      node.pathAccumulator[vecIndex] += y.second;
      returnVar = true;
    }
    return returnVar;
  }



  static bool reduce_batch(unsigned, unsigned long int*, unsigned int *,
                           ValTy*, size_t, DataCommMode) {
    return false;
  }

  /**
   * reset accumulator
   */
  static void reset(uint32_t node_id, struct NodeData &node, unsigned vecIndex) {
    node.pathAccumulator[vecIndex] = 0;
  }
};

struct BroadcastAPSP {
  using ValTy = std::pair<uint32_t, ShortPathType>;

  static ValTy extract(uint32_t node_id, const struct NodeData& node,
                       unsigned vecIndex) {
    return ValTy(node.minDistances[vecIndex], node.pathAccumulator[vecIndex]);
  }

  // defined to make compiler not complain
  static ValTy extract(uint32_t node_id, const struct NodeData & node) {
    GALOIS_DIE("Execution shouldn't get here this function needs an index arg\n");
    return ValTy(0, 0);
  }


  static bool extract_batch(unsigned, uint64_t*, unsigned int*, ValTy*, size_t*,
                            DataCommMode*) {
    return false;
  }

  static bool extract_batch(unsigned, ValTy*) {
    return false;
  }

  // if min distance is changed by the broadcast, then shortest path to add
  // becomes obsolete/incorrect, so it must be changed to 0
  static void setVal(uint32_t node_id, struct NodeData & node, ValTy y,
                     unsigned vecIndex) {
    assert(node.minDistances[vecIndex] >= y.first);

    // reset short path count if necessary
    if (node.minDistances[vecIndex] != y.first) {
      node.shortestPathNumbers[vecIndex] = 0;
    }

    uint32_t oldDistance = node.minDistances[vecIndex];
    node.minDistances[vecIndex] = y.first;
    node.dTree.setDistance(vecIndex, oldDistance, y.first);
    node.pathAccumulator[vecIndex] = y.second;
  }

  // defined so compiler won't complain
  static void setVal(uint32_t node_id, struct NodeData & node, ValTy y) {
    GALOIS_DIE("Execution shouldn't get here; needs index arg\n");
  }

  static bool setVal_batch(unsigned, uint64_t*, unsigned int*, ValTy*,
                           size_t, DataCommMode) {
    return false;
  }
};

struct BitsetAPSP {
  static unsigned numBitsets() { return bitset_APSP.size(); }
  static constexpr bool is_vector_bitset() { return true; }
  static constexpr bool is_valid() { return true; }
  static galois::DynamicBitSet& get(unsigned i) {
    return bitset_APSP[i];
  }
  static void reset_range(size_t begin, size_t end) {
    for (unsigned i = 0; i < bitset_APSP.size(); i++) {
      bitset_APSP[i].reset(begin, end);
    }
  }
};

////////////////////////////////////////////////////////////////////////////////
// Backphase
////////////////////////////////////////////////////////////////////////////////

GALOIS_SYNC_STRUCTURE_REDUCE_PAIR_WISE_ADD_ARRAY_SINGLE(depAccumulator,
                                                        galois::CopyableAtomic<float>);
GALOIS_SYNC_STRUCTURE_BROADCAST_VECTOR_SINGLE(depAccumulator,
                                              galois::CopyableAtomic<float>);
struct BitsetDep {
  static unsigned numBitsets() { return bitset_depAccumulator.size(); }
  static constexpr bool is_vector_bitset() { return true; }
  static constexpr bool is_valid() { return true; }
  static galois::DynamicBitSet& get(unsigned i) {
    return bitset_depAccumulator[i];
  }
  static void reset_range(size_t begin, size_t end) {
    for (unsigned i = 0; i < bitset_depAccumulator.size(); i++) {
      bitset_depAccumulator[i].reset(begin, end);
    }
  }
};
