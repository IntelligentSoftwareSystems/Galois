/** Betweeness Centrality (PR) -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
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
 *
 * @section Description
 *
 * Sync structures for PR-BC.
 *
 * @author Loc Hoang <l_hoang@utexas.edu>
 */

////////////////////////////////////////////////////////////////////////////////
// MinDistances
////////////////////////////////////////////////////////////////////////////////

/**
 * Manually defined sync structure for reducing minDistances. Needs to be manual
 * as there is a reset operation that is contingent on changes to the minimum
 * distance.
 */
struct ReducePairwiseMinAndResetDist {
  using ValTy = uint32_t;

  static ValTy extract(uint32_t node_id, const struct NodeData& node, 
                       unsigned vecIndex) {
    return node.minDistances[vecIndex];
  }

  static bool extract_reset_batch(unsigned, unsigned long long int*,
                                  unsigned int*, ValTy*, size_t*,
                                  DataCommMode*) {
    return false;
  }

  static bool extract_reset_batch(unsigned, ValTy*) {
    return false;
  }

  /**
   * Updates all distances with a min reduction.
   *
   * The important thing about this particular reduction is that if the
   * distance changes, then shortestPathToAdd must be set to 0 as it is
   * now invalid.
   */
  static bool reduce(uint32_t node_id, struct NodeData& node, ValTy y,
                     unsigned vecIndex) {
    bool returnVar = false;

    auto& myDistances = node.minDistances;

    uint32_t oldDist = galois::min(myDistances[vecIndex], y);

    // if there's a change, reset the shortestPathsAdd var
    if (oldDist > myDistances[vecIndex]) {
      node.shortestPathToAdd[vecIndex] = 0;
      returnVar = true;
    }

    return returnVar;
  }

  static bool reduce_batch(unsigned, unsigned long long int*, unsigned int *,
                           ValTy*, size_t, DataCommMode) {
    return false;
  }

  /**
   * do nothing for reset
   */
  static void reset(uint32_t node_id, struct NodeData &node, unsigned vecIndex) {
    return;
  }
};

GALOIS_SYNC_STRUCTURE_BROADCAST_VECTOR_SINGLE(minDistances, uint32_t);
GALOIS_SYNC_STRUCTURE_VECTOR_BITSET(minDistances);

////////////////////////////////////////////////////////////////////////////////
// Shortest Path
////////////////////////////////////////////////////////////////////////////////

GALOIS_SYNC_STRUCTURE_REDUCE_PAIR_WISE_ADD_ARRAY_SINGLE(shortestPathToAdd, 
                                                        uint64_t);
GALOIS_SYNC_STRUCTURE_BROADCAST_VECTOR_SINGLE(shortestPathToAdd, uint64_t);
GALOIS_SYNC_STRUCTURE_VECTOR_BITSET(shortestPathToAdd);

////////////////////////////////////////////////////////////////////////////////
// Dependency
////////////////////////////////////////////////////////////////////////////////

GALOIS_SYNC_STRUCTURE_REDUCE_PAIR_WISE_ADD_ARRAY_SINGLE(dependencyToAdd, 
                                                        galois::CopyableAtomic<float>);
GALOIS_SYNC_STRUCTURE_BROADCAST_VECTOR_SINGLE(dependencyToAdd, 
                                              galois::CopyableAtomic<float>);
GALOIS_SYNC_STRUCTURE_VECTOR_BITSET(dependencyToAdd);
