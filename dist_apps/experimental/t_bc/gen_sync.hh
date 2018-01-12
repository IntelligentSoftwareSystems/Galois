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

/**
 * Manually defined sync structure for reducing minDistances. Needs to be manual
 * as there is a reset operation that is contingent on changes to the minimum
 * distance.
 */
struct ReducePairwiseMinAndResetDist {
  using ValTy = std::vector<uint32_t>;

  static ValTy extract(uint32_t node_id, const struct NodeData& node) {
    //galois::gPrint("extract min\n");
    return node.minDistances;
  }

  static bool extract_reset_batch(unsigned from_id,
                                  unsigned long long int* b,
                                  unsigned int* o,
                                  ValTy* y,
                                  size_t* s,
                                  DataCommMode* data_mode) {
    return false;
  }

  static bool extract_reset_batch(unsigned from_id, ValTy *y) {
    return false;
  }

  /**
   * Updates all distances with a min reduction.
   *
   * The important thing about this particular reduction is that if the
   * distance changes, then shortestPathToAdd must be set to 0 as it is
   * now invalid.
   */
  static bool reduce(uint32_t node_id, struct NodeData& node, ValTy y) {
    bool returnVar = false;

    ValTy& myDistances = node.minDistances;

    for (unsigned i = 0; i < myDistances.size(); i++) {
      uint32_t oldDist = galois::min(myDistances[i], y[i]);

      // if there's a change, reset the shortestPathsAdd var
      if (oldDist > myDistances[i]) {
        node.shortestPathToAdd[i] = 0;
        returnVar = true;
      }
    }

    return returnVar;
  }

  static bool reduce_batch(unsigned from_id,
                           unsigned long long int *b,
                           unsigned int *o,
                           ValTy *y,
                           size_t s,
                           DataCommMode data_mode) {
    return false;
  }

  /**
   * do nothing for reset
   */
  static void reset (uint32_t node_id, struct NodeData &node) {
    return;
  }
};
GALOIS_SYNC_STRUCTURE_BROADCAST(minDistances, std::vector<uint32_t>);
GALOIS_SYNC_STRUCTURE_BITSET(minDistances);

GALOIS_SYNC_STRUCTURE_REDUCE_PAIR_WISE_ADD_ARRAY(shortestPathToAdd, 
                                                 std::vector<uint32_t>);
GALOIS_SYNC_STRUCTURE_BROADCAST(shortestPathToAdd, std::vector<uint32_t>);
GALOIS_SYNC_STRUCTURE_BITSET(shortestPathToAdd);


GALOIS_SYNC_STRUCTURE_REDUCE_PAIR_WISE_ADD_ARRAY(
  dependencyToAdd, 
  std::vector<galois::CopyableAtomic<float>>
);
GALOIS_SYNC_STRUCTURE_BROADCAST(dependencyToAdd, 
                                std::vector<galois::CopyableAtomic<float>>);
GALOIS_SYNC_STRUCTURE_BITSET(dependencyToAdd);
