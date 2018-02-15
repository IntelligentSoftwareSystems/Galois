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
  #ifndef _VECTOR_SYNC_
  using ValTy = uint32_t;
  #else
  using ValTy = galois::gstl::Vector<uint32_t>;
  #endif

  #ifndef _VECTOR_SYNC_
  static ValTy extract(uint32_t node_id, const struct NodeData& node, 
                       unsigned vecIndex) {
    return node.minDistances[vecIndex];
  }
  #else
  static ValTy extract(uint32_t node_id, const struct NodeData& node) {
    return node.minDistances;
  }
  #endif

  static bool extract_reset_batch(unsigned, unsigned long int*,
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
  #ifndef _VECTOR_SYNC_
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
  #else
  static bool reduce(uint32_t node_id, struct NodeData& node, ValTy y) {
    bool returnVar = false;

    auto& myDistances = node.minDistances;

    for (unsigned i = 0; i < myDistances.size(); i++) {
      uint32_t oldDist = galois::min(myDistances[i], y[i]);
      if (oldDist > myDistances[i]) {
        node.shortestPathToAdd[i] = 0;
        returnVar = true;
      }
    }

    return returnVar;
  }
  #endif


  static bool reduce_batch(unsigned, unsigned long int*, unsigned int *,
                           ValTy*, size_t, DataCommMode) {
    return false;
  }

  /**
   * do nothing for reset
   */
  #ifndef _VECTOR_SYNC_
  static void reset(uint32_t node_id, struct NodeData &node, unsigned vecIndex) {
    return;
  }
  #else
  static void reset(uint32_t node_id, struct NodeData &node) {
    return;
  }
  #endif
};

#ifndef _VECTOR_SYNC_
GALOIS_SYNC_STRUCTURE_BROADCAST_VECTOR_SINGLE(minDistances, uint32_t);
#else
GALOIS_SYNC_STRUCTURE_BROADCAST(minDistances, galois::gstl::Vector<uint32_t>);
#endif

////////////////////////////////////////////////////////////////////////////////
// Shortest Path
////////////////////////////////////////////////////////////////////////////////

#ifndef _VECTOR_SYNC_
GALOIS_SYNC_STRUCTURE_REDUCE_PAIR_WISE_ADD_ARRAY_SINGLE(shortestPathToAdd, 
                                                        uint64_t);
GALOIS_SYNC_STRUCTURE_BROADCAST_VECTOR_SINGLE(shortestPathToAdd, uint64_t);
#else
GALOIS_SYNC_STRUCTURE_REDUCE_PAIR_WISE_ADD_ARRAY(shortestPathToAdd, 
                                                 galois::gstl::Vector<uint64_t>);
GALOIS_SYNC_STRUCTURE_BROADCAST(shortestPathToAdd, 
                                galois::gstl::Vector<uint64_t>);
#endif

////////////////////////////////////////////////////////////////////////////////
// Dependency
////////////////////////////////////////////////////////////////////////////////

#ifndef _VECTOR_SYNC_
GALOIS_SYNC_STRUCTURE_REDUCE_PAIR_WISE_ADD_ARRAY_SINGLE(dependencyToAdd, 
                                                        galois::CopyableAtomic<float>);
GALOIS_SYNC_STRUCTURE_BROADCAST_VECTOR_SINGLE(dependencyToAdd, 
                                              galois::CopyableAtomic<float>);
#else
GALOIS_SYNC_STRUCTURE_REDUCE_PAIR_WISE_ADD_ARRAY(dependencyToAdd,
  galois::gstl::Vector<galois::CopyableAtomic<float>>);
GALOIS_SYNC_STRUCTURE_BROADCAST(dependencyToAdd, 
                           galois::gstl::Vector<galois::CopyableAtomic<float>>);
#endif

////////////////////////////////////////////////////////////////////////////////
// Bitsets
////////////////////////////////////////////////////////////////////////////////

#ifndef _VECTOR_SYNC_
GALOIS_SYNC_STRUCTURE_VECTOR_BITSET(minDistances);
GALOIS_SYNC_STRUCTURE_VECTOR_BITSET(shortestPathToAdd);
GALOIS_SYNC_STRUCTURE_VECTOR_BITSET(dependencyToAdd);
#else
GALOIS_SYNC_STRUCTURE_BITSET(minDistances);
GALOIS_SYNC_STRUCTURE_BITSET(shortestPathToAdd);
GALOIS_SYNC_STRUCTURE_BITSET(dependencyToAdd);
#endif
