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
// APSP synchronization
////////////////////////////////////////////////////////////////////////////////

struct APSPReduce {
  using ValTy = galois::TupleOfThree<uint32_t, uint32_t, ShortPathType>;

  static ValTy extract(uint32_t node_id, struct NodeData& node) {
    uint32_t indexToGet = node.roundIndexToSend;

    uint32_t a;
    uint32_t b;
    ShortPathType c;

    a = indexToGet;
    if (indexToGet != infinity) {
      // get min distance and # shortest paths
      b = node.sourceData[indexToGet].minDistance;
      c = node.sourceData[indexToGet].shortPathCount;
    } else {
      // no-op
      b = infinity;
      c = 0;
    }

    return ValTy(a, b, c);
  }

  static bool extract_batch(unsigned, uint8_t*, size_t*,
                            DataCommMode*) { return false; }

  static bool extract_batch(unsigned, uint8_t*) { return false; }

  static bool extract_reset_batch(unsigned, uint8_t*, size_t*,
                                  DataCommMode*) { return false; }

  static bool extract_reset_batch(unsigned, uint8_t*) { return false; }

  static bool reduce(uint32_t node_id, struct NodeData& node, ValTy y) {
    uint32_t rIndex = y.first;

    if (rIndex != infinity) {
      uint32_t rDistance = y.second;
      ShortPathType rNumPaths = y.third;

      // do updates based on received numbers
      uint32_t old = galois::min(node.sourceData[rIndex].minDistance, rDistance);

      // reset shortest paths if min dist changed (i.e. don't add to it)
      if (old > rDistance) {
        node.dTree.setDistance(rIndex, old, rDistance);
        assert(rNumPaths != 0);
        node.sourceData[rIndex].shortPathCount = rNumPaths;
      } else if (old == rDistance) {
        // add to short path
        node.sourceData[rIndex].shortPathCount += rNumPaths;
      }

      // if received distance is smaller than current candidate for sending, send
      // it out instead (if tie breaker wins i.e. lower in position)
      if (node.roundIndexToSend == infinity ||
          (node.sourceData[rIndex].minDistance <
            node.sourceData[node.roundIndexToSend].minDistance)) {
          node.roundIndexToSend = rIndex;
      } else if (node.sourceData[rIndex].minDistance ==
                 node.sourceData[node.roundIndexToSend].minDistance) {
        if (rIndex < node.roundIndexToSend) {
          node.roundIndexToSend = rIndex;
        }
      }

      // return true: if it received a message for some node, then that
      // node on a mirror needs to get the most updated value (i.e. value on
      // master)
      return true;
    }

    return false;
  }

  static bool reduce_batch(unsigned, uint8_t*, DataCommMode) { return false; }

  static bool reduce_mirror_batch(unsigned, uint8_t*, DataCommMode) { return false; }

  // reset the number of shortest paths (the master will now have it)
  static void reset(uint32_t node_id, struct NodeData &node) {
    if (node.roundIndexToSend != infinity) {
      node.sourceData[node.roundIndexToSend].shortPathCount = 0;
    }
  }

  static void setVal(uint32_t node_id, struct NodeData & node, ValTy y) {
    uint32_t rIndex = y.first;
    if (rIndex != infinity) {
      uint32_t rDistance = y.second;
      ShortPathType rNumPaths = y.third;

      // values from master are canonical ones for this round
      node.roundIndexToSend = rIndex;
      uint32_t oldDistance = node.sourceData[rIndex].minDistance;
      node.sourceData[rIndex].minDistance = rDistance;
      node.sourceData[rIndex].shortPathCount = rNumPaths;
      node.dTree.setDistance(rIndex, oldDistance, rDistance);
    }
  }

  static bool setVal_batch(unsigned, uint8_t*, DataCommMode) { return false; }
};

////////////////////////////////////////////////////////////////////////////////

struct DependencyReduce {
  using ValTy = std::pair<uint32_t, float>;

  static ValTy extract(uint32_t node_id, struct NodeData& node) {
    uint32_t indexToGet = node.roundIndexToSend;
    float thing;
    if (indexToGet != infinity) {
      thing = node.sourceData[indexToGet].dependencyValue;
    } else {
      thing = 0;
    }

    return ValTy(indexToGet, thing);
  }

  static bool extract_batch(unsigned, uint8_t*, size_t*,
                            DataCommMode*) { return false; }

  static bool extract_batch(unsigned, uint8_t*) { return false; }

  static bool extract_reset_batch(unsigned, uint8_t*, size_t*,
                                  DataCommMode*) { return false; }

  static bool extract_reset_batch(unsigned, uint8_t*) { return false; }

  static bool reduce(uint32_t node_id, struct NodeData& node, ValTy y) {
    uint32_t rIndex = y.first;

    if (rIndex != infinity) {
      if (node.roundIndexToSend != rIndex) {
        galois::gError(node.roundIndexToSend, " ", rIndex);
      }
      assert(node.roundIndexToSend == rIndex);

      float rToAdd = y.second;
      galois::atomicAdd(node.sourceData[rIndex].dependencyValue, rToAdd);
      return true;
    }

    return false;
  }

  static bool reduce_batch(unsigned, uint8_t*, DataCommMode) { return false; }

  static bool reduce_mirror_batch(unsigned, uint8_t*, DataCommMode) { return false; }

  // reset the number of shortest paths (the master will now have it)
  static void reset(uint32_t node_id, struct NodeData &node) {
    if (node.roundIndexToSend != infinity) {
      node.sourceData[node.roundIndexToSend].dependencyValue = 0;
    }
  }

  static void setVal(uint32_t node_id, struct NodeData & node, ValTy y) {
    uint32_t rIndex = y.first;
    if (rIndex != infinity) {
      float rDep = y.second;
      assert(node.roundIndexToSend == rIndex);
      node.sourceData[rIndex].dependencyValue = rDep;
    }
  }

  static bool setVal_batch(unsigned, uint8_t*, DataCommMode) { return false; }
};

////////////////////////////////////////////////////////////////////////////////
// Bitsets
////////////////////////////////////////////////////////////////////////////////

GALOIS_SYNC_STRUCTURE_BITSET(minDistances);
GALOIS_SYNC_STRUCTURE_BITSET(dependency);
