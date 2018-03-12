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
// APSP synchronization
////////////////////////////////////////////////////////////////////////////////

struct APSPReduce {
  // TODO change to triple (constant size)
  using ValTy = galois::gstl::Vector<uint64_t>;

  static ValTy extract(uint32_t node_id, struct NodeData& node) {
    uint32_t indexToGet = node.roundIndexToSend;

    // declare vector to be sent
    ValTy vecToSend;
    vecToSend.emplace_back(indexToGet);
    if (indexToGet != infinity) {
      // get min distance and # shortest paths
      vecToSend.emplace_back(node.minDistances[indexToGet]);
      vecToSend.emplace_back(node.shortestPathNumbers[indexToGet]);
      node.shortestPathNumbers[indexToGet] = 0;
    } else {
      // no-op
      vecToSend.emplace_back(infinity);
      vecToSend.emplace_back(0);
    }

    return vecToSend;
  }

  static bool extract_reset_batch(unsigned, unsigned long int*,
                                  unsigned int*, ValTy*, size_t*,
                                  DataCommMode*) { return false; }

  static bool extract_reset_batch(unsigned, ValTy*) { return false; }

  static bool reduce(uint32_t node_id, struct NodeData& node, ValTy y) {
    uint32_t rIndex = y[0];

    if (rIndex != infinity) {
      uint32_t rDistance = y[1];
      uint64_t rNumPaths = y[2];

      // do updates based on received numbers
      uint32_t old = galois::min(node.minDistances[rIndex], rDistance);

      // reset shortest paths if min dist changed (i.e. don't add to it)
      if (old > rDistance) {
        assert(rNumPaths != 0);
        node.shortestPathNumbers[rIndex] = rNumPaths;
      } else if (old == rDistance) {
        uint64_t oldS = node.shortestPathNumbers[rIndex];

        // add to short path
        node.shortestPathNumbers[rIndex] += rNumPaths;

        // overflow
        if (oldS > node.shortestPathNumbers[rIndex]) {
          galois::gDebug("Overflow detected in sync; capping at max uint64_t");
          galois::gDebug(oldS, " ", rNumPaths);
          node.shortestPathNumbers[rIndex] = 
            std::numeric_limits<uint64_t>::max();
        }
      }

      // if received distance is smaller than current candidate for sending, send
      // it out instead (if tie breaker wins i.e. lower in position)
      if (node.roundIndexToSend == infinity || 
            (node.minDistances[rIndex] < node.minDistances[node.roundIndexToSend])) {
          assert(!node.sentFlag[rIndex]);
          node.roundIndexToSend = rIndex;
      } else if (node.minDistances[rIndex] == 
                 node.minDistances[node.roundIndexToSend]) {
        if (rIndex < node.roundIndexToSend) {
          assert(!node.sentFlag[rIndex]);
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

  static bool reduce_batch(unsigned, unsigned long int*, unsigned int *,
                           ValTy*, size_t, DataCommMode) { return false; }

  // reset the number of shortest paths (the master will now have it)
  static void reset(uint32_t node_id, struct NodeData &node) { 
    if (node.roundIndexToSend != infinity) {
      node.shortestPathNumbers[node.roundIndexToSend] = 0;
    }
  }
};

struct APSPBroadcast {
  using ValTy = galois::gstl::Vector<uint64_t>;

  static ValTy extract(uint32_t node_id, const struct NodeData & node) {
    uint32_t indexToGet = node.roundIndexToSend;

    // declare vector to be sent
    ValTy vecToSend;
    vecToSend.emplace_back(indexToGet);

    if (indexToGet != infinity) {
      // get min distance and # shortest paths
      vecToSend.emplace_back(node.minDistances[indexToGet]);
      vecToSend.emplace_back(node.shortestPathNumbers[indexToGet]);
      assert(vecToSend[2] != 0); // should not send out 0 for # paths
    } else {
      vecToSend.emplace_back(infinity);
      vecToSend.emplace_back(0);
    }

    return vecToSend;
  }

  static bool extract_batch(unsigned, uint64_t*, unsigned int*, ValTy*, size_t*,
                            DataCommMode*) { return false; }

  static bool extract_batch(unsigned, ValTy*) { return false; }

  static void setVal(uint32_t node_id, struct NodeData & node, ValTy y) {
    uint32_t rIndex = y[0];
    if (rIndex != infinity) {
      uint32_t rDistance = y[1];
      uint32_t rNumPaths = y[2];

      // values from master are canonical ones for this round
      node.roundIndexToSend = rIndex;
      node.minDistances[rIndex] = rDistance;
      node.shortestPathNumbers[rIndex] = rNumPaths;
    }
  }

  static bool setVal_batch(unsigned, uint64_t*, unsigned int*, ValTy*,
                           size_t, DataCommMode) { return false; }
};

struct DependencyReduce {
  using ValTy = std::pair<uint32_t, float>;

  static ValTy extract(uint32_t node_id, struct NodeData& node) {
    uint32_t indexToGet = node.roundIndexToSend;
    float thing;
    if (indexToGet != infinity) {
      thing = node.dependencyValues[indexToGet];
    } else {
      thing = 0;
    }

    return ValTy(indexToGet, thing);
  }

  static bool extract_reset_batch(unsigned, unsigned long int*,
                                  unsigned int*, ValTy*, size_t*,
                                  DataCommMode*) { return false; }

  static bool extract_reset_batch(unsigned, ValTy*) { return false; }

  static bool reduce(uint32_t node_id, struct NodeData& node, ValTy y) {
    uint32_t rIndex = y.first;

    if (rIndex != infinity) {
      if (node.roundIndexToSend != rIndex) {
        galois::gPrint(node.roundIndexToSend, " ", rIndex, "\n");
      }
      assert(node.roundIndexToSend == rIndex);

      float rToAdd = y.second;
      galois::atomicAdd(node.dependencyValues[rIndex], rToAdd);
      return true;
    }

    return false;
  }

  static bool reduce_batch(unsigned, unsigned long int*, unsigned int *,
                           ValTy*, size_t, DataCommMode) { return false; }

  // reset the number of shortest paths (the master will now have it)
  static void reset(uint32_t node_id, struct NodeData &node) { 
    if (node.roundIndexToSend != infinity) {
      node.dependencyValues[node.roundIndexToSend] = 0;
    }
  }
};

struct DependencyBroadcast {
  using ValTy = std::pair<uint32_t, float>;

  static ValTy extract(uint32_t node_id, const struct NodeData & node) {
    uint32_t indexToGet = node.roundIndexToSend;

    float thing;
    if (indexToGet != infinity) {
      thing = node.dependencyValues[indexToGet];
    } else {
      thing = 0;
    }

    return ValTy(indexToGet, thing);
  }

  static bool extract_batch(unsigned, uint64_t*, unsigned int*, ValTy*, size_t*,
                            DataCommMode*) { return false; }

  static bool extract_batch(unsigned, ValTy*) { return false; }

  static void setVal(uint32_t node_id, struct NodeData & node, ValTy y) {
    uint32_t rIndex = y.first;
    if (rIndex != infinity) {
      float rDep = y.second;
      assert(node.roundIndexToSend == rIndex);
      node.dependencyValues[rIndex] = rDep;
    }
  }

  static bool setVal_batch(unsigned, uint64_t*, unsigned int*, ValTy*,
                           size_t, DataCommMode) { return false; }
};

////////////////////////////////////////////////////////////////////////////////
// Bitsets
////////////////////////////////////////////////////////////////////////////////

GALOIS_SYNC_STRUCTURE_BITSET(minDistances);
GALOIS_SYNC_STRUCTURE_BITSET(dependency);
