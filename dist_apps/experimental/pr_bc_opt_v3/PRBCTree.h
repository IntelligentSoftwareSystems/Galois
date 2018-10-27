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

#ifndef _PRBCTREE_
#define _PRBCTREE_
#include <boost/container/flat_map.hpp>
// #include "bitset_with_indicator.hh"
#include "galois/DynamicBitset.h"

const uint32_t infinity = std::numeric_limits<uint32_t>::max() >> 2;

/**
 * Binary tree class to make finding a source's message to send out during PRBC
 * easier.
 */
class PRBCTree {

  // using BitSet = bitset_with_indicator<uint32_t>;
  using BitSet = galois::DynamicBitSet;
  using FlatMap = boost::container::flat_map<uint32_t, BitSet,
                                              std::less<uint32_t>,
                                              galois::gstl::Pow2Alloc<std::pair<uint32_t, BitSet>>>;

  //! map to a bitset of nodes that belong in a particular distance group
  FlatMap distanceTree;
  //! number of sources that have already been sent out
  uint32_t numSentSources;
  //! number of non-infinity values (i.e. number of sources added already)
  uint32_t numNonInfinity;
  //! indicates if zero distance has been reached for backward iteration
  bool zeroReached;

  //! reverse iterator over map
  using TreeIter = typename FlatMap::reverse_iterator;
  //! Current iterator for reverse map
  TreeIter curKey;
  //! End key for reverse map iterator
  TreeIter endCurKey;

public:
  /**
   * Reset the map, initialize all distances to infinity, and reset the "sent"
   * vector and num sent sources.
   */
  void initialize() {
    distanceTree.clear();
    // reset number of sent sources
    numSentSources = 0;
    // reset number of non infinity sources that exist
    numNonInfinity = 0;
    // reset the flag for backward phase
    zeroReached = false;
  }

  /**
   * Assumes you're adding a NEW distance; i.e. there better not be a duplicate
   * of index somewhere.
   */
  void setDistance(uint32_t index, uint32_t newDistance) {
    if (distanceTree[newDistance].size() != numSourcesPerRound) {
      distanceTree[newDistance].resize(numSourcesPerRound);
    }
    distanceTree[newDistance].set_indicator(index);

    numNonInfinity++;
  }

  /**
   * Update the distance map: given an index to update as well as its old 
   * distance, remove the old distance and replace with new distance.
   */
  void setDistance(uint32_t index, uint32_t oldDistance, uint32_t newDistance) {
    if (oldDistance == newDistance) {
      return;
    }

    auto setIter = distanceTree.find(oldDistance);
    bool existed = false;
    // if it exists, remove it
    if (setIter != distanceTree.end()) {
      BitSet& setToChange = setIter->second;
      existed = setToChange.test_set_indicator(index, false); // Test, set, update
    }

    // if it didn't exist before, add to number of non-infinity nodes
    if (!existed) {
      numNonInfinity++;
    }

    if (distanceTree[newDistance].size() != numSourcesPerRound) {
      distanceTree[newDistance].resize(numSourcesPerRound);
    }
    distanceTree[newDistance].set_indicator(index);

  }

  /**
   * Get the index that needs to be sent out this round given the round number.
   */
  uint32_t getIndexToSend(uint32_t roundNumber) {
    uint32_t distanceToCheck = roundNumber - numSentSources;
    uint32_t indexToSend = infinity;

    auto setIter = distanceTree.find(distanceToCheck);
    if (setIter != distanceTree.end()) {
      BitSet& setToCheck = setIter->second;
      auto index = setToCheck.getIndicator();
      if (index != setToCheck.npos) {
        indexToSend = index;
      }
    }
    return indexToSend;
  }

  
  /**
   * Note that a particular source's message has already been sent in the data
   * structure and increment the number of sent sources.
   */
  void markSent(uint32_t roundNumber) {
    uint32_t distanceToCheck = roundNumber - numSentSources;
    BitSet& setToCheck = distanceTree[distanceToCheck];
    setToCheck.forward_indicator();

    numSentSources++;
  }

  /**
   * Return true if potentially more work exists to be done
   */
  bool moreWork() { return numNonInfinity > numSentSources; }

  /**
   * Begin the setup for the back propagation phase by setting up the 
   * iterators.
   */
  void prepForBackPhase() {
    curKey = distanceTree.rbegin();
    endCurKey = distanceTree.rend();

    if (curKey != endCurKey) {
      BitSet& curSet = curKey->second;
      #ifdef FLIP_MODE
        curSet.flip();
      #endif
      curSet.backward_indicator();
    }

  }

  /**
   * Given a round number, figure out which index needs to be sent out for the
   * back propagation phase.
   */
  uint32_t backGetIndexToSend(const uint32_t roundNumber, 
                              const uint32_t lastRound) {
    uint32_t indexToReturn = infinity;

    while (curKey != endCurKey) {
      uint32_t distance = curKey->first;
      if ((distance + numSentSources - 1) != (lastRound - roundNumber)){
        // round to send not reached yet; get out
        return infinity;
      }

      if (distance == 0) {
        zeroReached = true;
        return infinity;
      }

      BitSet& curSet = curKey->second;
      if (!curSet.nposInd()) {
          // this number should be sent out this round
          indexToReturn = curSet.backward_indicator();
          numSentSources--;
          break;
      } else {
        // set exhausted; go onto next set
        for (curKey++; curKey != endCurKey && curKey->second.none(); curKey++);

        // if another set exists, set it up, else do nothing
        if (curKey != endCurKey) {
          BitSet& nextSet = curKey->second;
          #ifdef FLIP_MODE
            nextSet.flip();
          #endif
          nextSet.backward_indicator();
        }
      }
    }

    if (curKey == endCurKey) {
      assert(numSentSources == 0);
    }

    return indexToReturn;
  }

  /**
   * Returns zeroReached variable.
   */
  bool isZeroReached() {
    return zeroReached;
  }
};

#endif
