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
const uint32_t infinity = std::numeric_limits<uint32_t>::max() / 4;

/**
 * Binary tree class to make finding a source's message to send out during PRBC
 * easier.
 */
class PRBCTree {
  using MapBitset = galois::gstl::Vector<bool>;
  //! map to a bitset of nodes that belong in a particular distance group
  galois::gstl::Map<uint32_t, MapBitset> distanceTree;
  //! marks if a message has already been sent for a source
  galois::gstl::Vector<bool> sentFlag;
  //! number of sources that have already been sent out
  uint32_t numSentSources;
  //! number of non-infinity values (i.e. number of sources added already)
  uint32_t numNonInfinity;
  //! last round number that something was sent out
  uint32_t lastRound;

  //! Reverse map iterator
  using TreeIter = 
    typename galois::gstl::Map<uint32_t, MapBitset>::const_reverse_iterator;
  //! Reverse bitset iterator
  using SetIter = typename MapBitset::const_reverse_iterator;

  //! Current iterator for reverse map
  TreeIter curKey;
  //! End key for reverse map iterator
  TreeIter endCurKey;

  //! Current iterator for reverse bitset
  SetIter curSet;
  //! end key for reverse bitset iterator
  SetIter endCurSet;
  //! Current index in the iterator for the bitmap
  unsigned curIndex;

 public:
  /**
   * Reset the map, initialize all distances to infinity, and reset the "sent"
   * vector and num sent sources.
   */
  void initialize() {
    distanceTree.clear();
    // reset sent flags
    sentFlag.resize(numSourcesPerRound);
    for (unsigned i = 0; i < numSourcesPerRound; i++) {
      sentFlag[i] = 0;
    }
    assert(numSentSources == 0);
    // reset number of sent sources
    numSentSources = 0;
    // reset number of non infinity sources that exist
    numNonInfinity = 0;
    // reset last round
    lastRound = 0;
  }

  /**
   * Assumes you're adding a NEW distance; i.e. there better not be a duplicate
   * of index somewhere.
   */
  void setDistance(uint32_t index, uint32_t newDistance) {
    // create bitset if necessary
    if (distanceTree[newDistance].size() == 0) {
      distanceTree[newDistance].resize(numSourcesPerRound);
    }
    distanceTree[newDistance][index] = 1;
    numNonInfinity++;
  }

  /**
   * Update the distance map: given an index to update as well as its old 
   * distance, remove the old distance and replace with new distance.
   */
  void setDistance(uint32_t index, uint32_t oldDistance, uint32_t newDistance) {
    auto setIter = distanceTree.find(oldDistance);
    size_t count = 0;
    // if it exists, remove it
    if (setIter != distanceTree.end()) {
      MapBitset& setToChange = setIter->second;
      if (setToChange[index]) {
        count = 1;
        setToChange[index] = 0;
      }
    }

    // if it didn't exist before, add to number of non-infinity nodes
    if (count == 0) {
      numNonInfinity++;
    }

    // create bitset if necessary
    if (distanceTree[newDistance].size() == 0) {
      distanceTree[newDistance].resize(numSourcesPerRound);
    }
    distanceTree[newDistance][index] = 1;
  }

  /**
   * Get the index that needs to be sent out this round given the round number.
   */
  uint32_t getIndexToSend(uint32_t roundNumber) {
    uint32_t distanceToCheck = roundNumber - numSentSources;
    uint32_t indexToSend = infinity;

    auto setIter = distanceTree.find(distanceToCheck);
    if (setIter != distanceTree.end()) {
      MapBitset& setToCheck = setIter->second;

      // this iteration at worst case is as bad as prbcv1.....
      for (unsigned int index = 0; index < setToCheck.size(); index++) {
        if (setToCheck[index]) {
          if (!sentFlag[index]) {
            indexToSend = index;
            break;
          }
        }
      }
    }
    return indexToSend;
  }

  
  /**
   * Note that a particular source's message has already been sent in the data
   * structure and increment the number of sent sources.
   */
  void markSent(uint32_t index) {
    sentFlag[index] = 1;
    numSentSources++; 
  }

  /**
   * Return true if potentially more work exists to be done
   */
  bool moreWork() {
    return numNonInfinity > numSentSources;
  }

  /**
   * Begin the setup for the back propagation phase by setting up the 
   * iterators.
   */
  void prepForBackPhase(uint32_t lastRoundNum) {
    lastRound = lastRoundNum;
    curKey = distanceTree.crbegin();
    endCurKey = distanceTree.crend();

    // i.e. non-empty tree
    if (curKey != endCurKey) {
      curSet = curKey->second.crbegin();
      endCurSet = curKey->second.crend();
      curIndex = numSourcesPerRound;
    }
  }

  // distance + numSentSources - 1 == lastRound - curRoundNumber

  /**
   * Given a round number, figure out which index needs to be sent out for the
   * back propagation phase.
   */
  uint32_t backGetIndexToSend(uint32_t roundNumber) {
    uint32_t indexToReturn = infinity;

    while (curKey != endCurKey) {
      // loop to non-zero element in bitset
      while (!(*curSet) && curSet != endCurSet) {
        curSet++;
        curIndex--;
      }

      if (curSet != endCurSet) {
        uint32_t curNumber = curIndex - 1;
        uint32_t distance = curKey->first;

        if ((distance + numSentSources - 1) == (lastRound - roundNumber)) {
          // this number should be sent out this round
          indexToReturn = curNumber;
          curSet++;
          numSentSources--;
          curIndex--;
          break;
        } else {
          // round not reached yet; get out
          break;
        }
      } else {
        // set exhausted; go onto next set
        curKey++;

        // if another set exists, set it up, else do nothing
        if (curKey != endCurKey) {
          curSet = curKey->second.crbegin();
          endCurSet = curKey->second.crend();
          curIndex = numSourcesPerRound;
        }
      }
    }

    if (curKey == endCurKey) {
      assert(numSentSources == 0);
    }

    return indexToReturn;
  }
};

#endif
