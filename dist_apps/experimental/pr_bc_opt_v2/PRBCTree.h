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
  galois::gstl::Map<uint32_t, galois::gstl::Set<uint32_t>> distanceTree;
  //! marks if a message has already been sent for a source
  galois::gstl::Vector<char> sentFlag;
  //! number of sources that have already been sent out
  uint32_t numSentSources;
  //! number of non-infinity values (i.e. number of sources added already)
  uint32_t numNonInfinity;
  //! last round number that something was sent out
  uint32_t lastRound;

  using TreeIter = 
    typename galois::gstl::Map<uint32_t, 
               galois::gstl::Set<uint32_t>>::const_reverse_iterator;
  using SetIter = typename galois::gstl::Set<uint32_t>::const_reverse_iterator;

  TreeIter curKey;
  TreeIter endCurKey;
  SetIter curSet;
  SetIter endCurSet;

 public:
  /**
   * Reset the map, initialize all distances to infinity, and reset the "sent"
   * vector and num sent sources.
   */
  void initialize(unsigned int numRoundSources) {
    distanceTree.clear();
    // reset sent flags
    sentFlag.resize(numRoundSources);
    for (unsigned i = 0; i < numRoundSources; i++) {
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
    distanceTree[newDistance].insert(index);
    numNonInfinity++;
  }

  /**
   * Update the distance map: given an index to update as well as its old 
   * distance, remove the old distance and replace with new distance.
   */
  void setDistance(uint32_t index, uint32_t oldDistance, uint32_t newDistance) {
    size_t count = 0;
    // if it exists, remove it
    if (distanceTree.find(oldDistance) != distanceTree.end()) {
      count = distanceTree[oldDistance].erase(index);
    }
    // if it didn't exist before, add to number of non-infinity nodes
    if (count == 0) {
      numNonInfinity++;
    }
    distanceTree[newDistance].insert(index);
  }

  /**
   * Get the index that needs to be sent out this round given the round number.
   */
  uint32_t getIndexToSend(uint32_t roundNumber) {
    uint32_t distanceToCheck = roundNumber - numSentSources;
    galois::gstl::Set<uint32_t>& setToCheck = distanceTree[distanceToCheck];

    uint32_t indexToSend = infinity;

    for (const uint32_t index : setToCheck) {
      if (!sentFlag[index]) {
        indexToSend = index;
        break;
      }
    }

    return indexToSend;
  }

  // distance + numSentSources - 1 == lastRound - curroundNumber
  
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
    }
  }

  /**
   * Given a round number, figure out which index needs to be sent out for the
   * back propagation phase.
   */
  uint32_t backGetIndexToSend(uint32_t roundNumber) {
    uint32_t indexToReturn = infinity;

    while (curKey != endCurKey) {
      if (curSet != endCurSet) {
        uint32_t curNumber = *curSet;
        uint32_t distance = curKey->first;

        if ((distance + numSentSources - 1) == (lastRound - roundNumber)) {
          // this number should be sent out this round
          indexToReturn = curNumber;
          curSet++;
          numSentSources--;
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
