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
#include <boost/dynamic_bitset.hpp>
const uint32_t infinity = std::numeric_limits<uint32_t>::max() >> 2;

/**
 * Binary tree class to make finding a source's message to send out during PRBC
 * easier.
 */
class PRBCTree {

  template <typename Block, typename Allocator>
  class bitset_with_index_indicator : public boost::dynamic_bitset<Block, Allocator> {
      // To match the type of npos (unsighed long by default)
      typedef typename std::remove_const<decltype(boost::dynamic_bitset<Block, Allocator>::npos)>::type size_type;

      //! indicate the index of bit to process
      size_type indicator;

      /**
       * To convert the bitset into block chain (vector of block-type).
       */
      std::vector<Block> integerize() const {
          std::vector<Block> blocks;
          /**
           * template <typename Block, typename Alloc, typename BlockOutputIterator>
           * void to_block_range(const dynamic_bitset<Block, Alloc>& b, BlockOutputIterator result)
           *
           * Effects:
           * Writes the bits of the bitset into the iterator result a block at a time.
           * The first block written represents the bits in the position range [0,bits_per_block) in the bitset,
           * the second block written the bits in the range [bits_per_block,2*bits_per_block), and so on.
           * For each block bval written, the bit (bval >> i) & 1 corresponds to the bit at position (b * bits_per_block + i) in the bitset.
           * 
           * Requires:
           * The type BlockOutputIterator must be a model of Output Iterator and its value_type must be the same type as Block.
           * Further, the size of the output range must be greater or equal b.num_blocks().
           */
          boost::to_block_range(*this, std::back_inserter(blocks));
          return blocks;
      }
  public:
      /* Constructor */
      bitset_with_index_indicator(uint32_t n = numSourcesPerRound) : boost::dynamic_bitset<Block, Allocator>(n) {
        this->reset();
        /**
         * dynamic_bitset::npos
         *
         * The maximum value of size_type.
         */
        this->indicator = this->npos;
      }

      /**
       * Accessors (get/set) for indicator.
       */
      size_type getIndicator() const {return this->indicator;}
      void setIndicator(size_type index) {this->indicator = index;}
      
      /**
       * Set a bit with the side-effect updating indicator to the first.
       */
      void set_indicator(size_type index) {
        this->set(index);
        if (index < indicator)
          this->indicator = index;
      }

      /**
       * size_type boost::dynamic_bitset::find_first() const;
       *
       * Returns: the lowest index i such as bit i is set, or npos if *this has no on bits.
       */

      /**
       * size_type boost::dynamic_bitset::find_next(size_type pos) const;
       *
       * Returns: the lowest index i greater than pos such as bit i is set, or npos if no such index exists.
       */

      /**
       * Similar to find_next().
       */
      size_type find_prev(size_type pos) const {

          // Return npos if no bit is set or to scan
          if (!(this->any()) || pos == 0)
              return this->npos;

          // Normalize pos in case of npos
          pos = (pos >= this->size()? this->size() : pos);
          --pos; // Find from the previous bit

          // Integerize into blocks
          std::vector<Block> blocks = this->integerize();

          auto curBlock = pos < this->bits_per_block? 0 : pos / this->bits_per_block;
          auto curOffset = pos < this->bits_per_block? pos : pos % this->bits_per_block;

          
          // Scan within current Block
          while (curOffset >= 0) {
              if (this->test(pos)){
                  return pos;}
              if (curOffset == 0)
                  break;
              pos--;
              curOffset--;
          }
          if (pos == 0)
              return this->npos;

          // Jump over zero blocks
          for (pos--, curBlock--, curOffset = this->bits_per_block - 1;
                  blocks[curBlock] == 0 && curBlock > 0;
                  curBlock--, pos -= this->bits_per_block);

          // Scan the last block (if non-zero)
          if (blocks[curBlock]) {
              while (curOffset >= 0) {
                  if (this->test(pos)){
                      return pos;}
                  pos--;
                  curOffset--;
              }
          }
          return this->npos;
      }

      /**
       * Similar to find_first().
       */
      size_type find_last() const {
          return this->find_prev(this->size());
      }

      /**
       * To move indicator to the next set bit.
       */
      void next_indicator() {
          this->indicator = 
            this->indicator == this->npos? 
            this->find_first() : this->find_next(this->indicator);
      }

      /**
       * To move indicator to the previous set bit.
       */
      void prev_indicator() {
          this->indicator = this->find_prev(this->indicator);
      }
  };

  using BitSet = bitset_with_index_indicator<uint32_t, galois::gstl::Pow2Alloc<uint32_t>>;
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
    // if it exists, remove it
    if (setIter != distanceTree.end()) {
      BitSet& setToChange = setIter->second;
      /**
       * bool boost::dynamic_bitset::test_set(size_type n, bool val = true)
       * 
       * Precondition: n < this->size().
       * 
       * Effects: Sets bit n if val is true, and clears bit n if val is false. 
       * 
       * Returns: true if the previous state of bit n was set and false if bit n is 0.
       */
      if (setToChange.test_set(index, false)) {
        if (index == setToChange.getIndicator()){
          setToChange.next_indicator();
        }
      }
    }

    // if it didn't exist before, add to number of non-infinity nodes
    if (oldDistance == infinity) {
      numNonInfinity++;
      // galois::gDebug("numNonInfinity: ", numNonInfinity);
    }
    distanceTree[newDistance].set_indicator(index); // Set & Update
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
          setToCheck.next_indicator();
        }
      }
    return indexToSend;
  }

  
  /**
   * Note that a particular source's message has already been sent in the data
   * structure and increment the number of sent sources.
   */
  void markSent(uint32_t index) { numSentSources++; }

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
      curSet.setIndicator(curSet.find_last());
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
      BitSet& curSet = curKey->second;
      auto curInd = curSet.getIndicator();
      if (curInd != curSet.npos) {
        uint32_t distance = curKey->first;

        if (distance == 0) {
          zeroReached = true;
        }

        if ((distance + numSentSources - 1) == (lastRound - roundNumber)) {
          // this number should be sent out this round
          indexToReturn = curInd;
          curSet.prev_indicator();
          numSentSources--;
          break;
        } else {
          // round to send not reached yet; get out
          break;
        }
      } else {
        // set exhausted; go onto next set
        for (curKey++; curKey != endCurKey; curKey++) {
          if (curKey->second.any()) break;
        }

        // if another set exists, set it up, else do nothing
        if (curKey != endCurKey) {
          BitSet& nextSet = curKey->second;
          nextSet.setIndicator(nextSet.find_last());
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
