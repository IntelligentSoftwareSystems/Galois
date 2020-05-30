/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
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

#ifndef _GALOIS_SPARSEBITVECTOR_
#define _GALOIS_SPARSEBITVECTOR_

#include <galois/AtomicWrapper.h>
#include <galois/Mem.h>
#include <utility>
#include <boost/iterator/iterator_facade.hpp>

namespace galois {

/**
 * Sparse bit vector using a linked list. Also thread safe; however, only
 * guarantee is that functions return values based on the state of the
 * vector AT THE TIME THE FUNCTION IS CALLED. (i.e. if concurrent update
 * happens in a function call, the update may or may not be visible).
 */
template <bool IsConcurrent>
struct SparseBitVector {
  using WORD                     = unsigned int;
  static const unsigned wordSize = sizeof(WORD) * 8;

  //////////////////////////////////////////////////////////////////////////////

  /**
   * Node in sparse bit vector linked list
   */
  struct Node {
    unsigned _base; // base to multiply by/used to sort linked list
    // If concurrent, then wrap these in a copyable atomic
    using WordType =
        typename std::conditional<IsConcurrent, galois::CopyableAtomic<WORD>,
                                  WORD>::type;
    WordType _bitVector; // stores set bits for a base

    using NodeType =
        typename std::conditional<IsConcurrent, galois::CopyableAtomic<Node*>,
                                  Node*>::type;
    NodeType _next; // pointer to next node in linked list

    /**
     * Needs a base when being constructed.
     *
     * @param base
     */
    Node(unsigned base) {
      _base      = base;
      _bitVector = 0;
      _next      = nullptr;
    }

    /**
     * Construct with an already set
     *
     * @param base
     */
    Node(unsigned base, unsigned offset) {
      _base = base;

      // set bit at offset
      unsigned toStore = 0;
      toStore |= ((WORD)1 << offset);
      _bitVector = toStore;

      _next = nullptr;
    }

    /**
     * Thread safe set. Uses compare and swap to atomically update the
     * bitvector.
     *
     * @param offset Offset to set the bit at
     * @returns true if the set bit wasn't set previously
     */
    template <bool A                            = IsConcurrent,
              typename std::enable_if<A>::type* = nullptr>
    bool set(unsigned offset) {
      unsigned expected = _bitVector;
      unsigned newValue = expected | ((WORD)1 << offset);
      bool changed      = (expected != newValue);

      while (changed && !std::atomic_compare_exchange_weak(
                            &_bitVector, &expected, newValue)) {
        // if cas fails, then update new value
        newValue = expected | ((WORD)1 << offset);
        changed  = (expected != newValue);
      }

      return changed;
    }

    /**
     * Sets the bit at the provided offset. Not thread safe/ not concurrent
     * version.
     *
     * @param offset Offset to set the bit at
     * @returns true if the set bit wasn't set previously
     */
    template <bool A                             = IsConcurrent,
              typename std::enable_if<!A>::type* = nullptr>
    bool set(unsigned offset) {
      WORD beforeBits = _bitVector;
      _bitVector |= ((WORD)1 << offset);
      return _bitVector != beforeBits;
    }

    /**
     * @param offset Offset into bits to check status of
     * @returns true if bit at offset is set, false otherwise
     */
    bool test(unsigned offset) const {
      WORD mask = (WORD)1 << offset;
      return ((_bitVector & mask) == mask);
    }

    /**
     * Determines if second has set all of the bits that this objects has set.
     *
     * @param second pointer to compare against
     * @returns true if second word's bits has everything that this
     * word's bits have
     */
    bool isSubsetEq(Node* second) const {
      WORD current = _bitVector;
      return (current & (second->_bitVector)) == current;
    }

    /**
     * Bitwise or with second's bits field on our field.
     *
     * @param second sbv node to do a bitwise or with
     * @returns 1 if something changed, 0 otherwise
     */
    template <bool A                            = IsConcurrent,
              typename std::enable_if<A>::type* = nullptr>
    unsigned unify(Node* second) {
      if (second) {
        WORD oldVector = _bitVector;
        WORD newVector = oldVector | (second->_bitVector);
        bool changed   = (oldVector != newVector);

        while (changed && !std::atomic_compare_exchange_weak(
                              &_bitVector, &oldVector, newVector)) {
          // if cas fails, update again
          newVector = oldVector | (second->_bitVector);
          changed   = (oldVector != newVector);
        }

        return changed;
      }

      return 0;
    }

    /**
     * Bitwise or with second's bits field on our field. Non-concurrent
     * version.
     *
     * @param second Node to do a bitwise or with
     * @returns 1 if something changed, 0 otherwise
     */
    template <bool A                             = IsConcurrent,
              typename std::enable_if<!A>::type* = nullptr>
    unsigned unify(Node* second) {
      if (second) {
        WORD oldBits = _bitVector;
        _bitVector |= second->_bitVector;
        return (_bitVector != oldBits);
      }

      return 0;
    }

    // TODO revise this to use a Galois allocator
    /**
     * @returns a pointer to a copy of this word without the preservation
     * of the linked list
     */
    Node* clone(galois::FixedSizeAllocator<Node>* nodeAllocator) const {
      Node* newWord = nodeAllocator->allocate(1);
      nodeAllocator->construct(newWord, 0);

      newWord->_base      = _base;
      newWord->_bitVector = _bitVector;
      newWord->_next      = nullptr;

      return newWord;
    }

    /**
     * TODO can probably use bit twiddling to get count more efficiently
     *
     * @returns The number of set bits in this word
     */
    unsigned count() const {
      unsigned numElements = 0;

      WORD bitMask = 1;
      WORD bits    = _bitVector;

      for (unsigned ii = 0; ii < wordSize; ++ii) {
        if (bits & bitMask) {
          ++numElements;
        }

        bitMask <<= 1;
      }
      return numElements;
    }

    /**
     * Gets the set bits in this word and adds them to the passed in
     * vector.
     *
     * @tparam VectorTy vector type that supports push_back
     * @param setBits Vector to add set bits to
     * @returns Number of set bits in this word
     */
    template <typename VectorTy>
    unsigned getAllSetBits(VectorTy& setbits) const {
      // or mask used to mask set bits
      WORD orMask     = 1;
      unsigned numSet = 0;
      WORD bits       = _bitVector;

      for (unsigned curBit = 0; curBit < wordSize; ++curBit) {
        if (bits & orMask) {
          setbits.push_back(_base * wordSize + curBit);
          numSet++;
        }

        orMask <<= 1;
      }

      return numSet;
    }
  };

  //////////////////////////////////////////////////////////////////////////////

  /**
   * Iterator for SparseBitVector
   *
   * BEHAVIOR IF THE BIT VECTOR IS ALTERED DURING ITERATION IS UNDEFINED.
   * (i.e. correctness is not guaranteed)
   */
  class SBVIterator
      : public boost::iterator_facade<SBVIterator, const unsigned,
                                      boost::forward_traversal_tag> {
    Node* currentHead;
    unsigned currentBit{0};
    unsigned currentValue{~0U};

    void advanceToNextBit(bool inclusive) {
      if (!inclusive) {
        currentBit++; // current bit doesn't count for checking
      }

      bool found = false;
      while (!found && currentHead != nullptr) {
        while (currentBit < wordSize) {
          if (currentHead->test(currentBit)) {
            found = true;
            break;
          } else {
            currentBit++;
          }
        }

        if (!found) {
          currentHead = (currentHead->_next);
          currentBit  = 0;
        }
      }

      if (currentHead != nullptr) {
        currentValue = (currentHead->_base * wordSize) + currentBit;
      } else {
        currentValue = -1;
      }
    }

  public:
    /**
     * This is the end for an iterator.
     */
    SBVIterator() : currentHead(nullptr), currentBit(0), currentValue(-1) {
      currentValue = -1;
    }

    SBVIterator(Node* firstHead)
        : currentHead(firstHead), currentBit(0), currentValue(-1) {
      advanceToNextBit(true);
    }

    SBVIterator(SparseBitVector* bv) : currentBit(0), currentValue(-1) {
      currentHead = (bv->head);
      advanceToNextBit(true);
    }

  private:
    friend class boost::iterator_core_access;

    /**
     * Goes to next bit of bitvector.
     */
    void increment() {
      if (currentHead != nullptr) {
        advanceToNextBit(false); // false = increment currentBit
      }                          // do nothing if head is nullptr (i.e. the end)
    }

    /**
     * @param other Another iterator to compare against
     * @returns true if other iterator currently points to the same location
     */
    bool equal(const SBVIterator& other) const {
      if (currentHead != nullptr) {
        if (other.currentHead == currentHead &&
            other.currentBit == currentBit) {
          return true;
        } else {
          return false;
        }
      } else {
        if (other.currentHead == nullptr) {
          return true;
        } else {
          return false;
        }
      }
    }

    /**
     * @returns the current value that the iterator is pointing to
     */
    const unsigned& dereference() const { return currentValue; }
  };

  //////////////////////////////////////////////////////////////////////////////

  using NodeType =
      typename std::conditional<IsConcurrent, galois::CopyableAtomic<Node*>,
                                Node*>::type;
  // head of linked list
  NodeType head;
  // allocator of new nodes
  galois::FixedSizeAllocator<Node>* nodeAllocator;

  /**
   * Default constructor = nullptrs
   */
  SparseBitVector() {
    head          = nullptr;
    nodeAllocator = nullptr;
  }

  /**
   * Free all nodes allocated with the node allocator
   */
  ~SparseBitVector() {
    if (nodeAllocator) {
      this->freeAll();
    }
  }

  void freeAll() {
    if (nodeAllocator) { // check to make sure nodeAllocator isn't null
      while (head) {
        Node* current = head;
        head          = current->_next;
        nodeAllocator->destroy(current);
        nodeAllocator->deallocate(current, 1);
      }
    }

    // set to null pointer so it doesn't try this again later on destructor
    nodeAllocator = nullptr;
  }

  /**
   * Initialize by setting head to nullptr and saving the word allocator.
   *
   * @param _nodeAllocator allocator object for nodes to use when creating
   * a new linked list node
   */
  void init(galois::FixedSizeAllocator<Node>* _nodeAllocator) {
    head          = nullptr;
    nodeAllocator = _nodeAllocator;
  }

  /**
   * @returns iterator to first set element of this bitvector
   */
  SBVIterator begin() { return SBVIterator(this); }

  /**
   * @returns end iterator of this bitvector.
   */
  SBVIterator end() { return SBVIterator(); }

  /**
   * Set the provided bit num in the bitvector. Will create a new word if the
   * word needed to set the bit doesn't exist yet + will rearrange linked
   * list of words as necessary.
   *
   * @param num The bit to set in the bitvector
   * @returns true if the bit set wasn't set previously
   */
  template <bool A = IsConcurrent, typename std::enable_if<A>::type* = nullptr>
  bool set(unsigned num) {
    unsigned baseWord;
    unsigned offsetIntoWord;

    // determine base word and bit that corresponds to the num
    std::tie(baseWord, offsetIntoWord) = getOffsets(num);

    Node* curPtr = head;
    Node* prev   = nullptr;

    // while true due to fact that compare and swap (CAS) may fail
    while (true) {
      // pointers should be in sorted order
      // loop through linked list to find the correct base word (if it exists)
      while (curPtr != nullptr && curPtr->_base < baseWord) {
        prev   = curPtr;
        curPtr = (curPtr->_next);
      }

      // if base already exists, then set the correct offset bit
      if (curPtr != nullptr && curPtr->_base == baseWord) {
        return curPtr->set(offsetIntoWord);
        // else the base wasn't found; create and set, then rearrange linked
        // list accordingly
      } else {
        Node* newWord = nodeAllocator->allocate(1);
        nodeAllocator->construct(newWord, baseWord, offsetIntoWord);

        // note at this point curPtr is the next element in the list that
        // the new one we create should point to
        newWord->_next = curPtr;

        // attempt a compare and swap: if it fails, that means the list was
        // altered, so go back to beginning of this loop to check again
        if (prev) {
          if (std::atomic_compare_exchange_weak(&(prev->_next), &curPtr,
                                                newWord)) {
            return true;
          } else {
            // if it fails, return to the top; current pointer has new value
            // that needs to be checked
            nodeAllocator->destroy(newWord);
            nodeAllocator->deallocate(newWord, 1);
          }
        } else {
          if (std::atomic_compare_exchange_weak(&head, &curPtr, newWord)) {
            return true;
          } else {
            // if it fails, return to the top; current pointer has new value
            // that needs to be checked
            nodeAllocator->destroy(newWord);
            nodeAllocator->deallocate(newWord, 1);
          }
        }
      }
    }
  }

  /**
   * Set the provided bit in the bitvector. Will create a new word if the
   * word needed to set the bit doesn't exist yet + will rearrange linked
   * list of words as necessary.
   *
   * Not thread safe.
   *
   * @param bit The bit to set in the bitvector
   * @returns true if the bit set wasn't set previously
   */
  template <bool A = IsConcurrent, typename std::enable_if<!A>::type* = nullptr>
  bool set(unsigned bit) {
    unsigned baseWord;
    unsigned offsetIntoWord;

    std::tie(baseWord, offsetIntoWord) = getOffsets(bit);

    Node* curPtr = head;
    Node* prev   = nullptr;

    // pointers should be in sorted order
    // loop through linked list to find the correct base word (if it exists)
    while (curPtr != nullptr && curPtr->_base < baseWord) {
      prev   = curPtr;
      curPtr = curPtr->_next;
    }

    // if base already exists, then set the correct offset bit
    if (curPtr != nullptr && curPtr->_base == baseWord) {
      return curPtr->set(offsetIntoWord);
      // else the base wasn't found; create and set, then rearrange linked list
      // accordingly
    } else {
      Node* newWord = nodeAllocator->allocate(1);
      nodeAllocator->construct(newWord, baseWord, offsetIntoWord);

      // this should point to prev's next, prev should point to this
      if (prev) {
        newWord->_next = prev->_next;
        prev->_next    = newWord;
      } else {
        if (curPtr == nullptr) {
          // this is the first word we are adding since both prev and head are
          // null; next is nothing
          newWord->_next = nullptr;
        } else {
          // this new word goes before curptr; if prev is null and curptr isn't,
          // it means it had to go before
          newWord->_next = head;
        }

        head = newWord;
      }

      return true;
    }
  }

  /**
   * Determines if a particular number bit in the bitvector is set.
   *
   * Note it may return false if a bit is set concurrently by another
   * thread.
   *
   * @param num Bit in bitvector to check status of
   * @returns true if the argument bit is set in this bitvector, false
   * otherwise. May also return false if the bit being tested for is set
   * concurrently by another thread.
   */
  bool test(unsigned num) const {
    unsigned baseWord;
    unsigned offsetIntoWord;

    std::tie(baseWord, offsetIntoWord) = getOffsets(num);
    Node* curPointer                   = head;

    while (curPointer != nullptr && curPointer->_base < baseWord) {
      curPointer = (curPointer->_next);
    }

    if (curPointer != nullptr && curPointer->_base == baseWord) {
      return curPointer->test(offsetIntoWord);
    } else {
      return false;
    }
  }

  /**
   * READ THIS REGARDING THE CONCURRENT VERSION OF THIS:
   *
   * This function, in some sense, will not return false incorrectly (i.e.
   * if it returns false, then at that point in time this vector actually
   * isnt' a subset of the other vector; however, it is entirely possible
   * that a concurrent update will change that as this function is returning
   * "false").
   *
   * So basically, no guarantees. It exists to somewhat optimize some steps
   * of PointsTo, but it should not be relied on if writes to THIS bitvector
   * can happen concurrently (writes to second are fine).
   *
   * In serial execution it will work as expected.
   *
   * @param second Vector to check if this vector is a subset of
   * @returns true if this vector is a subset of the second vector
   */
  bool isSubsetEq(const SparseBitVector& second) const {
    Node* ptrOne = head;
    Node* ptrTwo = second.head;

    while (ptrOne != nullptr && ptrTwo != nullptr) {
      if (ptrOne->_base == ptrTwo->_base) {
        if (!ptrOne->isSubsetEq(ptrTwo)) {
          return false;
        }

        // subset check successful; advance both pointers
        ptrOne = (ptrOne->_next);
        ptrTwo = (ptrTwo->_next);
      } else if (ptrOne->_base < ptrTwo->_base) {
        // ptrTwo has overtaken ptrOne, i.e. one has something (a base)
        // two doesn't
        return false;
      } else { // ptrOne > ptrTwo
        // greater than case; advance ptrTwo to see if it eventually
        // reaches what ptrOne is currently at
        ptrTwo = (ptrTwo->_next);
      }
    }

    if (ptrOne != nullptr) {
      // if ptrOne is not null, the loop exited because ptrTwo is nullptr,
      // meaning this vector has more than the other vector, i.e. not a subset
      return false;
    } else {
      // here means ptrOne == nullptr => it has sucessfully subset checked all
      // words that matter
      return true;
    }
  }

  /**
   * Concurrent version.
   *
   * Takes the passed in bitvector and does an "or" with it to update this
   * bitvector.
   *
   * ONLY GUARANTEE IS THAT YOU WILL GET THINGS IN second THAT EXISTED
   * AT TIME OF CALL; IF second IS UPDATED CONCURRENTLY, YOU MAY OR MAY
   * NOT GET THOSE UPDATES.
   *
   * @param second BitVector to merge this one with
   * @returns a non-negative value if something changed
   */
  template <bool A = IsConcurrent, typename std::enable_if<A>::type* = nullptr>
  unsigned unify(const SparseBitVector& second) {
    unsigned changed = 0;

    Node* prev   = nullptr;
    Node* ptrOne = head;
    Node* ptrTwo = second.head;

    while (ptrTwo != nullptr) {
      while (ptrOne != nullptr && ptrTwo != nullptr) {
        if (ptrOne->_base == ptrTwo->_base) {
          // merged ptrTwo's word with our word, then advance both
          changed += ptrOne->unify(ptrTwo);

          prev   = ptrOne;
          ptrOne = (ptrOne->_next);
          ptrTwo = (ptrTwo->_next);
        } else if (ptrOne->_base < ptrTwo->_base) {
          // advance our pointer until we reach new bases we don't have
          prev   = ptrOne;
          ptrOne = (ptrOne->_next);
        } else { // oneBase > twoBase
          // two has something we don't have; add it between prev and current
          // ptrone
          Node* newWord = ptrTwo->clone(nodeAllocator);
          // newWord comes before our current word
          (newWord->_next) = ptrOne;

          if (prev) {
            if (!std::atomic_compare_exchange_weak(&(prev->_next), &ptrOne,
                                                   newWord)) {
              // if it fails, return to the top; ptrOne has new value
              // that needs to be checked
              nodeAllocator->destroy(newWord);
              nodeAllocator->deallocate(newWord, 1);
              continue;
            }
          } else {
            if (!std::atomic_compare_exchange_weak(&head, &ptrOne, newWord)) {
              // if it fails, return to the top; ptrOne has new value
              // that needs to be checked
              nodeAllocator->destroy(newWord);
              nodeAllocator->deallocate(newWord, 1);
              continue;
            }
          }

          prev = newWord;
          // done with ptrTwo's word, advance
          ptrTwo = (ptrTwo->_next);

          changed++;
        }
      }

      // ptrOne = nullptr, but ptrTwo still has values; clone values
      // and attempt to add
      while (ptrTwo) {
        Node* newWord = ptrTwo->clone(nodeAllocator);

        // note ptrOne in below cases should be nullptr...
        if (prev) {
          if (!std::atomic_compare_exchange_weak(&(prev->_next), &ptrOne,
                                                 newWord)) {
            // if it fails, return to the top; ptrOne has new value
            // that needs to be checked
            nodeAllocator->destroy(newWord);
            nodeAllocator->deallocate(newWord, 1);
            break; // goes out to outermost while loop
          }
        } else {
          if (!std::atomic_compare_exchange_weak(&head, &ptrOne, newWord)) {
            // if it fails, return to the top; ptrOne has new value
            // that needs to be checked
            nodeAllocator->destroy(newWord);
            nodeAllocator->deallocate(newWord, 1);
            break; // goes out to outermost while loop
          }
        }

        prev   = newWord;
        ptrTwo = (ptrTwo->_next);

        changed++;
      }
    }

    return changed;
  }

  /**
   * Non-concurrent version.
   *
   * Takes the passed in bitvector and does an "or" with it to update this
   * bitvector.
   *
   * @param second BitVector to merge this one with
   * @returns a non-negative value if something changed
   */
  template <bool A = IsConcurrent, typename std::enable_if<!A>::type* = nullptr>
  unsigned unify(const SparseBitVector& second) {
    unsigned changed = 0;

    Node* prev   = nullptr;
    Node* ptrOne = head;
    Node* ptrTwo = second.head;

    while (ptrOne != nullptr && ptrTwo != nullptr) {
      if (ptrOne->_base == ptrTwo->_base) {
        // merged ptrTwo's word with our word, then advance both
        changed += ptrOne->unify(ptrTwo);

        prev   = ptrOne;
        ptrOne = ptrOne->_next;
        ptrTwo = ptrTwo->_next;
      } else if (ptrOne->_base < ptrTwo->_base) {
        // advance our pointer until we reach new bases we don't have
        prev   = ptrOne;
        ptrOne = (ptrOne->_next);
      } else { // oneBase > twoBase
        // two has something we don't have; add it between prev and current
        // ptrone
        Node* newWord = ptrTwo->clone(nodeAllocator);
        // newWord comes before our current word
        newWord->_next = ptrOne;

        if (prev) {
          prev->_next = newWord;
        } else {
          head = newWord;
        }

        prev = newWord;
        // done with ptrTwo's word, advance
        ptrTwo = ptrTwo->_next;

        changed++;
      }
    }

    // ptrOne = nullptr, but ptrTwo still has values; clone values and add
    while (ptrTwo) {
      Node* newWord = ptrTwo->clone(nodeAllocator);

      if (prev) {
        prev->_next = newWord;
      } else {
        head = newWord;
      }

      prev   = newWord;
      ptrTwo = (ptrTwo->_next);

      changed++;
    }

    return changed;
  }

  /**
   * @returns number of bits set by all words in this bitvector
   */
  unsigned count() const {
    unsigned nbits = 0;

    for (Node* ptr = head; ptr; ptr = (ptr->_next)) {
      nbits += ptr->count();
    }

    return nbits;
  }

  /**
   * Gets the set bits in this bitvector and returns them in a vector type.
   *
   * @tparam VectorTy vector type that supports push_back
   * @returns Vector with all set bits
   */
  std::vector<unsigned> getAllSetBits() const {
    std::vector<unsigned> setBits;

    // loop through all words in the bitvector and get their set bits
    for (Node* curPtr = head; curPtr != nullptr; curPtr = curPtr->_next) {
      curPtr->getAllSetBits(setBits);
    }

    return setBits;
  }

  /**
   * Output the bits that are set in this bitvector.
   *
   * @param out Stream to output to
   * @param prefix A string to append to the set bit numbers
   */
  void print(std::ostream& out, std::string prefix = std::string("")) const {
    std::vector<unsigned> setBits = getAllSetBits();
    out << "Elements(" << setBits.size() << "): ";

    for (auto setBitNum : setBits) {
      out << prefix << setBitNum << ", ";
    }

    out << "\n";
  }

private:
  /**
   * @param num Bit that needs to be set
   * @returns a pair signifying a base word and the offset into a
   * baseword that corresponds to num
   */
  std::pair<unsigned, unsigned> getOffsets(unsigned num) const {
    unsigned baseWord       = num / wordSize;
    unsigned offsetIntoWord = num % wordSize;

    return std::pair<unsigned, unsigned>(baseWord, offsetIntoWord);
  }
};

} // namespace galois

#endif
