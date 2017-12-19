/** SparseBitVector -*- C++ -*-
 * @file
 *
 * An inclusion-based points-to analysis algorithm to demostrate the Galois 
 * system.
 *
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2017, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS SOFTWARE
 * AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY, FITNESS FOR ANY
 * PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF PERFORMANCE, AND ANY
 * WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF DEALING OR USAGE OF TRADE.
 * NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH RESPECT TO THE USE OF THE
 * SOFTWARE OR DOCUMENTATION. Under no circumstances shall University be liable
 * for incidental, special, indirect, direct or consequential damages or loss of
 * profits, interruption of business, or related expenses which may arise from use
 * of Software or Documentation, including but not limited to those resulting from
 * defects in Software and/or Documentation, or loss or inaccuracy of data of any
 * kind.
 *
 * Sparse bit vector for compact storage of information.
 * 
 * @author Rupesh Nasre <rupesh0508@gmail.com>
 * @author Loc Hoang <l_hoang@utexas.edu>
 */
#ifndef GALOIS_SPARSEBITVECTOR_H
#define GALOIS_SPARSEBITVECTOR_H

#include <vector>
#include <string>
#include <ostream>

// TODO
// - get rid of new usage (new doesn't scale well); smart pointers probably
// better as well

namespace galois {

/**
 * Sparse bit vector represented as a linked list of words.
 * 
 * Stores objects as indices in sparse bit vectors.
 * Saves space when the data to be stored is sparsely populated.
 */
struct SparseBitVector {
  using WORD = unsigned long;
  // Number of bits in a word
  static const unsigned wordSize = sizeof(WORD) * 8;

  /**
   * A single word used as a bitvector. Contains functionality to alter it like
   * a bitvector.
   */
  struct OneWord {
    using WORD = unsigned long;
  
    WORD bits; // number that is used as the bitset
    unsigned base; // used to order the words of the vector
    struct OneWord* next; // pointer to next word on linked list 
                          // (using base as order)
  
    /**
     * Default is create a base at 0.
     */
    OneWord() { 
      OneWord(0);
    }
  
    /**
     * Creates a new word.
     *
     * @param _base base of this word, i.e. what order it should go in linked 
     * list
     */
    OneWord(unsigned _base) { 
      base = _base;
      bits = 0;
      next = nullptr;
    }
  
    /**
     * Creates a new word with an initial bit already set.
     *
     * @param _base base of this word, i.e. what order it should go in linked 
     * list
     * @param _initial Offset to first bit to set in the word
     */
    OneWord(unsigned _base, unsigned _initial) {
      base = _base;
      bits = 0;
      set(_initial);
      next = nullptr;
    }
  
    /**
     * Sets the bit at the provided offset.
     *
     * @param offset Offset to set the bit at
     * @returns true if the set bit wasn't set previously
     */
    bool set(unsigned offset) {
      WORD beforeBits = bits;
      bits |= ((WORD)1 << offset);
      return bits != beforeBits;
    }
  
    /**
     * @param offset Offset into bits to check status of
     * @returns true if bit at offset is set, false otherwise
     */
    bool test(unsigned offset) const {
      WORD mask = (WORD)1 << offset;
      return ((bits & mask) == mask);
    }
  
    /**
     * Bitwise or with second's bits field on our field.
     *
     * @param second OneWord to do a bitwise or with
     * @returns 1 if something changed, 0 otherwise
     *
     */
    unsigned unify(OneWord* second) {
      if (second) {
        WORD oldBits = bits;
        bits |= second->bits;
        return (bits != oldBits);
      }
  
      return 0;
    }
  
    /**
     * TODO can probably use bit twiddling to get count more efficiently
     *
     * @returns The number of set bits in this word
     */
    unsigned count() const {
      unsigned numElements = 0;
  
      WORD bitMask = 1;
  
      for (unsigned ii = 0; ii < wordSize; ++ii) {
        if (bits & bitMask) {
          ++numElements;
        }
  
        bitMask <<= 1;
      }
      return numElements;
    }
  
    /**
     * Determines if second has set all of the bits that this objects has set.
     *
     * @param second OneWord pointer to compare against
     * @returns true if second word's bits has everything that this
     * word's bits have
     */
    bool isSubsetEq(OneWord* second) const {
      return (bits & second->bits) == bits;
    }
  
    /**
     * @returns a pointer to a copy of this word without the preservation
     * of the linked list
     */
    OneWord* clone() const {
      OneWord* newWord = new OneWord(); // TODO don't use new, find better way
  
      newWord->base = base;
      newWord->bits = bits;
      newWord->next = nullptr;
  
      return newWord;
    }
  
    /**
     * @returns a pointer to a copy of this word WITH the preservation of
     * the linked list via copies of the list starting from this word
     */
    OneWord* cloneAll() const {
      OneWord* newListBeginning = clone();
  
      OneWord* curPtr = newListBeginning;
      OneWord* nextPtr = next;
  
      // clone down the linked list starting from this pointer
      while (nextPtr != nullptr) {
        curPtr->next = nextPtr->clone();
        nextPtr = nextPtr->next;
        curPtr = curPtr->next;
      }
  
      return newListBeginning;
    }
  
   /**
    * Gets the set bits in this word and adds them to the passed in 
    * vector.
    *
    * @tparam VectorTy vector type that supports push_back
    * @param setBits Vector to add set bits to
    * @returns Number of set bits in this word
    */
    template<typename VectorTy>
    unsigned getAllSetBits(VectorTy &setbits) const {
      // or mask used to mask set bits
      WORD orMask = 1;
      unsigned numSet = 0;
  
      for (unsigned curBit = 0; curBit < wordSize; ++curBit) {
        if (bits & orMask) {
          setbits.push_back(base * wordSize + curBit);
          numSet++;
        }
  
        orMask <<= 1;
      }
  
      return numSet;
    }
  };

  OneWord* head;

  SparseBitVector() {
    init();
  }

  /**
   * Initialize by setting head to nullptr
   */
  void init() {
    head = nullptr;
  }

  /**
   * Set the provided bit in the bitvector. Will create a new word if the
   * word needed to set the bit doesn't exist yet + will rearrange linked
   * list of words as necessary.
   *
   * TODO make this thread-safe
   *
   * @param bit The bit to set in the bitvector
   * @returns true if the bit set wasn't set previously
   */
  bool set(unsigned bit) {
    unsigned baseWord;
    unsigned offsetIntoWord;

    std::tie(baseWord, offsetIntoWord) = getOffsets(bit);

    OneWord* curPtr = head;
    OneWord* prev = nullptr;

    // pointers should be in sorted order 
    // loop through linked list to find the correct base word (if it exists)
    while (curPtr != nullptr && curPtr->base < baseWord) {
      prev = curPtr;
      curPtr = curPtr->next;
    }

    // if base already exists, then set the correct offset bit
    if (curPtr != nullptr && curPtr->base == baseWord) {
      return curPtr->set(offsetIntoWord);
    // else the base wasn't found; create and set, then rearrange linked list
    // accordingly
    } else {
      OneWord *newWord = new OneWord(baseWord, offsetIntoWord);

      // this should point to prev's next, prev should point to this
      if (prev) {
        newWord->next = prev->next;
        prev->next = newWord;
      } else {
        if (curPtr == nullptr) {
          // this is the first word we are adding since both prev and head are 
          // null; next is nothing
          newWord->next = nullptr;
        } else {
          // this new word goes before curptr; if prev is null and curptr isn't,
          // it means it had to go before
          newWord->next = head;
        }

        head = newWord;
      }

      return true;
    }
  }

  /**
   * Determines if a particular bit in the bitvector is set.
   *
   * @param bit Bit in bitvector to check status of
   * @returns True if the argument bit is set in this bitvector, false otherwise
   */
  bool test(unsigned bit) const {
    unsigned baseWord;
    unsigned offsetIntoWord;

    std::tie(baseWord, offsetIntoWord) = getOffsets(bit);
    OneWord* curPointer = head;

    while (curPointer != nullptr && curPointer->base < baseWord) {
      curPointer = curPointer->next;
    }

    if (curPointer != nullptr && curPointer->base == baseWord) {
      return curPointer->test(offsetIntoWord);
    } else {
      return false;
    }
  }

  /**
   * Takes the passed in bitvector and does an "or" with it to update this
   * bitvector.
   *
   * @param second BitVector to merge this one with
   * @returns a non-negative value if something changed
   */
  unsigned unify(const SparseBitVector& second) {
    unsigned changed = 0;

    OneWord* prev = nullptr;
    OneWord* ptrOne = head;
    OneWord* ptrTwo = second.head;

    while (ptrOne != nullptr && ptrTwo != nullptr) {
      if (ptrOne->base == ptrTwo->base) {
        // merged ptrTwo's word with our word, then advance both
        changed += ptrOne->unify(ptrTwo);

        prev = ptrOne;
        ptrOne = ptrOne->next;
        ptrTwo = ptrTwo->next;
      } else if (ptrOne->base < ptrTwo->base) {
        // advance our pointer until we reach "new" words
        prev = ptrOne;
        ptrOne = ptrOne->next;
      } else { // oneBase > twoBase
        // add ptrTwo's word that we don't have (otherwise would have been
        // handled by other case), fix linked list on our side
        OneWord* newWord = ptrTwo->clone();
        newWord->next = ptrOne; // this word comes before our current word

        // if previous word exists, make it point to this new word, 
        // else make this word the new head and prev pointer
        if (prev) {
          prev->next = newWord;
          prev = newWord;
        } else {
          head = prev = newWord;
        }

        // done with ptrTwo's word, advance
        ptrTwo = ptrTwo->next;

        changed++;
      }
    }

    // ptrOne = nullptr, but ptrTwo still has values; clone the values and
    // add them to our own bitvector
    if (ptrTwo) {
      OneWord* remaining = ptrTwo->cloneAll();

      if (prev) {
        prev->next = remaining;
      } else {
        head = remaining;
      }

      changed++;
    }

    return changed;
  }

  /**
   * @param second Vector to check if this vector is a subset of
   * @returns true if this vector is a subset of the second vector
   */
  bool isSubsetEq(const SparseBitVector& second) const {
    OneWord* ptrOne = head;
    OneWord* ptrTwo = second.head;

    while (ptrOne != nullptr && ptrTwo != nullptr) {
      if (ptrOne->base == ptrTwo->base) {
        if (!ptrOne->isSubsetEq(ptrTwo)) {
          return false;
        }

        // subset check successful; advance both pointers
        ptrOne = ptrOne->next;
        ptrTwo = ptrTwo->next;
      } else if (ptrOne->base < ptrTwo->base) {
        // ptrTwo has overtaken ptrOne, i.e. one has something (a base)
        // two doesn't
        return false;
      } else {  // ptrOne > ptrTwo
        // greater than case; advance ptrTwo to see if it eventually
        // reaches what ptrOne is currently at
        ptrTwo = ptrTwo->next;
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
   * @param bit Bit that needs to be set
   * @returns a pair signifying a base word and the offset into a 
   * baseword that corresponds to bit
   */
  std::pair<unsigned, unsigned> getOffsets(unsigned bit) const {
    unsigned baseWord = bit / wordSize;
    unsigned offsetIntoWord = bit % wordSize;
      
    return std::pair<unsigned, unsigned>(baseWord, offsetIntoWord);
  }

  /**
   * @returns number of bits set by all words in this bitvector
   */
  unsigned count() const {
    unsigned nbits = 0;

    for (OneWord *ptr = head; ptr; ptr = ptr->next) {
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
  template<typename VectorTy>
  VectorTy getAllSetBits() const {
    VectorTy setBits;

    // loop through all words in the bitvector and get their set bits
    for (OneWord* curPtr = head; curPtr != nullptr; curPtr = curPtr->next) {
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
    std::vector<unsigned> setBits = getAllSetBits<std::vector<unsigned>>();
    out << "Elements(" << setBits.size() << "): ";

    for (auto setBitNum : setBits) {
      out << prefix << setBitNum << ", ";
    }

    out << "\n";
  }
};

} // end namespace galois

#endif //  _GALOIS_SPARSEBITVECTOR_H
