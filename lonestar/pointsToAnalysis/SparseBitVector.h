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
 * TODO WHAT IS THIS
 * 
 * @author Rupesh Nasre <rupesh0508@gmail.com>
 * @author Loc Hoang <l_hoang@utexas.edu> (documentation, fixes, cleanup)
 */
#ifndef GALOIS_SPARSEBITVECTOR_H
#define GALOIS_SPARSEBITVECTOR_H

#include "galois/substrate/SimpleLock.h"

#include <vector>
#include <string>
#include <ostream>

namespace galois {

/**
 * Sparse bit vector.
 * 
 * Stores objects as indices in sparse bit vectors.
 * Saves space when the data to be stored is sparsely populated.
 */
struct SparseBitVector {
  using WORD = unsigned long;

  // Number of bits in a word
  static const unsigned wordsize = sizeof(WORD) * 8;

  struct OneWord {
    WORD bits; // number that is used as the bitset
    unsigned base; // used to order the words of the vector
    struct OneWord* next; // pointer to next word on linked list 
                          // (using base as order)

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

    unsigned unify(OneWord *second) {
      if (second) {
        WORD oribits = count();
        bits |= second->bits;
        return count() - oribits;
      }
      return 0;
    }

    unsigned count() {
      unsigned numElements = 0;
      WORD powerof2 = 1;

      for (unsigned ii = 0; ii < wordsize; ++ii) {
        if (bits & powerof2) {
                ++numElements;
        }
        powerof2 <<= 1;
      }
      return numElements;
    }

    inline bool isSubsetEq(OneWord *second) {
      return (bits & second->bits) == bits;
    }

    OneWord *clone() {
      OneWord *newword = new OneWord();
      newword->base = base;
      newword->bits = bits;
      newword->next = 0;
      return newword;
    }
    OneWord *cloneAll() {
      OneWord *newlist = clone();
      OneWord *ptr2;

      for (OneWord *newlistptr = newlist, *ptr = next; ptr;) {
        newlistptr->next = ptr->clone();
        ptr2 = ptr->next; 
        ptr = ptr2;
        newlistptr = newlistptr->next;
      }
      return newlist;
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
    unsigned getAllSetBits(VectorTy &setbits) {
      // or mask used to mask set bits
      WORD orMask = 1;
      unsigned numSet = 0;

      for (unsigned curBit = 0; curBit < wordsize; ++curBit) {
        if (bits & orMask) {
          setbits.push_back(base * wordsize + curBit);
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
   * @param bit The bit to set in the bitvector
   * @returns true if the bit set wasn't set previously
   */
  bool set(unsigned bit) {
    unsigned baseWord;
    unsigned offsetIntoWord;

    std::tie(baseWord, offsetIntoWord) = getOffsets(bit);

    OneWord* curPtr = head;
    OneWord* prev = nullptr;

    // pointers should be in sorted order TODO check this assumption
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
      // else this is the first word we are adding
      } else {
        newWord->next = nullptr;
        head = newWord;
      }

      return true;
    }
  }

  unsigned unify(SparseBitVector &second) {
    unsigned nchanged = 0;
    OneWord *prev = 0, *ptrone, *ptrtwo;
    for (ptrone = head, ptrtwo = second.head; ptrone && ptrtwo;) {
      if (ptrone->base == ptrtwo->base) {
        nchanged += ptrone->unify(ptrtwo);
        prev = ptrone; ptrone = ptrone->next;
        ptrtwo = ptrtwo->next;
      } else if (ptrone->base < ptrtwo->base) {
        prev = ptrone; 
        ptrone = ptrone->next;
      } else {
        OneWord *newword = ptrtwo->clone();
        newword->next = ptrone;
        if (prev) {
          prev->next = newword;
          prev = newword;
        } else {
          head = prev = newword;
        }
        ptrtwo = ptrtwo->next;
      }
    }
    if (ptrtwo) {
      OneWord *remaining = ptrtwo->cloneAll();
      if (prev) {
        prev->next = remaining;
      } else if (ptrtwo) {
        head = remaining;
      }
    }
    return nchanged;
  }
  bool isSubsetEq(SparseBitVector &second) {
    OneWord *ptrone, *ptrtwo;
    for (ptrone = head, ptrtwo = second.head; ptrone && ptrtwo; ptrone = ptrone->next) {
      if (ptrone->base == ptrtwo->base) {
        if (!ptrone->isSubsetEq(ptrtwo)) {
          return false;
        }
        ptrtwo = ptrtwo->next;
      } else if (ptrone->base > ptrtwo->base) {
          return false;
      }
    }
    if (ptrone) {
      return false;
    }
    return true;
  }

  /**
   * @param bit Bit that needs to be set
   * @returns a pair signifying a base word and the offset into a 
   * baseword that corresponds to bit
   */
  std::pair<unsigned, unsigned> getOffsets(unsigned bit) const {
    unsigned baseWord = bit / wordsize;
    unsigned offsetIntoWord = bit % wordsize;
      
    return std::pair<unsigned, unsigned>(baseWord, offsetIntoWord);
  }

  unsigned count() {
    unsigned nbits = 0;
    for (OneWord *ptr = head; ptr; ptr = ptr->next) {
      nbits += ptr->count();
    }
    return nbits;
  }

  /**
   * Gets the set bits in this bitvector and adds them to the passed in 
   * vector.
   *
   * @tparam VectorTy vector type that supports push_back
   * @param setBits Vector to add set bits to
   * @returns Number of set bits in this bitvector
   */
  template<typename VectorTy>
  unsigned getAllSetBits(VectorTy &setBits) {
    unsigned numBits = 0;

    // loop through all words in the bitvector and get their set bits
    for (OneWord* curPtr = head; curPtr != nullptr; curPtr = curPtr->next) {
      numBits += curPtr->getAllSetBits(setBits);
    }

    return numBits;
  }
  void print(std::ostream& out, std::string prefix = std::string("")) {
    std::vector<unsigned> setbits;
    unsigned nnodes = getAllSetBits(setbits);
    out << "Elements(" << nnodes << "): ";
    for (std::vector<unsigned>::iterator ii = setbits.begin(); ii != setbits.end(); ++ii) {
      out << prefix << *ii << ", ";
    }
    out << "\n";
  }
};
}

#endif //  _GALOIS_SPARSEBITVECTOR_H
