#ifndef GALOIS_SPARSEBITVECTOR_H
#define GALOIS_SPARSEBITVECTOR_H

#include <vector>
#include <string>
#include <ostream>
#include <utility>
#include <boost/iterator/iterator_facade.hpp>

// TODO
// - get rid of new usage (new doesn't scale well); smart pointers probably
// better as well
// - rename some things to be consistent with Concurrent Sparse Bit Vector

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

  //////////////////////////////////////////////////////////////////////////////
  // BEGIN NODE
  //////////////////////////////////////////////////////////////////////////////
  /**
   * A single word used as a bitvector. Contains functionality to alter it like
   * a bitvector.
   */
  struct SparseBitVectorNode {
    using WORD = unsigned long;
  
    WORD bits; // number that is used as the bitset
    unsigned base; // used to order the words of the vector
    struct SparseBitVectorNode* next; // pointer to next word on linked list 
                          // (using base as order)
  
    /**
     * Default is create a base at 0.
     */
    SparseBitVectorNode() { 
      SparseBitVectorNode(0);
    }
  
    /**
     * Creates a new word.
     *
     * @param _base base of this word, i.e. what order it should go in linked 
     * list
     */
    SparseBitVectorNode(unsigned _base) { 
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
    SparseBitVectorNode(unsigned _base, unsigned _initial) {
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
     * @param second SparseBitVectorNode to do a bitwise or with
     * @returns 1 if something changed, 0 otherwise
     */
    unsigned unify(SparseBitVectorNode* second) {
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
     * @param second SparseBitVectorNode pointer to compare against
     * @returns true if second word's bits has everything that this
     * word's bits have
     */
    bool isSubsetEq(SparseBitVectorNode* second) const {
      return (bits & second->bits) == bits;
    }
  
    /**
     * @returns a pointer to a copy of this word without the preservation
     * of the linked list
     */
    SparseBitVectorNode* clone() const {
      // TODO don't use new, find better way
      SparseBitVectorNode* newWord = new SparseBitVectorNode();
  
      newWord->base = base;
      newWord->bits = bits;
      newWord->next = nullptr;
  
      return newWord;
    }
  
    /**
     * @returns a pointer to a copy of this word WITH the preservation of
     * the linked list via copies of the list starting from this word
     */
    SparseBitVectorNode* cloneAll() const {
      SparseBitVectorNode* newListBeginning = clone();
  
      SparseBitVectorNode* curPtr = newListBeginning;
      SparseBitVectorNode* nextPtr = next;
  
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
  //////////////////////////////////////////////////////////////////////////////
  // END NODE
  //////////////////////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////////////////////////////
  // Begin Iterator
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
    SparseBitVectorNode* currentHead;
    unsigned currentBit;
    unsigned currentValue;
  
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
          currentHead = currentHead->next;
          currentBit = 0;
        }
      }
  
  
      if (currentHead != nullptr) {
        currentValue = (currentHead->base * wordSize) + 
                       currentBit;
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
  
    SBVIterator(SparseBitVectorNode* firstHead) 
        : currentHead(firstHead), currentBit(0), currentValue(-1) { 
      advanceToNextBit(true);
    }
  
    SBVIterator(SparseBitVector* bv) : currentBit(0), currentValue(-1) {
      currentHead = bv->head;
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
      } // do nothing if head is nullptr (i.e. the end)
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
    const unsigned& dereference() const {
      return currentValue;
    }
  };

  //////////////////////////////////////////////////////////////////////////////
  // End Iterator
  //////////////////////////////////////////////////////////////////////////////

  SparseBitVectorNode* head;

  /**
   * Constructor; inits head to nullptr
   */
  SparseBitVector() {
    init();
  }

  /**
   * @returns iterator to first set element of this bitvector
   */
  SBVIterator begin() {
    return SBVIterator(this);
  }

  /**
   * @returns end iterator of this bitvector.
   */
  SBVIterator end() {
    return SBVIterator();
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

    SparseBitVectorNode* curPtr = head;
    SparseBitVectorNode* prev = nullptr;

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
      SparseBitVectorNode *newWord = new SparseBitVectorNode(baseWord, 
                                                             offsetIntoWord);

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
    SparseBitVectorNode* curPointer = head;

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

    SparseBitVectorNode* prev = nullptr;
    SparseBitVectorNode* ptrOne = head;
    SparseBitVectorNode* ptrTwo = second.head;

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
        SparseBitVectorNode* newWord = ptrTwo->clone();
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
      SparseBitVectorNode* remaining = ptrTwo->cloneAll();

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
    SparseBitVectorNode* ptrOne = head;
    SparseBitVectorNode* ptrTwo = second.head;

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

    for (SparseBitVectorNode *ptr = head; ptr; ptr = ptr->next) {
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
    for (SparseBitVectorNode* curPtr = head; curPtr != nullptr; curPtr = curPtr->next) {
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
