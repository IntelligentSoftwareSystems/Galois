/**
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of XYZ License (a
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

// TODO this can be templatized with BV; very common functions/fields......

#ifndef GALOIS_SPARSEBITVECTORLL_H
#define GALOIS_SPARSEBITVECTORLL_H

#include "SparseBitVector.h"

namespace galois {

/**
 * Extensino of SparseBitVector class that adds a next pointer, an
 * ID, and various functions for linked list purposes
 */
struct SparseBitVectorWithNext : public SparseBitVector {
  SparseBitVectorWithNext* next;
  unsigned id;

  /**
   * Initialize fields to default.
   */
  SparseBitVectorWithNext() : next(nullptr), id(0) {}

  /**
   * Default id
   */
  SparseBitVectorWithNext(unsigned _id) : next(nullptr), id(_id) {}

  /**
   * Clone just this bitvector without cloning its linked list.
   *
   * @returns a pointer to a copy of this bitvector
   */
  SparseBitVectorWithNext* clone() const {
    SparseBitVectorWithNext* vectorClone = new SparseBitVectorWithNext(id);

    vectorClone->head = head->cloneAll();

    return vectorClone;
  }

  /**
   * Clone this bitvector as well as its entire linked list from this bitvector
   * on.
   *
   * @returns clone of linked list starting with this bitvector
   */
  SparseBitVectorWithNext* cloneAll() const {
    SparseBitVectorWithNext* vectorBegin = clone();

    SparseBitVectorWithNext* curPtr  = vectorBegin;
    SparseBitVectorWithNext* nextPtr = next;

    // clone down the linked list starting from this pointer
    while (nextPtr != nullptr) {
      curPtr->next = nextPtr->clone();
      nextPtr      = nextPtr->next;
      curPtr       = curPtr->next;
    }

    return vectorBegin;
  }
};

struct SparseBVLinkedList {
  SparseBitVectorWithNext* head; // head of linked list

  SparseBVLinkedList() { head = nullptr; }

  /**
   * Set the bit on the bitvector with the specified id, creating the bitvector
   * if it doesn't exist.
   *
   * @param id
   * @param bit
   * @returns True if a new bit was set
   */
  bool set(unsigned id, unsigned bit) {
    SparseBitVectorWithNext* curPtr = head;
    SparseBitVectorWithNext* prev   = nullptr;

    // pointers should be in sorted order
    // loop through linked list to find the correct id (if it exists)
    while (curPtr != nullptr && curPtr->id < id) {
      prev   = curPtr;
      curPtr = curPtr->next;
    }

    // if id already exists, then set the correct bit
    if (curPtr != nullptr && curPtr->id == id) {
      return curPtr->set(bit);
      // else the id wasn't found; create and set, then rearrange linked list
      // accordingly
    } else {
      SparseBitVectorWithNext* newVector = new SparseBitVectorWithNext(id);
      newVector->set(bit);

      // this should point to prev's next, prev should point to this
      if (prev) {
        newVector->next = prev->next;
        prev->next      = newVector;
      } else {
        if (curPtr == nullptr) {
          // this is the first vector we are adding since both prev and head are
          // null; next is nothing
          newVector->next = nullptr;
        } else {
          // this new vector goes before curptr; if prev is null and curptr
          // isn't, it means it had to go before
          newVector->next = head;
        }

        head = newVector;
      }

      return true;
    }
  }

  /**
   * Union of 2 different linked lists; everything in second goes into this
   * linked list.
   *
   * @param second linked list to merge this one with
   * @returns a non-negative value if something changed
   */
  unsigned unify(const SparseBVLinkedList& second) {
    unsigned changed = 0;

    SparseBitVectorWithNext* prev   = nullptr;
    SparseBitVectorWithNext* ptrOne = head;
    SparseBitVectorWithNext* ptrTwo = second.head;

    while (ptrOne != nullptr && ptrTwo != nullptr) {
      if (ptrOne->id == ptrTwo->id) {
        // merge then advance both
        changed += ptrOne->unify(*ptrTwo);

        prev   = ptrOne;
        ptrOne = ptrOne->next;
        ptrTwo = ptrTwo->next;
      } else if (ptrOne->id < ptrTwo->id) {
        // advance our pointer until we reach "new" ids
        prev   = ptrOne;
        ptrOne = ptrOne->next;
      } else { // oneBase > twoBase
        // add ptrTwo's id that we don't have (otherwise would have been
        // handled by other case), fix linked list on our side
        SparseBitVectorWithNext* newWord = ptrTwo->clone();
        newWord->next = ptrOne; // this word comes before our current word

        // if previous vector exists, make it point to this new word,
        // else make this word the new head and prev pointer
        if (prev) {
          prev->next = newWord;
          prev       = newWord;
        } else {
          head = prev = newWord;
        }

        // done with ptrTwo's vector, advance
        ptrTwo = ptrTwo->next;

        changed++;
      }
    }

    // ptrOne = nullptr, but ptrTwo still has values; clone the values and
    // add them to our own bitvector
    if (ptrTwo) {
      SparseBitVectorWithNext* remaining = ptrTwo->cloneAll();

      if (prev) {
        prev->next = remaining;
      } else {
        head = remaining;
      }

      changed++;
    }

    return changed;
  }
};

} // end namespace galois

#endif
