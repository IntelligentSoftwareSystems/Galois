// Galois Managed Conflict type wrapper -*- C++ -*-
/*
Galois, a framework to exploit amorphous data-parallelism in irregular
programs.

Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS SOFTWARE
AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY, FITNESS FOR ANY
PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF PERFORMANCE, AND ANY
WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF DEALING OR USAGE OF TRADE.
NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH RESPECT TO THE USE OF THE
SOFTWARE OR DOCUMENTATION. Under no circumstances shall University be liable
for incidental, special, indirect, direct or consequential damages or loss of
profits, interruption of business, or related expenses which may arise from use
of Software or Documentation, including but not limited to those resulting from
defects in Software and/or Documentation, or loss or inaccuracy of data of any
kind.

@author rupesh nasre. <rupesh0508@gmail.com>
*/

#ifndef GALOIS_SPARSEBITVECTOR_H
#define GALOIS_SPARSEBITVECTOR_H

#include "galois/Substrate/SimpleLock.h"

#include <vector>
#include <string>
#include <ostream>

namespace galois {

/**
 * Concurrent version of sparse bit vector.
 * 
 * Stores objects as indices in sparse bit vectors.
 * Saves space when the data to be stored is sparsely populated.
 */
struct SparseBitVector {
  typedef unsigned long WORD;
  typedef galois::substrate::SimpleLock LockType;
  static const unsigned wordsize = sizeof(WORD)*8;

  struct OneWord {
    WORD bits;
    unsigned base;
    struct OneWord *next;
    LockType kulup;

    bool set(unsigned oo) {
      WORD oribits = bits;
      bits |= ((WORD)1 << oo);
      return bits != oribits;
    }

    OneWord(unsigned bb, unsigned oo) {
      base = bb;
      set(oo);
      next = 0;
    }

    OneWord() { }

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
        //ptr->lock();
        newlistptr->next = ptr->clone();
        ptr2 = ptr->next; 
        //ptr->unlock();
        ptr = ptr2;
        newlistptr = newlistptr->next;
      }
      return newlist;
    }
    void getAllSetBits(std::vector<unsigned> &setbits) {
      WORD powerof2 = 1;
      unsigned bitno = 0;

      for (unsigned ii = 0; ii < wordsize; ++ii) {
        if (bits & powerof2) {
          setbits.push_back(base*wordsize + bitno);
        }
        powerof2 <<= 1;
        ++bitno;
      }
    }
    void lock() {
            kulup.lock();
    }
    void unlock() {
            kulup.unlock();
    }
  };

  OneWord *head;
  LockType headkulup;

  SparseBitVector() {
    init(0);
  }
  void init() {
    init(0);
  }
  void init(unsigned nelements) {
    head = 0;
  }
  void lock() {
    headkulup.lock();
  }
  void unlock() {
    headkulup.unlock();
  }
  bool set(unsigned bit) {
    unsigned base, offset;
    getOffsets(bit, base, offset);

    OneWord *ptr, *prev;
    ptr = head;
    prev = 0;
    for (; ptr && ptr->base <= base; ptr = ptr->next) {  // sorted order.
      if (ptr->base == base) {
        return ptr->set(offset);
      }
      prev = ptr;
    }
    OneWord *newword = new OneWord(base, offset);
    if (prev) {
      //prev->lock();
      newword->next = prev->next;
      prev->next = newword;
      //prev->unlock();
    } else {
      //lock();
      newword->next = head;
      head = newword;
      //unlock();
    }
    return true;
  }
  unsigned unify(SparseBitVector &second) {
    unsigned nchanged = 0;
    OneWord *prev = 0, *ptrone, *ptrtwo;
    for (ptrone = head, ptrtwo = second.head; ptrone && ptrtwo;) {
      if (ptrone->base == ptrtwo->base) {
        //ptrone->lock();
        nchanged += ptrone->unify(ptrtwo);
        prev = ptrone; ptrone = ptrone->next;
        ptrtwo = ptrtwo->next;
        //prev->unlock();
      } else if (ptrone->base < ptrtwo->base) {
        prev = ptrone; 
        //prev->lock();
        ptrone = ptrone->next;
        //prev->unlock();
      } else {
        OneWord *newword = ptrtwo->clone();
        newword->next = ptrone;
        if (prev) {
          //prev->lock();
          prev->next = newword;
          //prev->unlock();
          prev = newword;
        } else {
          //lock();
          head = prev = newword;
          //unlock();
        }
        ptrtwo = ptrtwo->next;
      }
    }
    if (ptrtwo) {
      OneWord *remaining = ptrtwo->cloneAll();
      if (prev) {
        //prev->lock();
        prev->next = remaining;
        //prev->unlock();
      } else if (ptrtwo) {
        //lock();
        head = remaining;
        //unlock();
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
  inline void getOffsets(unsigned bit, unsigned &ventry, unsigned &wbit) {
    ventry = bit / wordsize;
    wbit = bit % wordsize;
  }
  unsigned count() {
    unsigned nbits = 0;
    for (OneWord *ptr = head; ptr; ptr = ptr->next) {
      nbits += ptr->count();
    }
    return nbits;
  }
  unsigned getAllSetBits(std::vector<unsigned> &setbits) {
    unsigned nnodes = 0;
    for (OneWord *ptr = head; ptr; ptr = ptr->next) {
      ptr->getAllSetBits(setbits);
      ++nnodes;
    }
    return nnodes;
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
