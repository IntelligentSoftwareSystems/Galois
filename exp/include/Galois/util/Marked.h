/** Marked object -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
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
 *
 * @section e.g. Mark for removal, 
 *
 * Billiards Simulation Finding Partial Order
 *
 * @author <ahassaan@ices.utexas.edu>
 */

#ifndef GALOIS_UTIL_MARKED_H
#define GALOIS_UTIL_MARKED_H

#include <climits>

#include "Galois/Runtime/PerThreadWorkList.h"

template <typename T>
struct Marked: public T {
  private:
    static const unsigned MAX_VAL = UINT_MAX;

  public:
    unsigned ver;

    Marked (T _obj)
      : T (_obj), ver (MAX_VAL) 
    {}

    void mark (unsigned v) {
      assert (v < MAX_VAL);
      ver = v;
    }

    bool marked () const { return (ver < MAX_VAL); }

    unsigned version () const { return ver; }

};

template <typename WLTy, typename IterTy, typename R>
void initPerThreadMarked (IterTy begin, IterTy end, WLTy& workList, 
    R (WLTy::ContTy::*pushFn) (const typename WLTy::value_type&)  ) {

  const unsigned P = GaloisRuntime::ThreadPool::getActiveThreads ();

  typedef typename std::iterator_traits<IterTy>::difference_type DiffTy;
  typedef typename std::iterator_traits<IterTy>::value_type ValueTy;

  // integer division, where we want to round up. So adding P-1
  DiffTy block_size = (std::distance (begin, end) + (P-1) ) / P;

  assert (block_size >= 1);

  IterTy block_begin = begin;

  for (unsigned i = 0; i < P; ++i) {

    IterTy block_end = block_begin;

    if (std::distance (block_end, end) < block_size) {
      block_end = end;

    } else {
      std::advance (block_end, block_size);
    }

    for (; block_begin != block_end; ++block_begin) {
      // workList[i].push_back (Marked<ValueTy> (*block_begin));
      (workList[i].*pushFn) (Marked<ValueTy> (*block_begin));
    }

    if (block_end == end) {
      break;
    }
  }
}

#endif // GALOIS_UTIL_MARKED_H
