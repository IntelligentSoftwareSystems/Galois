// Atomic Types type -*- C++ -*-
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
*/

#ifndef _GALOIS_UTIL_ATOMIC_H
#define _GALOIS_UTIL_ATOMIC_H

namespace Galois {

  //TODO: it may be a good idea to add buffering to these classes so that the object is store one per cache line

  class AtomicBool {
    volatile bool val;

  public:
    AtomicBool (bool val = false) :
      val (val) {
    }

    inline bool get () const {
      return val;
    }

    inline void set (bool v) {
      val = v;
    }

    inline bool cas (bool expected, bool updated) {
      return (__sync_bool_compare_and_swap (&val, expected, updated));
    }

  };

  class AtomicInteger {
    volatile int val;

  public:
    AtomicInteger (int val = 0) :
      val (val) {
    }
    ;

    inline int get () const {
      return val;
    }

    inline void set (int v) {
      val = v;
    }

    inline bool cas (int expected, int update) {
      return __sync_bool_compare_and_swap (&val, expected, update);
    }

    inline int addAndGet (int delta) {
      return __sync_add_and_fetch (&val, delta);
    }

    inline int incrementAndGet () {
      return addAndGet (1);
    }

    inline int decrementAndGet () {
      return addAndGet (-1);
    }

    inline int getAndAdd (int delta) {
      return __sync_fetch_and_add (&val, delta);
    }

    inline int getAndIncrement () {
      return getAndAdd (1);
    }

    inline int getAndDecrement () {
      return getAndAdd (-1);
    }

  };
}



#endif //  _GALOIS_UTIL_ATOMIC_H
