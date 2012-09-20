/** Fixed-size ring buffer -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2012, The University of Texas at Austin. All rights reserved.
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
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#ifndef GALOIS_FIXEDSIZERING_H
#define GALOIS_FIXEDSIZERING_H

#include "Galois/LazyArray.h"

#include <boost/optional.hpp>

namespace Galois {

template<typename T, unsigned chunksize = 64>
class FixedSizeRing :private boost::noncopyable {
  unsigned start;
  unsigned count;
  LazyArray<T, chunksize> datac;

  T* at(unsigned i) {
    return &datac[i];
  }

public:
  typedef T value_type;

  FixedSizeRing() :start(0), count(0) {}
  ~FixedSizeRing() {
    clear();
  }

  unsigned size() const {
    return count;
  }

  bool empty() const {
    return count == 0;
  }

  bool full() const {
    return count == chunksize;
  }

  T& getAt(unsigned x) {
    return *at((start + x) % chunksize);
  }

  void clear() {
    for (unsigned x = 0; x < count; ++x)
      datac.kill((start + x) % chunksize);
    count = 0;
    start = 0;
  }

  value_type* push_front(const value_type& val) {
    if (full()) return 0;
    start = (start + chunksize - 1) % chunksize;
    ++count;
    datac.init(start,val);
    return &datac[start];
  }

  value_type* push_back(const value_type& val) {
    if (full()) return 0;
    int end = (start + count) % chunksize;
    ++count;
    datac.init(end,val);
    return &datac[end];
  }

  template<typename Iter>
  Iter push_back(Iter b, Iter e) {
    while (push_back(*b)) { ++b; }
    return b;
  }

  template<typename Iter>
  Iter push_front(Iter b, Iter e) {
    while (push_front(*b)) { ++b; }
    return b;
  }

  boost::optional<value_type> pop_front() {
    boost::optional<value_type> retval;
    if (!empty()) {
      retval = *at(start);
      datac.kill(start);
      start = (start + 1) % chunksize;
      --count;
    }
    return retval;
  }
  
  boost::optional<value_type> pop_back() {
    boost::optional<value_type> retval;
    if (!empty()) {
      int end = (start + count - 1) % chunksize;
      retval = *at(end);
      datac.kill(end);
      --count;
    }
    return retval;
  }
};
 
}

#endif
