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

  bool precondition() const {
    return count >= 0 && count <= chunksize && start >= 0 && start <= chunksize;
  }

public:
  typedef T value_type;
  typedef T* pointer;
  typedef T& reference;
  typedef const T& const_reference;

  FixedSizeRing() :start(0), count(0) { }

  ~FixedSizeRing() {
    clear();
  }

  unsigned size() const {
    assert(precondition());
    return count;
  }

  bool empty() const {
    assert(precondition());
    return count == 0;
  }

  bool full() const {
    assert(precondition());
    return count == chunksize;
  }

  reference getAt(unsigned x) {
    assert(precondition());
    assert(!empty());
    return *at((start + x) % chunksize);
  }

  void clear() {
    assert(precondition());
    for (unsigned x = 0; x < count; ++x)
      datac.kill((start + x) % chunksize);
    count = 0;
    start = 0;
  }

#ifdef GALOIS_HAS_RVALUE_REFERENCES
  template<typename U>
  pointer push_front(U&& val) {
    if (full()) return 0;
    start = (start + chunksize - 1) % chunksize;
    ++count;
    return datac.init(start, std::forward<U>(val));
  }

  template<typename... Args>
  pointer emplace_front(Args&&... args) {
    if (full()) return 0;
    start = (start + chunksize - 1) % chunksize;
    ++count;
    return datac.emplace(start, std::forward<Args>(args)...);
  }
#else
  pointer push_front(const value_type& val) {
    if (full()) return 0;
    start = (start + chunksize - 1) % chunksize;
    ++count;
    return datac.init(start,val);
  }
#endif

#ifdef GALOIS_HAS_RVALUE_REFERENCES
  template<typename U>
  pointer push_back(U&& val) {
    if (full()) return 0;
    unsigned end = (start + count) % chunksize;
    ++count;
    return datac.init(end, std::forward<U>(val));
  }

  template<typename... Args>
  pointer emplace_back(Args&&... args) {
    if (full()) return 0;
    unsigned end = (start + count) % chunksize;
    ++count;
    return datac.emplace(end, std::forward<Args>(args)...);
  }
#else
  pointer push_back(const value_type& val) {
    if (full()) return 0;
    unsigned end = (start + count) % chunksize;
    ++count;
    return datac.init(end,val);
  }
#endif

  reference front() {
    assert(precondition());
    assert(!empty());
    return *at(start);
  }

  const_reference front() const {
    assert(precondition());
    assert(!empty());
    return *at(start);
  }

  boost::optional<value_type> extract_front() {
    boost::optional<value_type> retval;
    if (!empty()) {
      retval = front();
      pop_front();
    }
    return retval;
  }

  void pop_front() {
    assert(precondition());
    assert(!empty());
    datac.kill(start);
    start = (start + 1) % chunksize;
    --count;
  }
  
  reference back() {
    assert(precondition());
    assert(!empty());
    return *at((start + count - 1) % chunksize);
  }

  const_reference back() const {
    assert(precondition());
    assert(!empty());
    return *at((start + count - 1) % chunksize); 
  }

  boost::optional<value_type> extract_back() {
    boost::optional<value_type> retval;
    if (!empty()) {
      retval = back();
      pop_back();
    }
    return retval;
  }

  void pop_back() {
    assert(precondition());
    assert(!empty());
    unsigned end = (start + count - 1) % chunksize;
    datac.kill(end);
    --count;
  }
};
 
}

#endif
