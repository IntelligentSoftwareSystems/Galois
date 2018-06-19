/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
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

#ifndef GALOIS_UNORDERED_SET_H
#define GALOIS_UNORDERED_SET_H

#include "galois/substrate/PaddedLock.h"
#include "galois/substrate/CompilerSpecific.h"

#include <algorithm>
#include <unordered_set>

#include "galois/Mem.h"

namespace galois {

/**
 * Thread-safe unordered set.
 */
template <typename T>
class ThreadSafeUnorderedSet {
  typedef std::unordered_set<T> Set;

public:
  typedef Set container_type;
  typedef typename container_type::value_type value_type;
  typedef typename container_type::reference reference;
  typedef typename container_type::const_reference const_reference;
  typedef typename container_type::pointer pointer;
  typedef typename container_type::size_type size_type;
  typedef typename container_type::const_iterator iterator;
  typedef typename container_type::const_iterator const_iterator;
  typedef galois::substrate::SimpleLock Lock_ty;

private:
  GALOIS_ATTRIBUTE_ALIGN_CACHE_LINE Lock_ty mutex;
  Set unorderedSet;

public:
  template <typename _T>
  using retype = ThreadSafeUnorderedSet<_T>;

  explicit ThreadSafeUnorderedSet() {}

  template <typename Iter>
  ThreadSafeUnorderedSet(Iter b, Iter e) {
    for (; b != e; ++b) {
      unorderedSet.insert(*b);
    }
  }

  bool empty() const {
    mutex.lock();
    bool ret = unorderedSet.empty();
    mutex.unlock();

    return ret;
  }

  size_type size() const {
    mutex.lock();
    size_type sz = unorderedSet.size();
    mutex.unlock();

    return sz;
  }

  bool find(const value_type& x) const {
    mutex.lock();
    bool ret = (unorderedSet.find(x) != unorderedSet.end());
    mutex.unlock();
    return ret;
  }

  bool push(const value_type& x) {
    mutex.lock();
    auto p = unorderedSet.insert(x);
    mutex.unlock();
    return p.second;
  }

  bool remove(const value_type& x) {
    mutex.lock();
    size_type s = unorderedSet.erase(x);
    bool ret    = (s > 0);
    mutex.unlock();

    return ret;
  }

  void clear() {
    mutex.lock();
    unorderedSet.clear();
    mutex.unlock();
  }

  const_iterator begin() const { return unorderedSet.begin(); }
  const_iterator end() const { return unorderedSet.end(); }
};

} // namespace galois

#endif
