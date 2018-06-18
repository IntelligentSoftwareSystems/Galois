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

#ifndef GALOIS_TWO_LEVEL_SET_H
#define GALOIS_TWO_LEVEL_SET_H

#include "galois/PriorityQueue.h"
#include "galois/UnorderedSet.h"

namespace galois {

/**
 * Thread-safe two-level set
 */
template <typename T, typename Set>
class ThreadSafeTwoLevelSetMaster {
public:
  typedef Set container_type;
  typedef typename container_type::value_type value_type;
  typedef typename container_type::reference reference;
  typedef typename container_type::const_reference const_reference;
  typedef typename container_type::pointer pointer;
  typedef typename container_type::size_type size_type;
  typedef typename container_type::const_iterator iterator;
  typedef typename container_type::const_iterator const_iterator;

private:
  Set* set;
  uintptr_t numFirstLevel;

private:
  uintptr_t index(const value_type& val) const {
    return (uintptr_t)val % numFirstLevel;
  }

public:
  template <typename _T>
  using retype =
      ThreadSafeTwoLevelSetMaster<_T, typename Set::template retype<_T>>;

  explicit ThreadSafeTwoLevelSetMaster(uintptr_t nFL = 0x1000)
      : numFirstLevel(nFL) {
    set = new Set[numFirstLevel];
  }

  ~ThreadSafeTwoLevelSetMaster() { delete[] set; }

  template <typename Iter>
  ThreadSafeTwoLevelSetMaster(Iter b, Iter e, uintptr_t nFL = 0x1000)
      : numFirstLevel(nFL) {
    set = new Set[numFirstLevel];
    for (; b != e; ++b)
      set[index(*b)].push(*b);
  }

  bool empty() const {
    for (uintptr_t i = 0; i < numFirstLevel; i++)
      if (!set[i].empty())
        return false;
    return true;
  }

  size_type size() const {
    size_type sz = 0;
    for (uintptr_t i = 0; i < numFirstLevel; i++)
      sz += set[i].size();
    return sz;
  }

  bool find(const value_type& x) const { return set[index(x)].find(x); }

  bool push(const value_type& x) { return set[index(x)].push(x); }

  bool remove(const value_type& x) { return set[index(x)].remove(x); }

  void clear() {
    for (uintptr_t i = 0; i < numFirstLevel; i++) {
      set[i].clear();
    }
  }

  // FIXME: should use two-level iterators
  const_iterator begin() const { return set[0].begin(); }
  const_iterator end() const { return set[numFirstLevel - 1].end(); }
};

template <typename T>
using ThreadSafeTwoLevelSet =
    ThreadSafeTwoLevelSetMaster<T, galois::ThreadSafeOrderedSet<T>>;

template <typename T>
using ThreadSafeTwoLevelHash =
    ThreadSafeTwoLevelSetMaster<T, galois::ThreadSafeUnorderedSet<T>>;
} // namespace galois

#endif
