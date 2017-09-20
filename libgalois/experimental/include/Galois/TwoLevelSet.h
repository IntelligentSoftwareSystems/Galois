/** TODO -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, version 2.1 of the
 * License.
 *
 * Galois is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Galois.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @section Copyright
 *
 * Copyright (C) 2015, The University of Texas at Austin. All rights
 * reserved.
 *
 * @section Description
 *
 * @author <ahassaan@ices.utexas.edu>
 * @author <yishanlu@cs.utexas.edu>
 */

#ifndef GALOIS_TWO_LEVEL_SET_H
#define GALOIS_TWO_LEVEL_SET_H

#include "Galois/PriorityQueue.h"
#include "Galois/UnorderedSet.h"

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
  Set *set;
  uintptr_t numFirstLevel;

private:
  uintptr_t index(const value_type& val) const 
  {
    return (uintptr_t)val % numFirstLevel; 
  }

public:
  template<typename _T>
  using retype = ThreadSafeTwoLevelSetMaster<_T, typename Set::template retype<_T> >;

  explicit ThreadSafeTwoLevelSetMaster(uintptr_t nFL = 0x1000)
    :numFirstLevel(nFL)
  {
    set = new Set[numFirstLevel];
  }

  ~ThreadSafeTwoLevelSetMaster() {
    delete[] set;
  }

  template <typename Iter>
  ThreadSafeTwoLevelSetMaster(Iter b, Iter e, uintptr_t nFL = 0x1000)
    :numFirstLevel(nFL) 
  {
    set = new Set[numFirstLevel];
    for (; b != e; ++b)
      set[index(*b)].push(*b);
  }

  bool empty() const {
    for(uintptr_t i = 0; i < numFirstLevel; i++)
      if(!set[i].empty())
        return false;
    return true;
  }

  size_type size() const {
    size_type sz = 0;
    for(uintptr_t i = 0; i < numFirstLevel; i++)
      sz += set[i].size();
    return sz;
  }

  bool find(const value_type& x) const {
    return set[index(x)].find(x);
  }

  bool push(const value_type& x) {
    return set[index(x)].push(x);
  }

  bool remove(const value_type& x) {
    return set[index(x)].remove(x);
  }

  void clear() {
    for(uintptr_t i = 0; i < numFirstLevel; i++) {
      set[i].clear();
    }
  }

  // FIXME: should use two-level iterators
  const_iterator begin() const { return set[0].begin(); }
  const_iterator end() const { return set[numFirstLevel-1].end(); }
};

template<typename T>
using ThreadSafeTwoLevelSet = ThreadSafeTwoLevelSetMaster<T, galois::ThreadSafeOrderedSet<T> >;

template<typename T>
using ThreadSafeTwoLevelHash = ThreadSafeTwoLevelSetMaster<T, galois::ThreadSafeUnorderedSet<T> >;
}

#endif
