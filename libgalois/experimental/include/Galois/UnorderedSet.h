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

#ifndef GALOIS_UNORDERED_SET_H
#define GALOIS_UNORDERED_SET_H

#include "Galois/Substrate/PaddedLock.h"
#include "Galois/Substrate/CompilerSpecific.h"

#include <algorithm>
#include <unordered_set>

#include "Galois/Mem.h"

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
  template<typename _T>
  using retype = ThreadSafeUnorderedSet<_T>;

  explicit ThreadSafeUnorderedSet() {}

  template <typename Iter>
  ThreadSafeUnorderedSet(Iter b, Iter e)
  {
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
    bool ret = (s > 0);
    mutex.unlock();

    return ret;
  }

  void clear () {
    mutex.lock ();
    unorderedSet.clear ();
    mutex.unlock ();
  }

  const_iterator begin() const { return unorderedSet.begin(); }
  const_iterator end() const { return unorderedSet.end(); }
};

}

#endif
