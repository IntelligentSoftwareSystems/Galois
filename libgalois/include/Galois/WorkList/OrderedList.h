/** Scalable priority worklist -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
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
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#ifndef GALOIS_WORKLIST_ORDEREDLIST_H
#define GALOIS_WORKLIST_ORDEREDLIST_H

#include "Galois/FlatMap.h"

namespace galois {
namespace WorkList {

template<class Compare = std::less<int>, typename T = int, bool concurrent = true>
class OrderedList : private boost::noncopyable, private Substrate::PaddedLock<concurrent> {
  typedef galois::flat_map<T, std::deque<T>, Compare> Map;

  Map map;

  using Substrate::PaddedLock<concurrent>::lock;
  using Substrate::PaddedLock<concurrent>::try_lock;
  using Substrate::PaddedLock<concurrent>::unlock;

public:
  template<typename Tnew>
  using retype = OrderedList<Compare, Tnew, concurrent>;

  template<bool b>
  using rethread = OrderedList<Compare, T, b>;

  typedef T value_type;

  void push(value_type val) {
    lock();
    std::deque<T>& list = map[val];
    list.push_back(val);
    unlock();
  }

  template<typename Iter>
  void push(Iter b, Iter e) {
    lock();
    while (b != e) {
      std::deque<T>& list = map[*b];
      list.push_back(*b);
      ++b;
    }
    unlock();
  }

  template<typename RangeTy>
  void push_initial(RangeTy range) {
    if (Substrate::ThreadPool::getTID() == 0)
      push(range.begin(), range.end());
  }

  galois::optional<value_type> pop() {
    lock();
    if (map.empty()) {
      unlock();
      return galois::optional<value_type>();
    }
    auto ii = map.begin();
    std::deque<T>& list = ii->second;
    galois::optional<value_type> v(list.front());
    list.pop_front();
    if (list.empty())
      map.erase(ii);
    unlock();
    return v;
  }
};
GALOIS_WLCOMPILECHECK(OrderedList)
}
}
#endif
