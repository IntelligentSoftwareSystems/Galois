/** Scalable priority worklist -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
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
#ifndef GALOIS_WORKLIST_ORDEREDLIST_H
#define GALOIS_WORKLIST_ORDEREDLIST_H

#include "Galois/FlatMap.h"

namespace Galois {
namespace WorkList {

template<class Compare = std::less<int>, typename T = int, bool concurrent = true>
class OrderedList : private boost::noncopyable, private Runtime::LL::PaddedLock<concurrent> {
  typedef Galois::flat_map<T, std::deque<T>, Compare> Map;

  Map map;

  using Runtime::LL::PaddedLock<concurrent>::lock;
  using Runtime::LL::PaddedLock<concurrent>::try_lock;
  using Runtime::LL::PaddedLock<concurrent>::unlock;

public:
  template<bool newconcurrent>
  struct rethread { typedef OrderedList<Compare, T, newconcurrent> type; };

  template<typename Tnew>
  struct retype { typedef OrderedList<Compare, Tnew, concurrent> type; };

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
    if (Runtime::LL::getTID() == 0)
      push(range.begin(), range.end());
  }

  Galois::optional<value_type> pop() {
    lock();
    if (map.empty()) {
      unlock();
      return Galois::optional<value_type>();
    }
    auto ii = map.begin();
    std::deque<T>& list = ii->second;
    Galois::optional<value_type> v(list.front());
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
