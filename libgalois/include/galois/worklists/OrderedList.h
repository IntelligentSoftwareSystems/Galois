#ifndef GALOIS_WORKLIST_ORDEREDLIST_H
#define GALOIS_WORKLIST_ORDEREDLIST_H

#include "galois/FlatMap.h"

namespace galois {
namespace worklists {

template<class Compare = std::less<int>, typename T = int, bool concurrent = true>
class OrderedList : private boost::noncopyable, private substrate::PaddedLock<concurrent> {
  typedef galois::flat_map<T, std::deque<T>, Compare> Map;

  Map map;

  using substrate::PaddedLock<concurrent>::lock;
  using substrate::PaddedLock<concurrent>::try_lock;
  using substrate::PaddedLock<concurrent>::unlock;

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
    if (substrate::ThreadPool::getTID() == 0)
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
