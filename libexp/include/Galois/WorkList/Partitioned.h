/** Partitioned Worklist -*- C++ -*-
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
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#ifndef GALOIS_WORKLIST_PARTITIONED_H
#define GALOIS_WORKLIST_PARTITIONED_H

#include "Galois/WorkList/Simple.h"
#include "Galois/WorkList/WorkListHelpers.h"
#include "Galois/Substrate/PerThreadStorage.h"
#include "Galois/Substrate/CacheLineStorage.h"

#include <atomic>
#include <deque>

namespace Galois {
namespace WorkList {

template<typename Indexer = DummyIndexer<int>, typename Container = GFIFO<>,
  int BlockPeriod=0,
  int MaxValue=64,
  typename T=int,
  bool Concurrent=true>
struct Partitioned : private boost::noncopyable {
  static_assert(MaxValue > 0, "MaxValue must be positive");

  template<bool _concurrent>
  using rethread = Partitioned<Indexer, Container, BlockPeriod, MaxValue, T, _concurrent>;

  template<typename _T>
  using retype = Partitioned<Indexer, typename Container::template retype<_T>, BlockPeriod, MaxValue, _T, Concurrent>;

  template<int _period>
  using with_block_period = Partitioned<Indexer, Container, _period, MaxValue, T, Concurrent>;

  template<int _max_value>
  using with_max_value = Partitioned<Indexer, Container, BlockPeriod, _max_value, T, Concurrent>;

  template<typename _container>
  using with_container = Partitioned<Indexer, _container, BlockPeriod, MaxValue, T, Concurrent>;

  template<typename _indexer>
  using with_indexer = Partitioned<_indexer, Container, BlockPeriod, MaxValue, T, Concurrent>;

  typedef T value_type;

private:
  typedef typename Container::template rethread<Concurrent> CTy;

  //Runtime::PerThreadStorage<unsigned> numPops;
  std::deque<CTy> items;
  Indexer indexer;
  Substrate::CacheLineStorage<std::atomic_int> current;
  Substrate::CacheLineStorage<std::atomic_bool> empty;

  //XXX
  //if (BlockPeriod && (p.numPops++ & ((1<<BlockPeriod)-1)) == 0 && betterBucket(p))

  GALOIS_ATTRIBUTE_NOINLINE
  Galois::optional<value_type> slowPop(int cur) {
    Galois::optional<value_type> r;

    if (empty.data.load(std::memory_order_acquire))
      return r;

    for (int i = 1; i < MaxValue; ++i) {
      int c = cur + i;
      if (c >= MaxValue)
        c -= MaxValue;
      r = items[c].pop();
      if (r) {
        if (current.data.load(std::memory_order_acquire) == cur)
          current.data.store(c, std::memory_order_release);
        return r;
      }
    }
    return r;
  }

public:
  Partitioned(const Indexer& x = Indexer()): indexer(x), current(0), empty(false) { 
    for (int i = 0; i < MaxValue; ++i)
      items.emplace_back();
  }

  ~Partitioned() {
    // Be explicit about deallocation order to improve likelihood of simple
    // garbage collection
    while (!items.empty())
      items.pop_back();
  }

  void push(const value_type& val)  {
    int index = indexer(val);
    if (index < 0 || index >= MaxValue)
      index = 0;
    items[index].push(val);
    if (empty.data.load(std::memory_order_acquire))
      empty.data.store(false, std::memory_order_release);
  }

  template<typename Iter>
  void push(Iter b, Iter e) {
    while (b != e)
      push(*b++);
  }

  template<typename RangeTy>
  void push_initial(RangeTy range) {
    auto rp = range.local_pair();
    push(rp.first, rp.second);
  }

  Galois::optional<value_type> pop() {
    int c = current.data.load(std::memory_order_acquire);
    Galois::optional<value_type> r = items[c].pop();
    if (r)
      return r;
    return slowPop(c);
  }
};
GALOIS_WLCOMPILECHECK(Partitioned)

template<typename Indexer = DummyIndexer<int>, typename Container = GFIFO<>,
  typename T=int,
  bool Concurrent=true>
struct ThreadPartitioned : private boost::noncopyable {
  template<bool _concurrent>
  using rethread = ThreadPartitioned<Indexer, Container, T, _concurrent>;

  template<typename _T>
  using retype = ThreadPartitioned<Indexer, typename Container::template retype<_T>, _T, Concurrent>;

  template<typename _container>
  using with_container = ThreadPartitioned<Indexer, _container, T, Concurrent>;

  template<typename _indexer>
  using with_indexer = ThreadPartitioned<_indexer, Container, T, Concurrent>;

  typedef T value_type;

private:
  typedef typename Container::template rethread<Concurrent> CTy;

  struct Item {
    CTy wl;
    std::atomic_bool empty;
    Item(): empty(true) { }
  };

  Substrate::PerThreadStorage<Item> items;
  std::deque<unsigned> mapping;
  Indexer indexer;
  Substrate::CacheLineStorage<std::atomic_bool> empty;
  unsigned mask;

  //XXX
  //if (BlockPeriod && (p.numPops++ & ((1<<BlockPeriod)-1)) == 0 && betterBucket(p))

  GALOIS_ATTRIBUTE_NOINLINE
  Galois::optional<value_type> slowPop() {
    Galois::optional<value_type> r;

    for (int i = 0; i < items.size(); ++i) {
      r = items.getRemote(i)->wl.pop();
      if (r) {
        return r;
      }
    }
    return r;
  }

public:
  ThreadPartitioned(const Indexer& x = Indexer()): indexer(x), empty(false) {
    unsigned n = Galois::getActiveThreads();

    for (mask = 1; mask < n; mask <<= 1)
      ;
    for (unsigned i = 0; i < mask; ++i) 
      mapping.push_back(i % n);
    mask -= 1;
  }

  void push(const value_type& val)  {
    int index = mapping[indexer(val) & mask];
    Item* item = items.getRemote(index);
    item->wl.push(val);
  }

  template<typename Iter>
  void push(Iter b, Iter e) {
    while (b != e)
      push(*b++);
  }

  template<typename RangeTy>
  void push_initial(RangeTy range) {
    auto rp = range.local_pair();
    push(rp.first, rp.second);
  }

  Galois::optional<value_type> pop() {
    Galois::optional<value_type> r = items.getLocal()->wl.pop();
    //if (r)
      return r;
    //return slowPop();
  }
};
GALOIS_WLCOMPILECHECK(ThreadPartitioned)
}
}
#endif
