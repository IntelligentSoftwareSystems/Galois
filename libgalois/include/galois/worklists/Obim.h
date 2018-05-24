/** Scalable priority worklist -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galois is a framework to exploit
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

#ifndef GALOIS_WORKLIST_OBIM_H
#define GALOIS_WORKLIST_OBIM_H

#include "galois/FlatMap.h"
#include "galois/runtime/Substrate.h"
#include "galois/substrate/PerThreadStorage.h"
#include "galois/substrate/Termination.h"
#include "galois/worklists/Chunked.h"
#include "galois/worklists/WorkListHelpers.h"

#include <deque>
#include <limits>
#include <type_traits>

namespace galois {
namespace worklists {

namespace internal {

template<typename T, typename Index, bool UseBarrier>
class OrderedByIntegerMetricData {
protected:
  struct ThreadData { };
  bool hasStored(ThreadData&, Index) { return false; }
  galois::optional<T> popStored(ThreadData&, Index) { return {}; }
};

template<typename T, typename Index>
class OrderedByIntegerMetricData<T, Index, true> {
protected:
  struct ThreadData {
    bool hasWork;
    std::deque<std::pair<Index,T>> stored;
  };

  substrate::Barrier& barrier;

  OrderedByIntegerMetricData(): barrier(runtime::getBarrier(runtime::activeThreads)) { }

  bool hasStored(ThreadData& p, Index idx) {
    for (auto& e : p.stored) {
      if (e.first == idx) {
        std::swap(e, p.stored.front());
        return true;
      }
    }
    return false;
  }

  galois::optional<T> popStored(ThreadData& p, Index idx) {
    galois::optional<T> item;
    for (auto ii = p.stored.begin(), ei = p.stored.end(); ii != ei; ++ii) {
      if (ii->first == idx) {
        item = ii->second;
        p.stored.erase(ii);
        break;
      }
    }
    return item;
  }
};

template<typename Index, bool UseDescending>
struct OrderedByIntegerMetricComparator {
  std::less<Index> compare;
  Index identity;

  template<typename C>
  struct with_local_map {
    typedef galois::flat_map<Index, C, std::less<Index> > type;
  };
  OrderedByIntegerMetricComparator(): identity(std::numeric_limits<Index>::min()) {}
};

template<typename Index>
struct OrderedByIntegerMetricComparator<Index, true> {
  std::greater<Index> compare;
  Index identity;

  template<typename C>
  struct with_local_map {
    typedef galois::flat_map<Index, C, std::greater<Index> > type;
  };
  OrderedByIntegerMetricComparator(): identity(std::numeric_limits<Index>::max()) {}
};

}

/**
 * Approximate priority scheduling. Indexer is a default-constructable class
 * whose instances conform to <code>R r = indexer(item)</code> where R is some
 * type with a total order defined by <code>operator&lt;</code> and
 * <code>operator==</code> and item is an element from the Galois set
 * iterator.
 *
 * An example:
 * \code
 * struct Item { int index; };
 *
 * struct Indexer {
 *   int operator()(Item i) const { return i.index; }
 * };
 *
 * typedef galois::worklists::OrderedByIntegerMetric<Indexer> WL;
 * galois::for_each<WL>(galois::iterate(items), Fn);
 * \endcode
 *
 * @tparam Indexer        Indexer class
 * @tparam Container      Scheduler for each bucket
 * @tparam BlockPeriod    Check for higher priority work every 2^BlockPeriod
 *                        iterations
 * @tparam BSP            Use back-scan prevention
 * @tparam UseBarrier     Eliminate priority inversions by placing a barrier between
 *                        priority levels
 * @tparam UseMonotonic   Assume that an activity at priority p will not schedule
 *                        work at priority p or any priority p1 where p1 < p.
 * @tparam UseDescending  Use descending order instead
 */
// TODO could move to general comparator but there are issues with atomic reads and initial values
// for arbitrary types
template<class Indexer = DummyIndexer<int>, typename Container = dChunkedFIFO<>,
  unsigned BlockPeriod=0,
  bool BSP=true,
  typename T=int,
  typename Index=int,
  bool UseBarrier=false,
  bool UseMonotonic=false,
  bool UseDescending=false,
  bool Concurrent=true>
struct OrderedByIntegerMetric :
  private boost::noncopyable,
  public internal::OrderedByIntegerMetricData<T, Index, UseBarrier>,
  public internal::OrderedByIntegerMetricComparator<Index, UseDescending>
{
  //static_assert(std::is_integral<Index>::value, "only integral index types supported");

  template<typename _T>
  using retype = OrderedByIntegerMetric<Indexer, typename Container::template retype<_T>, BlockPeriod, BSP, _T, typename std::result_of<Indexer(_T)>::type, UseBarrier, UseMonotonic, UseDescending, Concurrent>;

  template<bool _b>
  using rethread = OrderedByIntegerMetric<Indexer, Container, BlockPeriod, BSP, T,Index,UseBarrier,UseMonotonic,UseDescending,_b>;

  template<unsigned _period>
  struct with_block_period { typedef OrderedByIntegerMetric<Indexer, Container, _period, BSP, T, Index, UseBarrier, UseMonotonic, UseDescending, Concurrent> type; };

  template<typename _container>
  struct with_container { typedef OrderedByIntegerMetric<Indexer, _container, BlockPeriod, BSP, T, Index, UseBarrier, UseMonotonic, UseDescending, Concurrent> type; };

  template<typename _indexer>
  struct with_indexer { typedef OrderedByIntegerMetric<_indexer, Container, BlockPeriod, BSP, T, Index, UseBarrier, UseMonotonic, UseDescending, Concurrent> type; };

  template<bool _bsp>
  struct with_back_scan_prevention { typedef OrderedByIntegerMetric<Indexer, Container, BlockPeriod, _bsp, T, Index, UseBarrier, UseMonotonic, UseDescending, Concurrent> type; };

  template<bool _use_barrier>
  struct with_barrier { typedef OrderedByIntegerMetric<Indexer, Container, BlockPeriod, BSP, T, Index, _use_barrier, UseMonotonic, UseDescending, Concurrent> type; };

  template<bool _use_monotonic>
  struct with_monotonic { typedef OrderedByIntegerMetric<Indexer, Container, BlockPeriod, BSP, T, Index, UseBarrier, _use_monotonic, UseDescending, Concurrent> type; };

  template<bool _use_descending>
  struct with_descending { typedef OrderedByIntegerMetric<Indexer, Container, BlockPeriod, BSP, T, Index, UseBarrier, UseMonotonic, _use_descending, Concurrent> type; };

  typedef T value_type;
  typedef Index index_type;

private:
  typedef typename Container::template rethread<Concurrent> CTy;
  typedef internal::OrderedByIntegerMetricComparator<Index, UseDescending> Comparator;
  typedef typename Comparator::template with_local_map<CTy*>::type LMapTy;

  struct ThreadData: public internal::OrderedByIntegerMetricData<T, Index, UseBarrier>::ThreadData {
    LMapTy local;
    Index curIndex;
    Index scanStart;
    CTy* current;
    unsigned int lastMasterVersion;
    unsigned int numPops;

    ThreadData(Index initial) :
      curIndex(initial),
      scanStart(initial),
      current(0), lastMasterVersion(0), numPops(0) { }
  };

  typedef std::deque<std::pair<Index, CTy*> > MasterLog;

  // NB: Place dynamically growing masterLog after fixed-size PerThreadStorage
  // members to give higher likelihood of reclaiming PerThreadStorage
  substrate::PerThreadStorage<ThreadData> data;
  substrate::PaddedLock<Concurrent> masterLock;
  MasterLog masterLog;

  std::atomic<unsigned int> masterVersion;
  Indexer indexer;

  bool updateLocal(ThreadData& p) {
    if (p.lastMasterVersion != masterVersion.load(std::memory_order_relaxed)) {
      for (; p.lastMasterVersion < masterVersion.load(std::memory_order_relaxed); ++p.lastMasterVersion) {
        // XXX(ddn): Somehow the second block is better than
        // the first for bipartite matching (GCC 4.7.2)
#if 0
        p.local.insert(masterLog[p.lastMasterVersion]);
#else
        std::pair<Index, CTy*> logEntry = masterLog[p.lastMasterVersion];
        p.local[logEntry.first] = logEntry.second;
        assert(logEntry.second);
#endif
      }
      return true;
    }
    return false;
  }

  GALOIS_ATTRIBUTE_NOINLINE
  galois::optional<T> slowPop(ThreadData& p) {
    bool localLeader = substrate::ThreadPool::isLeader();
    Index msS = this->identity;

    updateLocal(p);

    if (BSP && !UseMonotonic) {
      msS = p.scanStart;
      if (localLeader) {
        for (unsigned i = 0; i < runtime::activeThreads; ++i) {
          Index o = data.getRemote(i)->scanStart;
          if (this->compare(o, msS))
            msS = o;
        }
      } else {
        Index o = data.getRemote(substrate::ThreadPool::getLeader())->scanStart;
        if (this->compare(o, msS))
          msS = o;
      }
    }

    for (auto ii = p.local.lower_bound(msS), ei = p.local.end(); ii != ei; ++ii) {
      galois::optional<T> item;
      if ((item = ii->second->pop())) {
        p.current = ii->second;
        p.curIndex = ii->first;
        p.scanStart = ii->first;
        return item;
      }
    }

    return galois::optional<value_type>();
  }

  GALOIS_ATTRIBUTE_NOINLINE
  CTy* slowUpdateLocalOrCreate(ThreadData& p, Index i) {
    //update local until we find it or we get the write lock
    do {
      CTy* C;
      updateLocal(p);
      if ((C = p.local[i]))
        return C;
    } while (!masterLock.try_lock());
    //we have the write lock, update again then create
    updateLocal(p);
    CTy*& C2 = p.local[i];
    if (!C2) {
      C2 = new CTy();
      p.lastMasterVersion = masterVersion.load(std::memory_order_relaxed) + 1;
      masterLog.push_back(std::make_pair(i, C2));
      masterVersion.fetch_add(1);
    }
    masterLock.unlock();
    return C2;
  }

  inline CTy* updateLocalOrCreate(ThreadData& p, Index i) {
    //Try local then try update then find again or else create and update the master log
    CTy* C;
    if ((C = p.local[i]))
      return C;
    //slowpath
    return slowUpdateLocalOrCreate(p, i);
  }

public:
  OrderedByIntegerMetric(const Indexer& x = Indexer()):
    data(this->identity),
    masterVersion(0), indexer(x) { }

  ~OrderedByIntegerMetric() {
    // Deallocate in LIFO order to give opportunity for simple garbage
    // collection
    for (auto ii = masterLog.rbegin(), ei = masterLog.rend(); ii != ei; ++ii) {
      delete ii->second;
    }
  }

  void push(const value_type& val) {
    Index index = indexer(val);
    ThreadData& p = *data.getLocal();

    assert(!UseMonotonic || this->compare(p.curIndex, index));

    // Fast path
    if (index == p.curIndex && p.current) {
      p.current->push(val);
      return;
    }

    // Slow path
    CTy* C = updateLocalOrCreate(p, index);
    if (BSP && this->compare(index, p.scanStart))
      p.scanStart = index;
    // Opportunistically move to higher priority work
    if (!UseBarrier && this->compare(index, p.curIndex)) {
      p.curIndex = index;
      p.current = C;
    }
    C->push(val);
  }

  template<typename Iter>
  void push(Iter b, Iter e) {
    while (b != e)
      push(*b++);
  }

  template<typename RangeTy>
  void push_initial(const RangeTy& range) {
    auto rp = range.local_pair();
    push(rp.first, rp.second);
  }

  galois::optional<value_type> pop() {
    // Find a successful pop
    ThreadData& p = *data.getLocal();
    CTy* C = p.current;

    if (this->hasStored(p, p.curIndex))
      return this->popStored(p, p.curIndex);

    if (!UseBarrier && BlockPeriod && (p.numPops++ & ((1<<BlockPeriod)-1)) == 0)
      return slowPop(p);

    galois::optional<value_type> item;
    if (C && (item = C->pop()))
      return item;

    if (UseBarrier)
      return item;

    // Slow path
    return slowPop(p);
  }

  template<bool Barrier=UseBarrier>
  auto empty() -> typename std::enable_if<Barrier, bool>::type {
    galois::optional<value_type> item;
    ThreadData& p = *data.getLocal();

    item = slowPop(p);
    if (item)
      p.stored.push_back(std::make_pair(p.curIndex, *item));
    p.hasWork = item;

    this->barrier.wait();

    bool hasWork = p.hasWork;
    Index curIndex = p.curIndex;
    CTy* C = p.current;

    for (unsigned i = 0; i < runtime::activeThreads; ++i) {
      ThreadData& o = *data.getRemote(i);
      if (o.hasWork && this->compare(o.curIndex, curIndex)) {
        curIndex = o.curIndex;
        C = o.current;
      }
      hasWork |= o.hasWork;
    }

    this->barrier.wait();

    p.current = C;
    p.curIndex = curIndex;

    if (UseMonotonic) {
      for (auto ii = p.local.begin(); ii != p.local.end(); ) {
        bool toBreak = ii->second == C;
        ii = p.local.erase(ii);
        if (toBreak)
          break;
      }
    }

    return !hasWork;
  }
};
GALOIS_WLCOMPILECHECK(OrderedByIntegerMetric)

} // end namespace worklists
} // end namespace galois

#endif
