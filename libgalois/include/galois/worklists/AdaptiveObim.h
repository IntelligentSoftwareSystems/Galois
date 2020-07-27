/** Scalable priority worklist
 *
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
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

#ifndef GALOIS_WORKLIST_ADAPTIVEOBIM_H
#define GALOIS_WORKLIST_ADAPTIVEOBIM_H

#include <atomic>
#include <cmath>
#include <iostream>
#include <limits>
#include <type_traits>

#include "galois/config.h"

#include "galois/FlatMap.h"
#include "galois/Timer.h"
#include "galois/substrate/PaddedLock.h"
#include "galois/substrate/PerThreadStorage.h"
#include "galois/worklists/Chunk.h"
#include "galois/worklists/WorkListHelpers.h"

namespace galois {
namespace worklists {

namespace internal {

template <typename Index, bool UseDescending>
struct AdaptiveOrderedByIntegerMetricComparator {
  typedef std::less<Index> compare_t;
  Index identity;
  Index earliest;

  AdaptiveOrderedByIntegerMetricComparator()
      : identity(std::numeric_limits<Index>::max()),
        earliest(std::numeric_limits<Index>::min()) {}
};

template <typename Index>
struct AdaptiveOrderedByIntegerMetricComparator<Index, true> {
  typedef std::greater<Index> compare_t;
  Index identity;
  Index earliest;

  AdaptiveOrderedByIntegerMetricComparator()
      : identity(std::numeric_limits<Index>::min()),
        earliest(std::numeric_limits<Index>::max()) {}
};

} // namespace internal

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
 * typedef galois::worklists::AdaptiveOrderedByIntegerMetric<Indexer> WL;
 * galois::for_each<WL>(items.begin(), items.end(), Fn);
 * \endcode
 *
 * @tparam Indexer        Indexer class
 * @tparam Container      Scheduler for each bucket
 * @tparam BlockPeriod    Check for higher priority work every 2^BlockPeriod
 *                        iterations
 * @tparam BSP            Use back-scan prevention
 * @tparam uniformBSP     Use uniform back-scan prevention
 * @tparam T              Work item type
 * @tparam Index          Indexer return type
 * @tparam UseMonotonic   Assume that an activity at priority p will not
 * schedule work at priority p or any priority p1 where p1 < p.
 * @tparam UseDescending  Use descending order instead
 * @tparam Concurrent     Whether or not to allow concurrent execution
 */
template <class Indexer      = DummyIndexer<int>,
          typename Container = PerSocketChunkFIFO<>, int BlockPeriod = 0,
          bool BSP = true, bool uniformBSP = false, int chunk_size = 64,
          typename T = int, typename Index = int, bool EnableUmerge = false,
          bool UseMonotonic = false, bool UseDescending = false,
          bool Concurrent = true>
struct AdaptiveOrderedByIntegerMetric
    : private boost::noncopyable,
      public internal::AdaptiveOrderedByIntegerMetricComparator<Index,
                                                                UseDescending> {
  template <typename _T>
  using retype = AdaptiveOrderedByIntegerMetric<
      Indexer, typename Container::template retype<_T>, BlockPeriod, BSP,
      uniformBSP, chunk_size, _T, typename std::result_of<Indexer(_T)>::type,
      EnableUmerge, UseMonotonic, UseDescending, Concurrent>;

  template <bool _b>
  using rethread = AdaptiveOrderedByIntegerMetric<
      Indexer, typename Container::template rethread<_b>, BlockPeriod, BSP,
      uniformBSP, chunk_size, T, Index, EnableUmerge, UseMonotonic,
      UseDescending, _b>;

  template <unsigned _period>
  struct with_block_period {
    typedef AdaptiveOrderedByIntegerMetric<
        Indexer, Container, _period, BSP, uniformBSP, chunk_size, T, Index,
        EnableUmerge, UseMonotonic, UseDescending, Concurrent>
        type;
  };
  template <typename _container>
  struct with_container {
    typedef AdaptiveOrderedByIntegerMetric<
        Indexer, _container, BlockPeriod, BSP, uniformBSP, chunk_size, T, Index,
        EnableUmerge, UseMonotonic, UseDescending, Concurrent>
        type;
  };

  template <typename _indexer>
  struct with_indexer {
    AdaptiveOrderedByIntegerMetric<
        _indexer, Container, BlockPeriod, BSP, uniformBSP, chunk_size, T, Index,
        EnableUmerge, UseMonotonic, UseDescending, Concurrent>
        type;
  };

  template <bool _bsp>
  struct with_back_scan_prevention {
    typedef AdaptiveOrderedByIntegerMetric<
        Indexer, Container, BlockPeriod, _bsp, uniformBSP, chunk_size, T, Index,
        EnableUmerge, UseMonotonic, UseDescending, Concurrent>
        type;
  };

  template <bool _enable_unmerge>
  struct with_unmerge {
    AdaptiveOrderedByIntegerMetric<
        Indexer, Container, BlockPeriod, BSP, uniformBSP, chunk_size, T, Index,
        _enable_unmerge, UseMonotonic, UseDescending, Concurrent>
        type;
  };

  template <bool _use_monotonic>
  struct with_monotonic {
    AdaptiveOrderedByIntegerMetric<
        Indexer, Container, BlockPeriod, BSP, uniformBSP, chunk_size, T, Index,
        EnableUmerge, _use_monotonic, UseDescending, Concurrent>
        type;
  };

  template <bool _use_descending>
  struct with_descending {
    AdaptiveOrderedByIntegerMetric<
        Indexer, Container, BlockPeriod, BSP, uniformBSP, chunk_size, T, Index,
        EnableUmerge, UseMonotonic, _use_descending, Concurrent>
        type;
  };

  typedef T value_type;
  typedef Index index_type;
  typedef uint32_t delta_type;

private:
  typedef typename Container::template rethread<Concurrent> CTy;
  typedef internal::AdaptiveOrderedByIntegerMetricComparator<Index,
                                                             UseDescending>
      Comparator;
  static inline typename Comparator::compare_t compare;
  delta_type delta;
  unsigned int counter;
  unsigned int maxIndex;
  unsigned int lastSizeMasterLog;

  // indexing mechanism
  // smaller delta insertions are prioritirized
  struct deltaIndex {
    Index k; // note: original index is stored here
    delta_type d;
    // taking the max of deltas and doing right shift eq. shifting priority with
    // d-max(d1, d2)

    deltaIndex() : k(0), d(0) {}
    deltaIndex(Index k1, delta_type d1) : k(k1), d(d1) {}
    bool operator==(const deltaIndex& a) const {
      unsigned delt = std::max(d, a.d);
      Index a1      = k >> delt;
      Index a2      = a.k >> delt;
      return (a1 == a2 && d == a.d);
    }
    bool operator<(const deltaIndex& a) const {
      unsigned delt = std::max(d, a.d);
      Index a1      = k >> delt;
      Index a2      = a.k >> delt;
      if (compare(a1, a2))
        return true;
      if (compare(a2, a1))
        return false;
      if (d < a.d)
        return true;
      return false;
    }
    bool operator>(const deltaIndex& a) const {
      unsigned delt = std::max(d, a.d);
      Index a1      = k >> delt;
      Index a2      = a.k >> delt;
      if (compare(a2, a1))
        return true;
      if (compare(a1, a2))
        return false;
      if (d > a.d)
        return true;
      return false;
    }
    bool operator<=(const deltaIndex& a) const {
      unsigned delt = std::max(d, a.d);
      Index a1      = k >> delt;
      Index a2      = a.k >> delt;
      if (compare(a1, a2))
        return true;
      if (compare(a2, a1))
        return false;
      if (d < a.d)
        return true;
      if (d == a.d)
        return true;
      return false;
    }
    bool operator>=(const deltaIndex& a) const {
      unsigned delt = std::max(d, a.d);
      Index a1      = k >> delt;
      Index a2      = a.k >> delt;
      if (compare(a2, a1))
        return true;
      if (compare(a1, a2))
        return false;
      if (d > a.d)
        return true;
      if (d == a.d)
        return true;
      return false;
    }
  };

  typedef galois::flat_map<deltaIndex, CTy*> LMapTy;

  struct ThreadData {
    LMapTy local;
    deltaIndex curIndex;
    deltaIndex scanStart;
    CTy* current;
    unsigned int lastMasterVersion;
    unsigned int numPops;

    unsigned int popsLastFix;
    unsigned int slowPopsLastPeriod;
    unsigned int pushesLastPeriod;
    unsigned int popsFromSameQ;
    struct {
      size_t pmodAllDeq;
      unsigned int priosLastPeriod;
      unsigned int numUmerges;
      Index maxPrioDiffLastPeriod;
    } stats;
    Index minPrio;
    Index maxPrio;
    substrate::PaddedLock<Concurrent> lock;

    void cleanup() {
      popsLastFix        = 0;
      slowPopsLastPeriod = 0;
      pushesLastPeriod   = 0;

      stats.priosLastPeriod       = 0;
      stats.maxPrioDiffLastPeriod = 0;

      minPrio = std::numeric_limits<Index>::max();
      maxPrio = std::numeric_limits<Index>::min();
    }

    inline bool isSlowPopFreq(double threshold) {
      // return ((double)slowPopsLastPeriod / (double)popsLastFix) > threshold;
      return ((double)slowPopsLastPeriod > (double)popsLastFix) * threshold;
    }

    ThreadData(Index initial)
        : curIndex(initial, 0), scanStart(initial, 0), current(0),
          lastMasterVersion(0), numPops(0), popsLastFix(0),
          slowPopsLastPeriod(0), pushesLastPeriod(0),
          popsFromSameQ(0), stats{0, 0, 0, 0},
          minPrio(std::numeric_limits<Index>::max()),
          maxPrio(std::numeric_limits<Index>::min()) {}
  };

  typedef std::deque<std::pair<deltaIndex, CTy*>> MasterLog;

  // NB: Place dynamically growing masterLog after fixed-size PerThreadStorage
  // members to give higher likelihood of reclaiming PerThreadStorage
  substrate::PerThreadStorage<ThreadData> data;
  substrate::PaddedLock<Concurrent> masterLock;
  MasterLog masterLog;

  galois::runtime::FixedSizeHeap heap;
  std::atomic<unsigned int> masterVersion;
  Indexer indexer;

  bool updateLocal(ThreadData& p) {
    if (p.lastMasterVersion != masterVersion.load(std::memory_order_relaxed)) {
      for (;
           p.lastMasterVersion < masterVersion.load(std::memory_order_relaxed);
           ++p.lastMasterVersion) {
        // XXX(ddn): Somehow the second block is better than
        // the first for bipartite matching (GCC 4.7.2)
#if 0
        p.local.insert(masterLog[p.lastMasterVersion]);
#else
        std::pair<deltaIndex, CTy*> logEntry = masterLog[p.lastMasterVersion];
        p.local[logEntry.first]              = logEntry.second;
        assert(logEntry.second);
#endif
      }
      return true;
    }
    return false;
  }

  GALOIS_ATTRIBUTE_NOINLINE
  galois::optional<T> slowPop(ThreadData& p) {
    // Failed, find minimum bin
    p.slowPopsLastPeriod++;
    unsigned myID = galois::substrate::ThreadPool::getTID();

    // first give it some time
    // then check the fdeq frequency
    if (myID == 0 && p.popsLastFix > counter &&
        p.isSlowPopFreq(1.0 / (double)(chunk_size))) {
      unsigned long numPushesThisStep      = p.pushesLastPeriod;
      unsigned long priosCreatedThisPeriod = p.stats.priosLastPeriod;
      unsigned long allPmodDeqCounts       = p.stats.pmodAllDeq;
      Index minOfMin                       = p.minPrio;
      Index maxOfMax                       = p.maxPrio;
      p.cleanup();
      for (unsigned i = 1; i < runtime::activeThreads; ++i) {
        while (!data.getRemote(i)->lock.try_lock())
          ;

        Index& otherMinPrio = data.getRemote(i)->minPrio;
        minOfMin            = std::min(minOfMin, otherMinPrio, compare);
        Index& otherMaxPrio = data.getRemote(i)->maxPrio;
        maxOfMax            = std::max(otherMaxPrio, maxOfMax, compare);
        numPushesThisStep += data.getRemote(i)->pushesLastPeriod;
        priosCreatedThisPeriod += data.getRemote(i)->stats.priosLastPeriod;
        allPmodDeqCounts += data.getRemote(i)->stats.pmodAllDeq;

        data.getRemote(i)->cleanup();
        data.getRemote(i)->lock.unlock();
      }

      if ((double)numPushesThisStep) {
        Index prioRange = (maxOfMax >> delta) - (minOfMin >> delta);
        // Division is expensive
        // double fillRatio = ((double)numPushesThisStep / (double)prioRange);
        if (numPushesThisStep < (chunk_size >> 1) * prioRange) {
          // Ditto
          // double xx = ((double)(chunk_size) / fillRatio);
          double xx = std::log2(chunk_size) - std::log2(numPushesThisStep) +
                      std::log2(prioRange);
          assert(xx);
          delta += std::floor(xx);
          counter <<= 1;
        }
      }
    }
    // serif added here
    // make sure delta is bigger than 0 so that we can actually unmerge things
    // give it some time and check the same queue pops
    else if (EnableUmerge && delta > 0 && myID == 0 &&
             p.popsLastFix > counter && p.popsFromSameQ > (chunk_size << 2)) {
      if (((p.maxPrio >> delta) - (p.minPrio >> delta)) < 16 &&
          ((double)p.pushesLastPeriod /
           ((double)((p.maxPrio >> delta) - (p.minPrio >> delta)))) >
              4 * chunk_size) { // this is a check to make sure we are also
                                // pushing with the same frequency end of
                                // execution
        double diff = ((p.maxPrio >> delta) - (p.minPrio >> delta)) >= 1
                          ? ((p.maxPrio >> delta) - (p.minPrio >> delta))
                          : 1;
        double xx = 16 / diff;
        if (delta > (unsigned int)(std::floor(std::log2(xx))))
          delta -= (unsigned int)(std::floor(std::log2(xx)));
        else
          delta = 0;

        p.cleanup();
        for (unsigned i = 1; i < runtime::activeThreads; ++i) {
          while (!data.getRemote(i)->lock.try_lock())
            ;
          data.getRemote(i)->cleanup();
          data.getRemote(i)->lock.unlock();
        }
        p.stats.numUmerges++;
      }
      p.popsFromSameQ = 0;
    }
    // p.popsFromSameQ = 0;

    bool localLeader = substrate::ThreadPool::isLeader();
    deltaIndex msS(this->earliest, 0);

    updateLocal(p);

    if (BSP && !UseMonotonic) {
      msS = p.scanStart;
      if (localLeader || uniformBSP) {
        for (unsigned i = 0; i < runtime::activeThreads; ++i) {
          msS = std::min(msS, data.getRemote(i)->scanStart);
        }
      } else {
        msS = std::min(
            msS, data.getRemote(substrate::ThreadPool::getLeader())->scanStart);
      }
    }

    for (auto ii = p.local.lower_bound(msS), ei = p.local.end(); ii != ei;
         ++ii) {
      galois::optional<T> item;
      if ((item = ii->second->pop())) {
        p.current   = ii->second;
        p.curIndex  = ii->first;
        p.scanStart = ii->first;
        p.lock.unlock();
        return item;
      }
    }

    p.lock.unlock();
    return galois::optional<value_type>();
  }

  GALOIS_ATTRIBUTE_NOINLINE
  CTy* slowUpdateLocalOrCreate(ThreadData& p, deltaIndex i) {
    // update local until we find it or we get the write lock
    do {
      updateLocal(p);
      CTy* lC;
      if ((lC = p.local[i]))
        return lC;
    } while (!masterLock.try_lock());
    // we have the write lock, update again then create
    updateLocal(p);
    CTy*& C2 = p.local[i];
    if (!C2) {
      C2                  = new (heap.allocate(sizeof(CTy))) CTy();
      p.lastMasterVersion = masterVersion.load(std::memory_order_relaxed) + 1;
      masterLog.push_back(std::make_pair(i, C2));
      masterVersion.fetch_add(1);
      p.stats.priosLastPeriod++;
    }
    masterLock.unlock();
    return C2;
  }

  inline CTy* updateLocalOrCreate(ThreadData& p, deltaIndex i) {
    // Try local then try update then find again or else create and update the
    // master log
    CTy* lC;
    if ((lC = p.local[i]))
      return lC;
    // slowpath
    return slowUpdateLocalOrCreate(p, i);
  }

public:
  AdaptiveOrderedByIntegerMetric(const Indexer& x = Indexer())
      : data(this->earliest), heap(sizeof(CTy)), masterVersion(0), indexer(x) {
    delta   = 0;
    counter = chunk_size;
  }

  ~AdaptiveOrderedByIntegerMetric() {
    ThreadData& p = *data.getLocal();
    updateLocal(p);
    // Deallocate in LIFO order to give opportunity for simple garbage
    // collection
    // Print stats for priroity counts here
    for (auto ii = masterLog.rbegin(), ei = masterLog.rend(); ii != ei; ++ii) {
      CTy* lC = ii->second;
      lC->~CTy();
      heap.deallocate(lC);
    }
  }

  void push(const value_type& val) {
    deltaIndex index;
    ThreadData& p = *data.getLocal();
    while (!p.lock.try_lock())
      ;

    p.pushesLastPeriod++;
    index.k = indexer(val);
    index.d = delta;
    if (index.k > p.maxPrio) {
      p.maxPrio = index.k;
    }
    if (index.k < p.minPrio) {
      p.minPrio = index.k;
    }

    // Fast path
    if (index == p.curIndex && p.current) {
      p.current->push(val);
      p.lock.unlock();
      return;
    }

    // Slow path
    CTy* C = updateLocalOrCreate(p, index);
    if (BSP && index < p.scanStart)
      p.scanStart = index;
    // Opportunistically move to higher priority work
    if (index < p.curIndex) {
      // we moved to a higher prio
      p.popsFromSameQ = 0;

      p.curIndex = index;
      p.current  = C;
    }
    C->push(val);

    p.lock.unlock();
  }

  template <typename Iter>
  size_t push(Iter b, Iter e) {
    size_t npush;
    for (npush = 0; b != e; npush++)
      push(*b++);
    return npush;
  }

  template <typename RangeTy>
  size_t push_initial(const RangeTy& range) {
    auto rp = range.local_pair();
    return push(rp.first, rp.second);
  }

  galois::optional<value_type> pop() {
    ThreadData& p = *data.getLocal();
    while (!p.lock.try_lock())
      ;
    CTy* C = p.current;

    p.popsLastFix++;
    p.stats.pmodAllDeq++;

    if (BlockPeriod && ((p.numPops++ & ((1ull << BlockPeriod) - 1)) == 0))
      return slowPop(p);

    galois::optional<value_type> item;
    if (C && (item = C->pop())) {
      p.popsFromSameQ++;

      p.lock.unlock();
      return item;
    }

    // Slow path
    return slowPop(p);
  }
};
GALOIS_WLCOMPILECHECK(AdaptiveOrderedByIntegerMetric)

} // end namespace worklists
} // end namespace galois

#endif
