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

// #define UNMERGE_ENABLED

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
 * typedef galois::WorkList::AdaptiveOrderedByIntegerMetric<Indexer> WL;
 * galois::for_each<WL>(items.begin(), items.end(), Fn);
 * \endcode
 *
 * @tparam Indexer Indexer class
 * @tparam Container Scheduler for each bucket
 * @tparam BlockPeriod Check for higher priority work every 2^BlockPeriod
 *                     iterations
 * @tparam BSP Use back-scan prevention
 * @tparam uniformBSP Use uniform back-scan prevention
 * @tparam T Work item type
 * @tparam Index Indexer return type
 * @tparam UseMonotonic Assume that an activity at priority p will not
 *                      schedule work at priority p or any priority p1
 *                      where p1 < p.
 * @tparam UseDescending Use descending order instead
 * @tparam Concurrent Whether or not to allow concurrent execution
 *
 */
template <class Indexer      = DummyIndexer<int>,
          typename Container = PerSocketChunkFIFO<>, int BlockPeriod = 0,
          bool BSP = true, bool uniformBSP = false, int chunk_size = 64,
          typename T = int, typename Index = int, bool UseMonotonic = false,
          bool UseDescending = false, bool Concurrent = true>
struct AdaptiveOrderedByIntegerMetric : private boost::noncopyable {
  template <bool Concurrent_>
  using rethread = AdaptiveOrderedByIntegerMetric<
      Indexer, typename Container::template rethread<Concurrent_>, BlockPeriod,
      BSP, uniformBSP, chunk_size, T, Index, UseMonotonic, UseDescending,
      Concurrent_>;

  template <typename T_>
  using retype = AdaptiveOrderedByIntegerMetric<
      Indexer, typename Container::template retype<T_>, BlockPeriod, BSP,
      uniformBSP, chunk_size, T_, typename std::result_of<Indexer(T_)>::type,
      UseMonotonic, UseDescending, Concurrent>;

  template <unsigned BlockPeriod_>
  using with_block_period =
      AdaptiveOrderedByIntegerMetric<Indexer, Container, BlockPeriod_, BSP,
                                     uniformBSP, chunk_size, T, Index,
                                     UseMonotonic, UseDescending, Concurrent>;

  template <typename Container_>
  using with_container =
      AdaptiveOrderedByIntegerMetric<Indexer, Container_, BlockPeriod, BSP,
                                     uniformBSP, chunk_size, T, Index,
                                     UseMonotonic, UseDescending, Concurrent>;

  template <typename Indexer_>
  using with_indexer =
      AdaptiveOrderedByIntegerMetric<Indexer_, Container, BlockPeriod, BSP,
                                     uniformBSP, chunk_size, T, Index,
                                     UseMonotonic, UseDescending, Concurrent>;

  template <bool BSP_>
  using with_back_scan_prevention =
      AdaptiveOrderedByIntegerMetric<Indexer, Container, BlockPeriod, BSP_,
                                     uniformBSP, chunk_size, T, Index,
                                     UseMonotonic, UseDescending, Concurrent>;

  template <bool UseMonotonic_>
  using with_monotonic =
      AdaptiveOrderedByIntegerMetric<Indexer, Container, BlockPeriod, BSP,
                                     uniformBSP, chunk_size, T, Index,
                                     UseMonotonic_, UseDescending, Concurrent>;

  template <bool UseDescending_>
  using with_descending =
      AdaptiveOrderedByIntegerMetric<Indexer, Container, BlockPeriod, BSP,
                                     uniformBSP, chunk_size, T, Index,
                                     UseMonotonic, UseDescending_, Concurrent>;

  using compare_t =
      std::conditional_t<UseDescending, std::greater<Index>, std::less<Index>>;

  typedef T value_type;
  typedef Index index_type;

private:
  static inline compare_t compare;
  static constexpr Index earliest = UseDescending
                                        ? std::numeric_limits<Index>::min()
                                        : std::numeric_limits<Index>::max();
  // static constexpr identity = UseDescending ?
  // std::numeric_limits<Index>::max() : std::numeric_limits<Index>::min();
  unsigned int delta;
  unsigned int counter;
  unsigned int maxIndex;
  unsigned int lastSizeMasterLog;

  // indexing mechanism
  // smaller delta insertions are prioritirized
  struct deltaIndex {
    Index k; // note: original index is stored here
    unsigned int d;
    // taking the max of deltas and doing right shift eq. shifting priority with
    // d-max(d1, d2)

    deltaIndex() : k(0), d(0) {}
    deltaIndex(Index k1, unsigned int d1) : k(k1), d(d1) {}
    unsigned int id() { return k; }
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

  typedef typename Container::template rethread<Concurrent> CTy;
  typedef galois::flat_map<deltaIndex, CTy*> LMapTy;

  struct ThreadData {
    LMapTy local;
    deltaIndex curIndex;
    deltaIndex scanStart;
    CTy* current;
    unsigned int lastMasterVersion;
    unsigned int numPops;

    unsigned int sinceLastFix;
    unsigned int slowPopsLastPeriod;
    unsigned int pushesLastPeriod;
    unsigned int priosLastPeriod;

    unsigned long pmodAllDeq;

    unsigned int popsFromSameQ;
    unsigned int ctr;

    Index maxPrioDiffLastPeriod;
    Index minPrio;
    Index maxPrio;
    substrate::PaddedLock<Concurrent> lock;

    ThreadData(Index initial)
        : curIndex(initial, 0), scanStart(initial, 0), current(0),
          lastMasterVersion(0), numPops(0), sinceLastFix(0),
          slowPopsLastPeriod(0), pushesLastPeriod(0), priosLastPeriod(0),
          pmodAllDeq(0), popsFromSameQ(0), ctr(0), maxPrioDiffLastPeriod(0),
          minPrio(std::numeric_limits<Index>::max()),
          maxPrio(std::numeric_limits<Index>::min()) {}
  };

  typedef std::deque<std::pair<deltaIndex, CTy*>> MasterLog;

  // NB: Place dynamically growing masterLog after fixed-size PerThreadStorage
  // members to give higher likelihood of reclaiming PerThreadStorage
  substrate::PerThreadStorage<ThreadData> current;
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
    if (myID == 0 && p.sinceLastFix > counter &&
        ((double)(p.slowPopsLastPeriod) / (double)(p.sinceLastFix)) >
            1.0 / (double)(chunk_size)) {
      for (unsigned i = 1; i < runtime::activeThreads; ++i) {
        while (current.getRemote(i)->lock.try_lock())
          ;
      }
      unsigned long priosCreatedThisPeriod = 0;
      unsigned long numPushesThisStep      = 0;
      unsigned long allPmodDeqCounts       = 0;
      Index minOfMin                       = std::numeric_limits<Index>::max();
      Index maxOfMax                       = std::numeric_limits<Index>::min();
      for (unsigned i = 0; i < runtime::activeThreads; ++i) {
        Index& otherMinPrio = current.getRemote(i)->minPrio;
        minOfMin = compare(minOfMin, otherMinPrio) ? minOfMin : otherMinPrio;
        Index& otherMaxPrio = current.getRemote(i)->maxPrio;
        maxOfMax = compare(otherMaxPrio, maxOfMax) ? maxOfMax : otherMaxPrio;
        priosCreatedThisPeriod += current.getRemote(i)->priosLastPeriod;
        numPushesThisStep += current.getRemote(i)->pushesLastPeriod;
        allPmodDeqCounts += current.getRemote(i)->pmodAllDeq;
        current.getRemote(i)->sinceLastFix          = 0;
        current.getRemote(i)->slowPopsLastPeriod    = 0;
        current.getRemote(i)->pushesLastPeriod      = 0;
        current.getRemote(i)->priosLastPeriod       = 0;
        current.getRemote(i)->maxPrioDiffLastPeriod = 0;

        current.getRemote(i)->minPrio = std::numeric_limits<Index>::max();
        current.getRemote(i)->maxPrio = std::numeric_limits<Index>::min();
      }

      if (((double)numPushesThisStep /
           ((double)((maxOfMax >> delta) - (minOfMin >> delta)))) <
          chunk_size / 2) {
        double xx = ((double)(chunk_size) /
                     ((double)numPushesThisStep /
                      ((double)((maxOfMax >> delta) - (minOfMin >> delta)))));
        delta += std::floor(std::log2(xx));
        counter *= 2;
      }

      for (unsigned i = 1; i < runtime::activeThreads; ++i) {
        current.getRemote(i)->lock.unlock();
      }
    }
#ifdef UNMERGE_ENABLED
    // serif added here
    // make sure delta is bigger than 0 so that we can actually unmerge things
    // give it some time and check the same queue pops
    else if (delta > 0 && myID == 0 && p.sinceLastFix > counter &&
             p.popsFromSameQ > 4 * chunk_size) {
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

        for (unsigned i = 1; i < runtime::activeThreads; ++i) {
          while (current.getRemote(i)->lock.try_lock())
            ;
        }

        for (unsigned i = 0; i < runtime::activeThreads; ++i) {

          current.getRemote(i)->sinceLastFix          = 0;
          current.getRemote(i)->slowPopsLastPeriod    = 0;
          current.getRemote(i)->pushesLastPeriod      = 0;
          current.getRemote(i)->priosLastPeriod       = 0;
          current.getRemote(i)->maxPrioDiffLastPeriod = 0;

          current.getRemote(i)->minPrio = std::numeric_limits<Index>::max();
          current.getRemote(i)->maxPrio = std::numeric_limits<Index>::min();
        }

        for (unsigned i = 1; i < runtime::activeThreads; ++i) {
          current.getRemote(i)->lock.unlock();
        }
        p.ctr++;
      }
      p.popsFromSameQ = 0;
    }
#endif
    p.popsFromSameQ = 0;

    updateLocal(p);
    bool localLeader = substrate::ThreadPool::isLeader();

    deltaIndex msS;
    msS.k = std::numeric_limits<Index>::min();
    msS.d = 0;
    if (BSP && !UseMonotonic) {
      msS = p.scanStart;
      if (localLeader || uniformBSP) {
        for (unsigned i = 0; i < runtime::activeThreads; ++i) {
          msS = std::min(msS, current.getRemote(i)->scanStart);
        }
      } else {
        msS = std::min(
            msS,
            current.getRemote(substrate::ThreadPool::getLeader())->scanStart);
      }
    }

    for (auto ii = p.local.lower_bound(msS), ee = p.local.end(); ii != ee;
         ++ii) {
      galois::optional<T> retval;
      if ((retval = ii->second->pop())) {
        p.current   = ii->second;
        p.curIndex  = ii->first;
        p.scanStart = ii->first;
        p.lock.unlock();
        return retval;
      }
    }
    p.lock.unlock();
    return galois::optional<value_type>();
  }

  GALOIS_ATTRIBUTE_NOINLINE
  CTy* slowUpdateLocalOrCreate(ThreadData& p, deltaIndex i) {
    // update local until we find it or we get the write lock
    do {
      CTy* lC;
      updateLocal(p);
      if ((lC = p.local[i]))
        return lC;
    } while (!masterLock.try_lock());
    // we have the write lock, update again then create
    updateLocal(p);
    CTy*& lC2 = p.local[i];
    if (!lC2) {
      lC2                 = new (heap.allocate(sizeof(CTy))) CTy();
      p.lastMasterVersion = masterVersion.load(std::memory_order_relaxed) + 1;
      masterLog.push_back(std::make_pair(i, lC2));
      masterVersion.fetch_add(1);
      p.priosLastPeriod++;
    }
    masterLock.unlock();
    return lC2;
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
      : current(earliest), heap(sizeof(CTy)), masterVersion(0), indexer(x) {
    delta   = 0;
    counter = chunk_size;
  }

  ~AdaptiveOrderedByIntegerMetric() {
    ThreadData& p = *current.getLocal();
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
    ThreadData& p = *current.getLocal();
    while (!p.lock.try_lock())
      ;
    Index ind = indexer(val);
    deltaIndex index;
    index.k = ind;
    index.d = delta;
    if (index.k > p.maxPrio) {
      p.maxPrio = index.k;
    }
    if (index.k < p.minPrio) {
      p.minPrio = index.k;
    }
    p.pushesLastPeriod++;

    // Fast path
    if (index == p.curIndex && p.current) {
      p.current->push(val);
      p.lock.unlock();
      return;
    }

    // Slow path
    CTy* lC = updateLocalOrCreate(p, index);
    if (BSP && index < p.scanStart)
      p.scanStart = index;
    // Opportunistically move to higher priority work
    if (index < p.curIndex) {
      // we moved to a higher prio
      p.popsFromSameQ = 0;

      p.curIndex = index;
      p.current  = lC;
    }
    lC->push(val);
    p.lock.unlock();
  }

  template <typename Iter>
  unsigned int push(Iter b, Iter e) {
    int npush;
    for (npush = 0; b != e; npush++)
      push(*b++);
    return npush;
  }

  template <typename RangeTy>
  unsigned int push_initial(const RangeTy& range) {
    auto rp = range.local_pair();
    return push(rp.first, rp.second);
  }

  galois::optional<value_type> pop() {
    ThreadData& p = *current.getLocal();
    while (!p.lock.try_lock())
      ;

    p.sinceLastFix++;

    unsigned myID = galois::substrate::ThreadPool::getTID();

    current.getRemote(myID)->pmodAllDeq++;

    CTy* C = p.current;
    if (BlockPeriod &&
        (BlockPeriod < 0 || ((p.numPops++ & ((1ull << BlockPeriod) - 1)) == 0)))
      return slowPop(p);

    galois::optional<value_type> retval;
    if (C && (retval = C->pop())) {
      p.popsFromSameQ++;
      p.lock.unlock();
      return retval;
    }

    // Slow path
    return slowPop(p);
  }
};
GALOIS_WLCOMPILECHECK(AdaptiveOrderedByIntegerMetric)
} // namespace worklists
} // namespace galois

#endif
