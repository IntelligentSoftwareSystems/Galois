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
#ifndef GALOIS_WORKLIST_OBIM_H
#define GALOIS_WORKLIST_OBIM_H

#include "Galois/config.h"
#include "Galois/FlatMap.h"
#include "Galois/Runtime/PerThreadStorage.h"
#include "Galois/Runtime/Termination.h"
#include "Galois/WorkList/Chunked.h"
#include "Galois/WorkList/WorkListHelpers.h"

#include GALOIS_CXX11_STD_HEADER(type_traits)
#include <limits>

namespace Galois {
namespace WorkList {

namespace detail {

template<typename T, typename Index, bool UseBarrier>
class OrderedByIntegerMetricData { 
protected:
  struct ThreadData { };
  bool hasStored(ThreadData&, Index) { return false; }
  Galois::optional<T> popStored(ThreadData&, Index) { return {}; }
};

template<typename T, typename Index>
class OrderedByIntegerMetricData<T, Index, true> {
protected:
  struct ThreadData {
    bool hasWork;
    std::deque<std::pair<Index,T>> stored;
  };

  Runtime::TerminationDetection& term;
  Runtime::Barrier& barrier;

  OrderedByIntegerMetricData(): term(Runtime::getSystemTermination()), barrier(Runtime::getSystemBarrier()) { }

  bool hasStored(ThreadData& p, Index idx) {
    for (auto& e : p.stored) {
      if (e.first == idx) {
        std::swap(e, p.stored.front());
        return true;
      }
    }
    return false;
  }

  Galois::optional<T> popStored(ThreadData& p, Index idx) {
    Galois::optional<T> item;
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
 * typedef Galois::WorkList::OrderedByIntegerMetric<Indexer> WL;
 * Galois::for_each<WL>(items.begin(), items.end(), Fn);
 * \endcode
 *
 * @tparam Indexer Indexer class
 * @tparam Container Scheduler for each bucket
 * @tparam BlockPeriod Check for higher priority work every 2^BlockPeriod
 *                     iterations
 * @tparam BSP Use back-scan prevention
 */
template<class Indexer = DummyIndexer<int>, typename Container = dChunkedFIFO<>,
  unsigned BlockPeriod=0,
  bool BSP=true,
  typename T=int,
  typename Index=int,
  bool UseBarrier=false,
  bool Concurrent=true>
struct OrderedByIntegerMetric : private boost::noncopyable, public detail::OrderedByIntegerMetricData<T, Index, UseBarrier> {
  template<bool _concurrent>
  struct rethread { typedef OrderedByIntegerMetric<Indexer, typename Container::template rethread<_concurrent>::type, BlockPeriod, BSP, T, Index, UseBarrier, _concurrent> type; };

  template<typename _T>
  struct retype { typedef OrderedByIntegerMetric<Indexer, typename Container::template retype<_T>::type, BlockPeriod, BSP, _T, typename std::result_of<Indexer(_T)>::type, UseBarrier, Concurrent> type; };

  template<unsigned _period>
  struct with_block_period { typedef OrderedByIntegerMetric<Indexer, Container, _period, BSP, T, Index, UseBarrier, Concurrent> type; };

  template<typename _container>
  struct with_container { typedef OrderedByIntegerMetric<Indexer, _container, BlockPeriod, BSP, T, Index, UseBarrier, Concurrent> type; };

  template<typename _indexer>
  struct with_indexer { typedef OrderedByIntegerMetric<_indexer, Container, BlockPeriod, BSP, T, Index, UseBarrier, Concurrent> type; };

  template<bool _bsp>
  struct with_back_scan_prevention { typedef OrderedByIntegerMetric<Indexer, Container, BlockPeriod, _bsp, T, Index, UseBarrier, Concurrent> type; };

  template<bool _use_barrier>
  struct with_barrier { typedef OrderedByIntegerMetric<Indexer, Container, BlockPeriod, BSP, T, Index, _use_barrier, Concurrent> type; };

  typedef T value_type;
  typedef Index index_type;

private:
  typedef typename Container::template rethread<Concurrent>::type CTy;
  typedef Galois::flat_map<Index, CTy*> LMapTy;
  //typedef std::map<Index, CTy*> LMapTy;

  struct ThreadData: public detail::OrderedByIntegerMetricData<T, Index, UseBarrier>::ThreadData {
    LMapTy local;
    Index curIndex;
    Index scanStart;
    CTy* current;
    unsigned int lastMasterVersion;
    unsigned int numPops;

    ThreadData() :
      curIndex(std::numeric_limits<Index>::min()), 
      scanStart(std::numeric_limits<Index>::min()),
      current(0), lastMasterVersion(0), numPops(0) { }
  };

  typedef std::deque<std::pair<Index, CTy*> > MasterLog;

  // NB: Place dynamically growing masterLog after fixed-size PerThreadStorage
  // members to give higher likelihood of reclaiming PerThreadStorage
  Runtime::PerThreadStorage<ThreadData> data;
  Runtime::LL::PaddedLock<Concurrent> masterLock;
  MasterLog masterLog;

  std::atomic<unsigned int> masterVersion;
  Indexer indexer;

  bool updateLocal(ThreadData& p) {
    if (p.lastMasterVersion != masterVersion.load(std::memory_order_relaxed)) {
      //masterLock.lock();
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
      //masterLock.unlock();
      return true;
    }
    return false;
  }

  GALOIS_ATTRIBUTE_NOINLINE
  Galois::optional<T> slowPop(ThreadData& p) {
    //Failed, find minimum bin
    updateLocal(p);
    unsigned myID = Runtime::LL::getTID();
    bool localLeader = Runtime::LL::isPackageLeaderForSelf(myID);

    Index msS = std::numeric_limits<Index>::min();
    if (BSP) {
      msS = p.scanStart;
      if (localLeader) {
        for (unsigned i = 0; i < Runtime::activeThreads; ++i)
          msS = std::min(msS, data.getRemote(i)->scanStart);
      } else {
        msS = std::min(msS, data.getRemote(Runtime::LL::getLeaderForThread(myID))->scanStart);
      }
    }

    for (auto ii = p.local.lower_bound(msS), ee = p.local.end(); ii != ee; ++ii) {
      Galois::optional<T> item;
      if ((item = ii->second->pop())) {
        p.current = ii->second;
        p.curIndex = ii->first;
        p.scanStart = ii->first;
        return item;
      }
    }
    return Galois::optional<value_type>();
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
  OrderedByIntegerMetric(const Indexer& x = Indexer()): masterVersion(0), indexer(x) { }

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
    // Fast path
    if (index == p.curIndex && p.current) {
      p.current->push(val);
      return;
    }

    // Slow path
    CTy* C = updateLocalOrCreate(p, index);
    if (BSP && index < p.scanStart)
      p.scanStart = index;
    // Opportunistically move to higher priority work
    if (!UseBarrier && index < p.curIndex) {
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

  Galois::optional<value_type> pop() {
    // Find a successful pop
    ThreadData& p = *data.getLocal();
    CTy* C = p.current;

    if (this->hasStored(p, p.curIndex))
      return this->popStored(p, p.curIndex);

    if (!UseBarrier && BlockPeriod && (p.numPops++ & ((1<<BlockPeriod)-1)) == 0)
      return slowPop(p);

    Galois::optional<value_type> item;
    if (C && (item = C->pop()))
      return item;

    if (UseBarrier)
      return item;
    
    // Slow path
    return slowPop(p);
  }

  template<bool Barrier=UseBarrier>
  auto empty() -> typename std::enable_if<Barrier, bool>::type {
    Galois::optional<value_type> item;
    ThreadData& p = *data.getLocal();

    item = slowPop(p);
    if (item)
      p.stored.push_back(std::make_pair(p.curIndex, *item));
    p.hasWork = item;

    this->barrier.wait();

    bool hasWork = p.hasWork;
    Index curIndex = p.curIndex;
    CTy* C = p.current;

    for (unsigned i = 0; i < Runtime::activeThreads; ++i) {
      ThreadData& o = *data.getRemote(i);
      if (curIndex > o.curIndex) {
        curIndex = o.curIndex;
        C = o.current;
      }
      hasWork |= o.hasWork;
    }

    this->barrier.wait();
    
    p.current = C;
    p.curIndex = curIndex;

    return !hasWork;
  }
};
GALOIS_WLCOMPILECHECK(OrderedByIntegerMetric)

} // end namespace WorkList
} // end namespace Galois

#endif
