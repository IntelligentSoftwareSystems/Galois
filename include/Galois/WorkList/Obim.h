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
#include "Galois/WorkList/Fifo.h"
#include "Galois/WorkList/WorkListHelpers.h"

#include GALOIS_CXX11_STD_HEADER(type_traits)
#include <limits>

namespace Galois {
namespace WorkList {

/**
 * Approximate priority scheduling. Indexer is a default-constructable class
 * whose instances conform to <code>R r = indexer(item)</code> where R is
 * some type with a total order defined by <code>operator&lt;</code> and <code>operator==</code>
 * and item is an element from the Galois set iterator.
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
template<class Indexer = DummyIndexer<int>, typename Container = FIFO<>,
  unsigned BlockPeriod=0,
  bool BSP=true,
  typename T=int,
  typename Index=int,
  bool Concurrent=true>
struct OrderedByIntegerMetric : private boost::noncopyable {
  template<bool _concurrent>
  struct rethread { typedef OrderedByIntegerMetric<Indexer, typename Container::template rethread<_concurrent>::type, BlockPeriod, BSP, T, Index, _concurrent> type; };

  template<typename _T>
  struct retype { typedef OrderedByIntegerMetric<Indexer, typename Container::template retype<_T>::type, BlockPeriod, BSP, _T, typename std::result_of<Indexer(_T)>::type, Concurrent> type; };

  template<unsigned _period>
  struct with_block_period { typedef OrderedByIntegerMetric<Indexer, Container, _period, BSP, T, Index, Concurrent> type; };

  template<typename _container>
  struct with_container { typedef OrderedByIntegerMetric<Indexer, _container, BlockPeriod, BSP, T, Index, Concurrent> type; };

  template<typename _indexer>
  struct with_indexer { typedef OrderedByIntegerMetric<_indexer, Container, BlockPeriod, BSP, T, Index, Concurrent> type; };

  template<bool _bsp>
  struct with_back_scan_prevention { typedef OrderedByIntegerMetric<Indexer, Container, BlockPeriod, _bsp, T, Index, Concurrent> type; };

  typedef T value_type;

private:
  typedef typename Container::template rethread<Concurrent>::type CTy;
  typedef Galois::flat_map<Index, CTy*> LMapTy;
  //typedef std::map<Index, CTy*> LMapTy;

  struct perItem {
    LMapTy local;
    Index curIndex;
    Index scanStart;
    CTy* current;
    unsigned int lastMasterVersion;
    unsigned int numPops;

    perItem() :
      curIndex(std::numeric_limits<Index>::min()), 
      scanStart(std::numeric_limits<Index>::min()),
      current(0), lastMasterVersion(0), numPops(0) { }
  };

  typedef std::deque<std::pair<Index, CTy*> > MasterLog;

  // NB: Place dynamically growing masterLog after fixed-size PerThreadStorage
  // members to give higher likelihood of reclaiming PerThreadStorage
  Runtime::PerThreadStorage<perItem> current;
  Runtime::LL::PaddedLock<Concurrent> masterLock;
  MasterLog masterLog;

  volatile unsigned int masterVersion;
  Indexer indexer;

  bool updateLocal(perItem& p) {
    if (p.lastMasterVersion != masterVersion) {
      //masterLock.lock();
      for (; p.lastMasterVersion < masterVersion; ++p.lastMasterVersion)
        p.local.insert(masterLog[p.lastMasterVersion]);
      //masterLock.unlock();
      return true;
    }
    return false;
  }

  GALOIS_ATTRIBUTE_NOINLINE
  bool betterBucket(perItem& p) {
    updateLocal(p);
    unsigned myID = Runtime::LL::getTID();
    bool localLeader = Runtime::LL::isPackageLeaderForSelf(myID);

    Index msS = std::numeric_limits<Index>::min();
    if (BSP) {
      msS = p.scanStart;
      if (localLeader)
	for (unsigned i = 0; i <  Runtime::activeThreads; ++i)
	  msS = std::min(msS, current.getRemote(i)->scanStart);
      else
	msS = std::min(msS, current.getRemote(Runtime::LL::getLeaderForThread(myID))->scanStart);
    }

    return p.curIndex != msS;
  }

  GALOIS_ATTRIBUTE_NOINLINE
  Galois::optional<T> slowPop(perItem& p) {
    //Failed, find minimum bin
    updateLocal(p);
    unsigned myID = Runtime::LL::getTID();
    bool localLeader = Runtime::LL::isPackageLeaderForSelf(myID);

    Index msS = std::numeric_limits<Index>::min();
    if (BSP) {
      msS = p.scanStart;
      if (localLeader)
	for (unsigned i = 0; i < Runtime::activeThreads; ++i)
	  msS = std::min(msS, current.getRemote(i)->scanStart);
      else
	msS = std::min(msS, current.getRemote(Runtime::LL::getLeaderForThread(myID))->scanStart);
    }

    for (auto ii = p.local.lower_bound(msS), ee = p.local.end(); ii != ee; ++ii) {
      Galois::optional<T> retval;
      if ((retval = ii->second->pop())) {
	p.current = ii->second;
	p.curIndex = ii->first;
	p.scanStart = ii->first;
	return retval;
      }
    }
    return Galois::optional<value_type>();
  }

  GALOIS_ATTRIBUTE_NOINLINE
  CTy* slowUpdateLocalOrCreate(perItem& p, Index i) {
    //update local until we find it or we get the write lock
    do {
      CTy* lC;
      updateLocal(p);
      if ((lC = p.local[i]))
	return lC;
    } while (!masterLock.try_lock());
    //we have the write lock, update again then create
    updateLocal(p);
    CTy*& lC2 = p.local[i];
    if (!lC2) {
      lC2 = new CTy();
      p.lastMasterVersion = masterVersion + 1;
      masterLog.push_back(std::make_pair(i, lC2));
      __sync_fetch_and_add(&masterVersion, 1);
    }
    masterLock.unlock();
    return lC2;
  }

  inline CTy* updateLocalOrCreate(perItem& p, Index i) {
    //Try local then try update then find again or else create and update the master log
    CTy* lC;
    if ((lC = p.local[i]))
      return lC;
    //slowpath
    return slowUpdateLocalOrCreate(p, i);
  }

public:
  OrderedByIntegerMetric(const Indexer& x = Indexer()): masterVersion(0), indexer(x) { }

  ~OrderedByIntegerMetric() {
    // Deallocate in LIFO order to give opportunity for simple garbage
    // collection
    for (typename MasterLog::reverse_iterator ii = masterLog.rbegin(), ei = masterLog.rend(); ii != ei; ++ii) {
      delete ii->second;
    }
  }

  void push(const value_type& val) {
    Index index = indexer(val);
    perItem& p = *current.getLocal();
    //fastpath
    if (index == p.curIndex && p.current) {
      p.current->push(val);
      return;
    }

    //slow path
    CTy* lC = updateLocalOrCreate(p, index);
    if (BSP && index < p.scanStart)
      p.scanStart = index;
    //opportunistically move to higher priority work
    if (index < p.curIndex) {
      p.curIndex = index;
      p.current = lC;
    }
    lC->push(val);
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
    //Find a successful pop
    perItem& p = *current.getLocal();
    CTy* C = p.current;
    if (BlockPeriod && (p.numPops++ & ((1<<BlockPeriod)-1)) == 0 && betterBucket(p))
      return slowPop(p);

    Galois::optional<value_type> retval;
    if (C && (retval = C->pop()))
      return retval;

    //failed: slow path
    return slowPop(p);
  }
};
GALOIS_WLCOMPILECHECK(OrderedByIntegerMetric)

} // end namespace WorkList
} // end namespace Galois

#endif
