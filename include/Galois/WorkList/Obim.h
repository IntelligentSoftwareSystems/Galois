/** Scalable priority worklist -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2012, The University of Texas at Austin. All rights reserved.
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

//#include <map>
#include "Galois/FlatMap.h"

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
 * @tparam ContainerTy Scheduler for each bucket
 * @tparam CheckPeriod Check for higher priority work every 2^CheckPeriod
 *                     iterations
 * @tparam BSP Use back-scan prevention
 */
template<class Indexer = DummyIndexer<int>, typename ContainerTy = FIFO<>, unsigned CheckPeriod=0, bool BSP=true, typename T = int, typename IndexTy = int, bool concurrent = true>
class OrderedByIntegerMetric : private boost::noncopyable {
  typedef typename ContainerTy::template rethread<concurrent> CTy;

  typedef Galois::flat_map<IndexTy, CTy*> LMapTy;
  //typedef std::map<IndexTy, CTy*> LMapTy;

  struct perItem {
    LMapTy local;
    IndexTy curIndex;
    IndexTy scanStart;
    CTy* current;
    unsigned int lastMasterVersion;
    unsigned int numPops;

    perItem() :
      curIndex(std::numeric_limits<IndexTy>::min()), 
      scanStart(std::numeric_limits<IndexTy>::min()),
      current(0), lastMasterVersion(0), numPops(0) { }
  };

  typedef std::deque<std::pair<IndexTy, CTy*> > MasterLog;

  // NB: Place dynamically growing masterLog after fixed-size PerThreadStorage
  // members to give higher likelihood of reclaiming PerThreadStorage
  Runtime::PerThreadStorage<perItem> current;
  Runtime::LL::PaddedLock<concurrent> masterLock;
  MasterLog masterLog;

  volatile unsigned int masterVersion;
  Indexer I;

  void updateLocal_i(perItem& p) {
    for (; p.lastMasterVersion < masterVersion; ++p.lastMasterVersion)
      p.local.insert(masterLog[p.lastMasterVersion]);
  }

  bool updateLocal(perItem& p) {
    if (p.lastMasterVersion != masterVersion) {
      //masterLock.lock();
      updateLocal_i(p);
      //masterLock.unlock();
      return true;
    }
    return false;
  }

  GALOIS_ATTRIBUTE_NOINLINE
  bool better_bucket(perItem& p) {
    updateLocal(p);
    unsigned myID = Runtime::LL::getTID();
    bool localLeader = Runtime::LL::isPackageLeaderForSelf(myID);

    IndexTy msS = std::numeric_limits<IndexTy>::min();
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
  boost::optional<T> _slow_pop(perItem& p) {
    //Failed, find minimum bin
    updateLocal(p);
    unsigned myID = Runtime::LL::getTID();
    bool localLeader = Runtime::LL::isPackageLeaderForSelf(myID);

    IndexTy msS = std::numeric_limits<IndexTy>::min();
    if (BSP) {
      msS = p.scanStart;
      if (localLeader)
	for (unsigned i = 0; i < Runtime::activeThreads; ++i)
	  msS = std::min(msS, current.getRemote(i)->scanStart);
      else
	msS = std::min(msS, current.getRemote(Runtime::LL::getLeaderForThread(myID))->scanStart);
    }

    for (auto ii = p.local.lower_bound(msS), ee = p.local.end(); ii != ee; ++ii) {
      boost::optional<T> retval;
      if ((retval = ii->second->pop())) {
	p.current = ii->second;
	p.curIndex = ii->first;
	p.scanStart = ii->first;
	return retval;
      }
    }
    return boost::optional<value_type>();
  }

  GALOIS_ATTRIBUTE_NOINLINE
  CTy* _slow_updateLocalOrCreate(perItem& p, IndexTy i) {
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

  inline CTy* updateLocalOrCreate(perItem& p, IndexTy i) {
    //Try local then try update then find again or else create and update the master log
    CTy* lC;
    if ((lC = p.local[i]))
      return lC;
    //slowpath
    return _slow_updateLocalOrCreate(p, i);
  }

 public:
  template<bool newconcurrent>
    using rethread = OrderedByIntegerMetric<Indexer,ContainerTy,CheckPeriod,BSP,T,IndexTy,newconcurrent>;
  template<typename Tnew>
    using retype = OrderedByIntegerMetric<Indexer,typename ContainerTy::template retype<Tnew>,CheckPeriod,BSP,Tnew,decltype(Indexer()(Tnew())),concurrent>;

  typedef T value_type;

  OrderedByIntegerMetric(const Indexer& x = Indexer())
    :masterVersion(0), I(x)
  { }

  ~OrderedByIntegerMetric() {
    // Deallocate in LIFO order to give opportunity for simple garbage
    // collection
    for (typename MasterLog::reverse_iterator ii = masterLog.rbegin(), ee = masterLog.rend(); ii != ee; ++ii) {
      delete ii->second;
    }
  }

  void push(const value_type& val) {
    IndexTy index = I(val);
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

  boost::optional<value_type> pop() {
    //Find a successful pop
    perItem& p = *current.getLocal();
    CTy* C = p.current;
    if (CheckPeriod && (p.numPops++ & ((1<<CheckPeriod)-1)) == 0 && better_bucket(p))
      return _slow_pop(p);

    boost::optional<value_type> retval;
    if (C && (retval = C->pop()))
      return retval;

    //failed: slow path
    return _slow_pop(p);
  }
};
GALOIS_WLCOMPILECHECK(OrderedByIntegerMetric)

} // end namespace WorkList
} // end namespace Galois

#endif
