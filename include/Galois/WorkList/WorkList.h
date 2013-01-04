/** Scalable local worklists -*- C++ -*-
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
#ifndef GALOIS_RUNTIME_WORKLIST_H
#define GALOIS_RUNTIME_WORKLIST_H

#include "Galois/Runtime/PerThreadStorage.h"
#include "Galois/Runtime/Barrier.h"
#include "Galois/Runtime/ThreadPool.h"
#include "Galois/WorkList/WorkListHelpers.h"
#include "Galois/Runtime/ll/PaddedLock.h"
#include "Galois/Runtime/ll/gio.h"
#include "Galois/Runtime/mm/Mem.h"

#include "Galois/gdeque.h"
#include "Galois/FixedSizeRing.h"
#include "Galois/util/GAlgs.h"

#include <limits>
#include <iterator>
#include <map>
#include <vector>
#include <deque>
#include <algorithm>
#include <iterator>
#include <utility>

#include <boost/utility.hpp>
#include <boost/optional.hpp>
#include <boost/ref.hpp>

namespace Galois {
namespace Runtime {
namespace WorkList {

// Worklists may not be copied.
// Worklists should be default instantiatable
// All classes (should) conform to:
template<typename T, bool concurrent>
class AbstractWorkList {
  AbstractWorkList(const AbstractWorkList&);
  const AbstractWorkList& operator=(const AbstractWorkList&);

public:
  AbstractWorkList() { }

  //! T is the value type of the WL
  typedef T value_type;

  //! change the concurrency flag
  template<bool newconcurrent>
  struct rethread {
    typedef AbstractWorkList<T, newconcurrent> WL;
  };

  //! change the type the worklist holds
  template<typename Tnew>
  struct retype {
    typedef AbstractWorkList<Tnew, concurrent> WL;
  };

  //! push a value onto the queue
  void push(const value_type& val) { abort(); }

  //! push a range onto the queue
  template<typename Iter>
  void push(Iter b, Iter e) { abort(); }

  //! push initial range onto the queue
  //! called with the same b and e on each thread
  template<typename RangeTy>
  void push_initial(RangeTy) { abort(); }

  //Optional, but this is the likely interface for stealing
  //! steal from a similar worklist
  boost::optional<value_type> steal(AbstractWorkList& victim, bool half, bool pop);

  //! pop a value from the queue.
  boost::optional<value_type> pop() { abort(); }
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template<typename T = int, bool concurrent = true>
class LIFO : private boost::noncopyable, private LL::PaddedLock<concurrent> {
  std::deque<T> wl;

  using LL::PaddedLock<concurrent>::lock;
  using LL::PaddedLock<concurrent>::try_lock;
  using LL::PaddedLock<concurrent>::unlock;

public:
  template<bool newconcurrent>
  struct rethread {
    typedef LIFO<T, newconcurrent> WL;
  };
  template<typename Tnew>
  struct retype {
    typedef LIFO<Tnew, concurrent> WL;
  };

  typedef T value_type;

  void push(const value_type& val) {
    lock();
    wl.push_back(val);
    unlock();
  }

  template<typename Iter>
  void push(Iter b, Iter e) {
    lock();
    wl.insert(wl.end(),b,e);
    unlock();
  }

  template<typename RangeTy>
  void push_initial(RangeTy range) {
    if (LL::getTID() == 0)
      push(range.begin(), range.end());
  }

  boost::optional<value_type> steal(LIFO& victim, bool half, bool pop) {
    boost::optional<value_type> retval;
    //guard against self stealing
    if (&victim == this) return retval;
    //Ordered lock to preent deadlock
    if (!LL::TryLockPairOrdered(*this, victim)) return retval;
    if (half) {
      typename std::deque<T>::iterator split = Galois::split_range(victim.wl.begin(), victim.wl.end());
      wl.insert(wl.end(), victim.wl.begin(), split);
      victim.wl.erase(victim.wl.begin(), split);
    } else {
      if (wl.empty()) {
	wl.swap(victim.wl);
      } else {
	wl.insert(wl.end(), victim.wl.begin(), victim.wl.end());
	victim.wl.clear();
      }
    }
    if (pop && !wl.empty()) {
      retval = wl.back();
      wl.pop_back();
    }
    UnLockPairOrdered(*this, victim);
    return retval;
  }

  boost::optional<value_type> pop()  {
    boost::optional<value_type> retval;
    lock();
    if (!wl.empty()) {
      retval = wl.back();
      wl.pop_back();
    }
    unlock();
    return retval;
  }
};
GALOIS_WLCOMPILECHECK(LIFO)

template<typename T = int, bool concurrent = true>
class FIFO : private boost::noncopyable, private LL::PaddedLock<concurrent>  {
  std::deque<T> wl;

  using LL::PaddedLock<concurrent>::lock;
  using LL::PaddedLock<concurrent>::try_lock;
  using LL::PaddedLock<concurrent>::unlock;

public:
  template<bool newconcurrent>
  struct rethread {
    typedef FIFO<T, newconcurrent> WL;
  };
  template<typename Tnew>
  struct retype {
    typedef FIFO<Tnew, concurrent> WL;
  };

  typedef T value_type;

  void push(const value_type& val) {
    lock();
    wl.push_back(val);
    unlock();
  }

  template<typename Iter>
  void push(Iter b, Iter e) {
    lock();
    wl.insert(wl.end(),b,e);
    unlock();
  }

  template<typename RangeTy>
  void push_initial(RangeTy range) {
    if (LL::getTID() == 0)
      push(range.begin(), range.end());
  }

  void steal(FIFO& victim) {
    if (!LL::TryLockPairOrdered(*this, victim))
      return;
    typename std::deque<T>::iterator split = Galois::split_range(victim.wl.begin(), wl.victim.end());
    wl.insert(wl.end(), victim.wl.begin(), split);
    victim.wl.erase(victim.wl.begin(), split);
    UnLockPairOrdered(*this, victim);
  }

  boost::optional<value_type> pop() {
    boost::optional<value_type> retval;
    lock();
    if (!wl.empty()) {
      retval = wl.front();
      wl.pop_front();
    }
    unlock();
    return retval;
  }
};
GALOIS_WLCOMPILECHECK(FIFO)

template<typename T = int, bool concurrent = true>
class GFIFO : private boost::noncopyable, private LL::PaddedLock<concurrent>  {
  Galois::gdeque<T> wl;

  using LL::PaddedLock<concurrent>::lock;
  using LL::PaddedLock<concurrent>::try_lock;
  using LL::PaddedLock<concurrent>::unlock;

public:
  template<bool newconcurrent>
  struct rethread {
    typedef GFIFO<T, newconcurrent> WL;
  };
  template<typename Tnew>
  struct retype {
    typedef GFIFO<Tnew, concurrent> WL;
  };

  typedef T value_type;

  void push(const value_type& val) {
    lock();
    wl.push_back(val);
    unlock();
  }

  template<typename Iter>
  void push(Iter b, Iter e) {
    lock();
    while (b != e)
      wl.push_back(*b++);
    unlock();
  }

  template<typename RangeTy>
  void push_initial(RangeTy range) {
    if (LL::getTID() == 0)
      push(range.begin(), range.end());
  }

  boost::optional<value_type> pop() {
    boost::optional<value_type> retval;
    lock();
    if (!wl.empty()) {
      retval = wl.front();
      wl.pop_front();
    }
    unlock();
    return retval;
  }
};
GALOIS_WLCOMPILECHECK(GFIFO)

//! Delay evaluation of result_of until we have all our types rather than
//! dummy values
template<bool,class>
struct safe_result_of { };

template<class F, class... ArgTypes>
struct safe_result_of<true, F(ArgTypes...)> {
  typedef typename std::result_of<F(ArgTypes...)>::type type;
};

template<class F, class... ArgTypes>
struct safe_result_of<false, F(ArgTypes...)> {
  typedef int type;
};

template<class Indexer = DummyIndexer<int>, typename ContainerTy = FIFO<>, bool BSP=true, typename T = int, bool concurrent = true, bool retyped=false>
class OrderedByIntegerMetric : private boost::noncopyable {
  typedef typename safe_result_of<retyped,Indexer(T)>::type IndexerValue;
  typedef typename ContainerTy::template rethread<concurrent>::WL CTy;

  struct perItem {
    std::map<IndexerValue, CTy*> local;
    IndexerValue curIndex;
    IndexerValue scanStart;
    CTy* current;
    unsigned int lastMasterVersion;

    perItem() :
      curIndex(std::numeric_limits<IndexerValue>::min()), 
      scanStart(std::numeric_limits<IndexerValue>::min()),
      current(0), lastMasterVersion(0) { }
  };

  std::deque<std::pair<IndexerValue, CTy*> > masterLog;
  LL::PaddedLock<concurrent> masterLock;
  volatile unsigned int masterVersion;

  Indexer I;

  PerThreadStorage<perItem> current;

  void updateLocal_i(perItem& p) {
    for (; p.lastMasterVersion < masterVersion; ++p.lastMasterVersion) {
      std::pair<IndexerValue, CTy*> logEntry = masterLog[p.lastMasterVersion];
      p.local[logEntry.first] = logEntry.second;
      assert(logEntry.second);
    }
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

  CTy* updateLocalOrCreate(perItem& p, IndexerValue i) {
    //Try local then try update then find again or else create and update the master log
    CTy*& lC = p.local[i];
    if (lC)
      return lC;
    //update local until we find it or we get the write lock
    do {
      updateLocal(p);
      if (lC)
	return lC;
    } while (!masterLock.try_lock());
    //we have the write lock, update again then create
    updateLocal(p);
    if (!lC) {
      lC = new CTy();
      p.lastMasterVersion = masterVersion + 1;
      masterLog.push_back(std::make_pair(i, lC));
      __sync_fetch_and_add(&masterVersion, 1);
      p.local[i] = lC;
    }
    masterLock.unlock();
    return lC;
  }

 public:
  template<bool newconcurrent>
  struct rethread {
    typedef OrderedByIntegerMetric<Indexer,ContainerTy,BSP,T,newconcurrent,retyped> WL;
  };
  template<typename Tnew>
  struct retype {
    typedef OrderedByIntegerMetric<Indexer,typename ContainerTy::template retype<Tnew>::WL,BSP,Tnew,concurrent,true> WL;
  };

  typedef T value_type;

  OrderedByIntegerMetric(const Indexer& x = Indexer())
    :masterVersion(0), I(x)
  { }

  ~OrderedByIntegerMetric() {
    for (typename std::deque<std::pair<IndexerValue, CTy*> >::iterator ii = masterLog.begin(), ee = masterLog.end(); ii != ee; ++ii) {
      delete ii->second;
    }
  }

  void push(const value_type& val) {
    IndexerValue index = I(val);
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
  void push_initial(RangeTy range) {
    push(range.local_begin(), range.local_end());
  }

  boost::optional<value_type> pop() {
    //Find a successful pop
    perItem& p = *current.getLocal();
    CTy*& C = p.current;
    boost::optional<value_type> retval;
    if (C && (retval = C->pop()))
      return retval;
    //Failed, find minimum bin
    updateLocal(p);
    unsigned myID = LL::getTID();
    bool localLeader = LL::isPackageLeaderForSelf(myID);

    IndexerValue msS = std::numeric_limits<IndexerValue>::min();
    if (BSP) {
      msS = p.scanStart;
      if (localLeader)
	for (unsigned i = 0; i <  galoisActiveThreads; ++i)
	  msS = std::min(msS, current.getRemote(i)->scanStart);
      else
	msS = std::min(msS, current.getRemote(LL::getLeaderForThread(myID))->scanStart);
    }

    for (typename std::map<IndexerValue, CTy*>::iterator ii = p.local.lower_bound(msS),
        ee = p.local.end(); ii != ee; ++ii) {
      if ((retval = ii->second->pop())) {
	C = ii->second;
	p.curIndex = ii->first;
	p.scanStart = ii->first;
	return retval;
      }
    }
    return boost::optional<value_type>();
  }
};
GALOIS_WLCOMPILECHECK(OrderedByIntegerMetric)

template<typename GlobalQueueTy = FIFO<>, typename LocalQueueTy = FIFO<>, typename T = int >
class LocalQueues : private boost::noncopyable {
  typedef typename LocalQueueTy::template rethread<false>::WL lWLTy;
  PerThreadStorage<lWLTy> local;
  GlobalQueueTy global;

public:
  template<bool newconcurrent>
  struct rethread {
    typedef LocalQueues<GlobalQueueTy, LocalQueueTy, T> WL;
  };
  template<typename Tnew>
  struct retype {
    typedef LocalQueues<typename GlobalQueueTy::template retype<Tnew>::WL, typename LocalQueueTy::template retype<Tnew>::WL, Tnew> WL;
  };

  typedef T value_type;

  void push(const value_type& val) {
    local.getLocal()->push(val);
  }

  template<typename Iter>
  void push(Iter b, Iter e) {
    local.getLocal()->push(b,e);
  }

  template<typename RangeTy>
  void push_initial(RangeTy range) {
    global.push_initial(range);
  }

  boost::optional<value_type> pop() {
    boost::optional<value_type> ret = local.getLocal()->pop();
    if (ret)
      return ret;
    return global.pop();
  }
};
GALOIS_WLCOMPILECHECK(LocalQueues)

//This overly complex specialization avoids a pointer indirection for non-distributed WL when accessing PerLevel
template<bool d, typename TQ>
struct squeues;

template<typename TQ>
struct squeues<true,TQ> {
  PerPackageStorage<TQ> queues;
  TQ& get(int i) { return *queues.getRemote(i); }
  TQ& get() { return *queues.getLocal(); }
  int myEffectiveID() { return LL::getTID(); }
  int size() { return galoisActiveThreads; }
};

template<typename TQ>
struct squeues<false,TQ> {
  TQ queue;
  TQ& get(int i) { return queue; }
  TQ& get() { return queue; }
  int myEffectiveID() { return 0; }
  int size() { return 0; }
};

template<typename T, template<typename, bool> class QT, bool distributed = false, bool isStack = false, int chunksize=64, bool concurrent=true>
class ChunkedMaster : private boost::noncopyable {
  class Chunk : public Galois::FixedSizeRing<T, chunksize>, public QT<Chunk, concurrent>::ListNode {};

  MM::FixedSizeAllocator heap;

  struct p {
    Chunk* cur;
    Chunk* next;
  };

  typedef QT<Chunk, concurrent> LevelItem;

  PerThreadStorage<p> data;
  squeues<distributed, LevelItem> Q;

  Chunk* mkChunk() {
    return new (heap.allocate(sizeof(Chunk))) Chunk();
  }
  
  void delChunk(Chunk* C) {
    C->~Chunk();
    heap.deallocate(C);
  }

  void pushChunk(Chunk* C)  {
    LevelItem& I = Q.get();
    I.push(C);
  }

  Chunk* popChunkByID(unsigned int i)  {
    LevelItem& I = Q.get(i);
    return I.pop();
  }

  Chunk* popChunk()  {
    int id = Q.myEffectiveID();
    Chunk* r = popChunkByID(id);
    if (r)
      return r;

    for (int i = id + 1; i < (int) Q.size(); ++i) {
      r = popChunkByID(i);
      if (r) 
	return r;
    }

    for (int i = 0; i < id; ++i) {
      r = popChunkByID(i);
      if (r)
	return r;
    }

    return 0;
  }

  T* pushi(const T& val, p* n)  {
    T* retval = 0;

    if (n->next && (retval = n->next->push_back(val)))
      return retval;
    if (n->next)
      pushChunk(n->next);
    n->next = mkChunk();
    retval = n->next->push_back(val);
    assert(retval);
    return retval;
  }

public:
  typedef T value_type;

  template<bool newconcurrent>
  struct rethread {
    typedef ChunkedMaster<T, QT, distributed, isStack, chunksize, newconcurrent> WL;
  };
  template<typename Tnew>
  struct retype {
    typedef ChunkedMaster<Tnew, QT, distributed, isStack, chunksize, concurrent> WL;
  };

  ChunkedMaster() : heap(sizeof(Chunk)) { }

  void flush() {
    p& n = *data.getLocal();
    if (n.next)
      pushChunk(n.next);
    n.next = 0;
  }
  
  //! Most worklists have void return value for push. This push returns address
  //! of placed item to facilitate some internal runtime uses. The address is
  //! generally not safe to use in the presence of concurrent pops.
  value_type* push(const value_type& val)  {
    p* n = data.getLocal();
    return pushi(val, n);
  }

  template<typename Iter>
  void push(Iter b, Iter e) {
    p* n = data.getLocal();
    while (b != e)
      pushi(*b++, n);
  }

  template<typename RangeTy>
  void push_initial(RangeTy range) {
    push(range.local_begin(), range.local_end());
  }

  boost::optional<value_type> pop()  {
    p& n = *data.getLocal();
    boost::optional<value_type> retval;
    if (isStack) {
      if (n.next && (retval = n.next->extract_back()))
	return retval;
      if (n.next)
	delChunk(n.next);
      n.next = popChunk();
      if (n.next)
	return n.next->extract_back();
      return boost::optional<value_type>();
    } else {
      if (n.cur && (retval = n.cur->extract_front()))
	return retval;
      if (n.cur)
	delChunk(n.cur);
      n.cur = popChunk();
      if (!n.cur) {
	n.cur = n.next;
	n.next = 0;
      }
      if (n.cur)
	return n.cur->extract_front();
      return boost::optional<value_type>();
    }
  }
};

template<int chunksize=64, typename T = int, bool concurrent=true>
class ChunkedFIFO : public ChunkedMaster<T, ConExtLinkedQueue, false, false, chunksize, concurrent> {};
GALOIS_WLCOMPILECHECK(ChunkedFIFO)

template<int chunksize=64, typename T = int, bool concurrent=true>
class ChunkedLIFO : public ChunkedMaster<T, ConExtLinkedStack, false, true, chunksize, concurrent> {};
GALOIS_WLCOMPILECHECK(ChunkedLIFO)

template<int chunksize=64, typename T = int, bool concurrent=true>
class dChunkedFIFO : public ChunkedMaster<T, ConExtLinkedQueue, true, false, chunksize, concurrent> {};
GALOIS_WLCOMPILECHECK(dChunkedFIFO)

template<int chunksize=64, typename T = int, bool concurrent=true>
class dChunkedLIFO : public ChunkedMaster<T, ConExtLinkedStack, true, true, chunksize, concurrent> {};
GALOIS_WLCOMPILECHECK(dChunkedLIFO)

template<typename OwnerFn=DummyIndexer<int>, typename WLTy=ChunkedLIFO<256>, typename T = int>
class OwnerComputesWL : private boost::noncopyable {
  typedef typename WLTy::template retype<T>::WL lWLTy;

  typedef lWLTy cWL;
  typedef lWLTy pWL;

  OwnerFn Fn;
  PerPackageStorage<cWL> items;
  PerPackageStorage<pWL> pushBuffer;

public:
  template<bool newconcurrent>
  struct rethread {
    typedef OwnerComputesWL<OwnerFn,typename WLTy::template rethread<newconcurrent>::WL, T> WL;
  };

  template<typename Tnew>
  struct retype {
    typedef OwnerComputesWL<OwnerFn,typename WLTy::template retype<Tnew>::WL,Tnew> WL;
  };

  typedef T value_type;

  void push(const value_type& val)  {
    unsigned int index = Fn(val);
    unsigned int tid = LL::getTID();
    unsigned int mindex = LL::getPackageForThread(index);
    //std::cerr << "[" << index << "," << index % active << "]\n";
    if (mindex == LL::getPackageForSelf(tid))
      items.getLocal()->push(val);
    else
      pushBuffer.getRemote(mindex)->push(val);
  }

  template<typename ItTy>
  void push(ItTy b, ItTy e) {
    while (b != e)
      push(*b++);
  }

  template<typename RangeTy>
  void push_initial(RangeTy range) {
    push(range.local_begin(), range.local_end());
    for (unsigned int x = 0; x < pushBuffer.size(); ++x)
      pushBuffer.getRemote(x)->flush();
  }

  boost::optional<value_type> pop() {
    cWL& wl = *items.getLocal();
    boost::optional<value_type> retval = wl.pop();
    if (retval)
      return retval;
    pWL& p = *pushBuffer.getLocal();
    while ((retval = p.pop()))
      wl.push(*retval);
    return wl.pop();
  }
};
GALOIS_WLCOMPILECHECK(OwnerComputesWL)

template<class ContainerTy=dChunkedFIFO<>, class T=int, bool concurrent = true>
class BulkSynchronous : private boost::noncopyable {

  typedef typename ContainerTy::template rethread<concurrent>::WL CTy;

  struct TLD {
    unsigned round;
    TLD(): round(0) { }
  };

  CTy wls[2];
  Galois::Runtime::PerThreadStorage<TLD> tlds;
  Galois::Runtime::GBarrier barrier1;
  Galois::Runtime::GBarrier barrier2;
  Galois::Runtime::LL::CacheLineStorage<volatile long> some;
  volatile bool empty;

 public:
  typedef T value_type;

  template<bool newconcurrent>
  struct rethread {
    typedef BulkSynchronous<ContainerTy,T,newconcurrent> WL;
  };
  template<typename Tnew>
  struct retype {
    typedef BulkSynchronous<typename ContainerTy::template retype<Tnew>::WL,Tnew,concurrent> WL;
  };

  BulkSynchronous(): empty(false) {
    unsigned num = galoisActiveThreads;
    barrier1.reinit(num);
    barrier2.reinit(num);
  }

  void push(const value_type& val) {
    wls[(tlds.getLocal()->round + 1) & 1].push(val);
  }

  template<typename ItTy>
  void push(ItTy b, ItTy e) {
    while (b != e)
      push(*b++);
  }

  template<typename RangeTy>
  void push_initial(RangeTy range) {
    push(range.local_begin(), range.local_end());
    tlds.getLocal()->round = 1;
    some.data = true;
  }

  boost::optional<value_type> pop() {
    TLD& tld = *tlds.getLocal();
    boost::optional<value_type> r;
    
    while (true) {
      if (empty)
        return r; // empty

      r = wls[tld.round].pop();
      if (r)
        return r;

      barrier1.wait();
      if (Galois::Runtime::LL::getTID() == 0) {
        if (!some.data)
          empty = true;
        some.data = false; 
      }
      tld.round = (tld.round + 1) & 1;
      barrier2.wait();

      r = wls[tld.round].pop();
      if (r) {
        some.data = true;
        return r;
      }
    }
  }
};
GALOIS_WLCOMPILECHECK(BulkSynchronous)

//End namespace

}
}
} // end namespace Galois

#endif

