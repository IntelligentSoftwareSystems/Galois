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
#include "Galois/Runtime/WorkListHelpers.h"
#include "Galois/Runtime/ll/PaddedLock.h"
#include "Galois/Runtime/mm/Mem.h"
#include "Galois/util/GAlgs.h"

#include "Galois/gdeque.h"

#include <limits>
#include <iterator>
#include <map>
#include <vector>
#include <deque>
#include <algorithm>
#include <iterator>

#include <boost/utility.hpp>
#include <boost/optional.hpp>
#include <boost/ref.hpp>

namespace GaloisRuntime {
namespace WorkList {

// Worklists may not be copied.
// Worklists should be default instantiatable
// All classes (should) conform to:
template<typename T, bool concurrent>
class AbstractWorkList {
public:
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
  void push(const value_type& val);

  //! push a range onto the queue
  template<typename Iter>
  void push(Iter b, Iter e);

  //! push initial range onto the queue
  //! called with the same b and e on each thread
  template<typename Iter>
  void push_initial(Iter b, Iter e);

  //Optional, but this is the likely interface for stealing
  //! steal from a similar worklist
  boost::optional<value_type> steal(AbstractWorkList& victim, bool half, bool pop);


  //! pop a value from the queue.
  boost::optional<value_type> pop();
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template<typename Iter>
void fill_work(Iter& b, Iter& e) {
  unsigned int a = galoisActiveThreads;
  unsigned int id = LL::getTID();
  unsigned int dist = std::distance(b, e);
  unsigned int num = (dist + a - 1) / a; //round up
  unsigned int A = std::min(num * id, dist);
  unsigned int B = std::min(num * (id + 1), dist);
  e = b;
  std::advance(b, A);
  std::advance(e, B);
}

template<class WL, typename Iter>
void fill_work(WL& wl, Iter b, Iter e) {
  Iter b2 = b;
  Iter e2 = e;
  fill_work(b2, e2);
  wl.push(b2, e2);
}


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

  template<typename Iter>
  void push_initial(Iter b, Iter e) {
    if (LL::getTID() == 0)
      push(b,e);
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

  template<typename Iter>
  void push_initial(Iter b, Iter e) {
    if (LL::getTID() == 0)
      push(b,e);
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

  template<typename Iter>
  void push_initial(Iter b, Iter e) {
    if (LL::getTID() == 0)
      push(b,e);
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

template<class Indexer = DummyIndexer<int>, typename ContainerTy = FIFO<>, bool BSP=true, typename T = int, bool concurrent = true>
class OrderedByIntegerMetric : private boost::noncopyable {

  typedef typename ContainerTy::template rethread<concurrent>::WL CTy;

  struct perItem {
    CTy* current;
    unsigned int curVersion;
    unsigned int lastMasterVersion;
    unsigned int scanStart;
    std::map<unsigned int, CTy*> local;
    perItem() :current(NULL), curVersion(0), lastMasterVersion(0), scanStart(0) {}
  };

  std::deque<std::pair<unsigned int, CTy*> > masterLog;
  LL::PaddedLock<concurrent> masterLock;
  volatile unsigned int masterVersion;

  Indexer I;

  PerThreadStorage<perItem> current;

  void updateLocal_i(perItem& p) {
    for (; p.lastMasterVersion < masterVersion; ++p.lastMasterVersion) {
      std::pair<unsigned int, CTy*> logEntry = masterLog[p.lastMasterVersion];
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

  CTy* updateLocalOrCreate(perItem& p, unsigned int i) {
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
    typedef OrderedByIntegerMetric<Indexer,ContainerTy,BSP,T,newconcurrent> WL;
  };
  template<typename Tnew>
  struct retype {
    typedef OrderedByIntegerMetric<Indexer,typename ContainerTy::template retype<Tnew>::WL,BSP,Tnew,concurrent> WL;
  };

  typedef T value_type;

  OrderedByIntegerMetric(const Indexer& x = Indexer())
    :masterVersion(0), I(x)
  { }

  ~OrderedByIntegerMetric() {
    for (typename std::deque<std::pair<unsigned int, CTy*> >::iterator ii = masterLog.begin(), ee = masterLog.end(); ii != ee; ++ii) {
      delete ii->second;
    }
  }

  void push(const value_type& val) {
    unsigned int index = I(val);
    perItem& p = *current.getLocal();
    //fastpath
    if (index == p.curVersion && p.current) {
      p.current->push(val);
      return;
    }

    //slow path
    CTy* lC = updateLocalOrCreate(p, index);
    if (BSP && index < p.scanStart)
      p.scanStart = index;
    //opportunistically move to higher priority work
    if (index < p.curVersion) {
      p.curVersion = index;
      p.current = lC;
    }
    lC->push(val);
  }

  template<typename Iter>
  void push(Iter b, Iter e) {
    while (b != e)
      push(*b++);
  }

  template<typename Iter>
  void push_initial(Iter b, Iter e) {
    fill_work(*this, b, e);
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
    bool localLeader = LL::isLeaderForPackage(myID);

    unsigned msS = 0;
    if (BSP) {
      msS = p.scanStart;
      if (localLeader)
	for (unsigned i = 0; i <  galoisActiveThreads; ++i)
	  msS = std::min(msS, current.getRemote(i)->scanStart);
      else
	msS = std::min(msS, current.getRemote(LL::getLeaderForThread(myID))->scanStart);
    }

    for (typename std::map<unsigned int, CTy*>::iterator ii = p.local.lower_bound(msS), ee = p.local.end();
	 ii != ee; ++ii) {
      if ((retval = ii->second->pop())) {
	C = ii->second;
	p.curVersion = ii->first;
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

  template<typename Iter>
  void push_initial(Iter b, Iter e) {
    global.push_initial(b,e);
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

// template<typename TQ>
// struct squeues<true,TQ> {
//   PerLevel<TQ> queues;
//   TQ& get(int i) { return queues.get(i); }
//   TQ& get() { return queues.get(); }
//   int myEffectiveID() { return queues.myEffectiveID(); }
//   int size() { return queues.size(); }
// };

template<typename TQ>
struct squeues<true,TQ> {
  PerPackageStorage<TQ> queues;
  TQ& get(int i) { return *queues.getRemote(i); }
  TQ& get() { return *queues.getLocal(); }
  int myEffectiveID() { return LL::getTID(); } //queues.myEffectiveID(); }
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
  class Chunk : public FixedSizeRing<T, chunksize>, public QT<Chunk, concurrent>::ListNode {};

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
    return pushi(val,n);
  }

  template<typename Iter>
  void push(Iter b, Iter e) {
    p* n = data.getLocal();
    while (b != e)
      pushi(*b++, n);
  }

  template<typename Iter>
  void push_initial(Iter b, Iter e) {
    fill_work(*this, b, e);
  }

  boost::optional<value_type> pop()  {
    p& n = *data.getLocal();
    boost::optional<value_type> retval;
    if (isStack) {
      if (n.next && (retval = n.next->pop_back()))
	return retval;
      if (n.next)
	delChunk(n.next);
      n.next = popChunk();
      if (n.next)
	return n.next->pop_back();
      return boost::optional<value_type>();
    } else {
      if (n.cur && (retval = n.cur->pop_front()))
	return retval;
      if (n.cur)
	delChunk(n.cur);
      n.cur = popChunk();
      if (!n.cur) {
	n.cur = n.next;
	n.next = 0;
      }
      if (n.cur)
	return n.cur->pop_front();
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

//FIXME: make this valid for input iterators (e.g. no default constructor)
template<typename IterTy = int*>
class ForwardAccessRange {
  //! Thread-local data
  struct TLD {
    IterTy begin;
    IterTy end;
  };

  PerThreadStorage<TLD> tlds;
  
public:

  //! T is the value type of the WL
  typedef typename std::iterator_traits<IterTy>::value_type value_type;

  template<bool newconcurrent>
  struct rethread {
    typedef ForwardAccessRange<IterTy> WL;
  };

  template<typename Tnew>
  struct retype {
    typedef ForwardAccessRange<IterTy> WL;
  };

  //! push a value onto the queue
  void push(const value_type& val) {
    abort();
  }

  //! push a range onto the queue
  template<typename Iter>
  void push(Iter b, Iter e) {
    if (b == e)
      return;
    abort();
  }

  //stagger each thread's start item
  void push_initial(IterTy b, IterTy e) {
    TLD& tld = *tlds.getLocal();
    tld.begin = Galois::safe_advance(b, e, LL::getTID());
    tld.end = e;
  }

  //! pop a value from the queue.
  // move through range in num thread strides
  boost::optional<value_type> pop() {
    TLD& tld = *tlds.getLocal();
    if (tld.begin != tld.end) {
      boost::optional<value_type> retval = *tld.begin;
      tld.begin = Galois::safe_advance(tld.begin, tld.end, galoisActiveThreads);
      assert(retval);
      return retval;
    }
    return boost::optional<value_type>();
  }
};
GALOIS_WLCOMPILECHECK(ForwardAccessRange)

template<bool Stealing=false, typename IterTy = int*>
class RandomAccessRange {
public:
  //! T is the value type of the WL
  typedef typename std::iterator_traits<IterTy>::value_type value_type;

private:
  //! Thread-local data
  struct TLD {
    IterTy begin;
    IterTy end;
    IterTy stealBegin;
    IterTy stealEnd;
    LL::SimpleLock<true> stealLock;
  };

  PerThreadStorage<TLD> tlds;

  bool do_steal(TLD& stld, TLD& otld) {
    bool retval = false;
    otld.stealLock.lock();
    if (otld.stealBegin != otld.stealEnd) {
      stld.begin = otld.stealBegin;
      otld.stealBegin = stld.end = Galois::split_range(otld.stealBegin, otld.stealEnd);
      retval = true;
    }
    otld.stealLock.unlock();
    return retval;
  }

  void try_steal(TLD& tld) {
    //First try stealing from self
    if (do_steal(tld,tld))
      return;
    //Then try stealing from neighbor
    if (do_steal(tld, *tlds.getRemote((LL::getTID() + 1) % galoisActiveThreads))) {
      //redistribute stolen items for neighbor to steal too
      if (std::distance(tld.begin, tld.end) > 1) {
	tld.stealLock.lock();
	tld.stealEnd = tld.end;
	tld.stealBegin = tld.end = Galois::split_range(tld.begin, tld.end);
	tld.stealLock.unlock();
      }
    }
  }

  GALOIS_ATTRIBUTE_NOINLINE
  boost::optional<value_type> pop_slowpath(TLD& tld) {
    if (Stealing && tld.begin == tld.end)
      try_steal(tld);
    if (tld.begin != tld.end)
      return boost::optional<value_type>(*tld.begin++);
    return boost::optional<value_type>();
  }

public:
  template<bool newconcurrent>
  struct rethread {
    typedef RandomAccessRange<Stealing,IterTy> WL;
  };

  template<typename Tnew>
  struct retype {
    typedef RandomAccessRange<Stealing,IterTy> WL;
  };

  //! push a value onto the queue
  void push(const value_type& val) {
    abort();
  }

  //! push a range onto the queue
  template<typename Iter>
  void push(Iter b, Iter e) {
    if (b == e)
      return;
    abort();
  }

  //stagger each thread's start item
  void push_initial(IterTy b, IterTy e) {
    TLD& tld = *tlds.getLocal();
    tld.begin = b;
    tld.end = e;
    fill_work(tld.begin, tld.end);

    if (Stealing) {
      tld.stealEnd = tld.end;
      tld.stealBegin = tld.end = Galois::split_range(tld.begin, tld.end);
    }
  }

  //! pop a value from the queue.
  // move through range in num thread strides
  boost::optional<value_type> pop() {
    TLD& tld = *tlds.getLocal();
    if (tld.begin != tld.end)
      return boost::optional<value_type>(*tld.begin++);
    return pop_slowpath(tld);
  }
};
GALOIS_WLCOMPILECHECK(RandomAccessRange)

//FIXME: make this more typed
template<typename IterTy = int*>
class LocalAccessRange {
  //! Thread-local data
  struct TLD {
    IterTy begin;
    IterTy end;
  };

  PerThreadStorage<TLD> tlds;
  
public:

  //! T is the value type of the WL
  typedef typename std::iterator_traits<IterTy>::value_type value_type;

  template<bool newconcurrent>
  struct rethread {
    typedef LocalAccessRange<IterTy> WL;
  };

  template<typename Tnew>
  struct retype {
    typedef LocalAccessRange<IterTy> WL;
  };

  //! push a value onto the queue
  void push(const value_type& val) {
    abort();
  }

  //! push a range onto the queue
  template<typename Iter>
  void push(Iter b, Iter e) {
    if (b == e)
      return;
    abort();
  }

  //stagger each thread's start item
  template<typename LocalTy>
  void push_initial(LocalTy b, LocalTy e) {
    TLD& tld = *tlds.getLocal();
    tld.begin = b.resolve();
    tld.end = e.resolve();
  }

  //! pop a value from the queue.
  // move through range in num thread strides
  boost::optional<value_type> pop() {
    TLD& tld = *tlds.getLocal();
    if (tld.begin != tld.end) {
      boost::optional<value_type> retval = *tld.begin++;
      assert(*retval);
      return retval;
    }
    return boost::optional<value_type>();
  }
};
//GALOIS_WLCOMPILECHECK(LocalAccessRange);

//FIXME: make this more typed
template<typename IterTy, typename WLTy>
class LocalAccessDist {
  WLTy w;
  
public:

  //! T is the value type of the WL
  typedef typename std::iterator_traits<IterTy>::value_type value_type;

  template<bool newconcurrent>
  struct rethread {
    typedef LocalAccessDist<IterTy, typename WLTy::template rethread<newconcurrent>::WL > WL;
  };

  template<typename Tnew>
  struct retype {
    typedef LocalAccessDist<IterTy, typename WLTy::template retype<Tnew>::WL > WL;
  };

  //! push a value onto the queue
  void push(const value_type& val) {
    w.push();
  }

  //! push a range onto the queue
  template<typename Iter>
  void push(Iter b, Iter e) {
    w.push(b,e);
  }

  //stagger each thread's start item
  template<typename LocalTy>
  void push_initial(LocalTy b, LocalTy e) {
    w.push(b.resolve(), e.resolve());
  }

  //! pop a value from the queue.
  // move through range in num thread strides
  boost::optional<value_type> pop() {
    return w.pop();
  }
};
//GALOIS_WLCOMPILECHECK(LocalAccessDist);

template<typename OwnerFn=DummyIndexer<int>, typename T = int>
class OwnerComputesWL : private boost::noncopyable {
  typedef ChunkedLIFO<256, T> cWL;
  //  typedef ChunkedLIFO<256, T> pWL;
  typedef ChunkedLIFO<256, T> pWL;

  OwnerFn Fn;
  PerLevel<cWL> Items;
  PerLevel<pWL> pushBuffer;

public:
  template<bool newconcurrent>
  struct rethread {
    typedef OwnerComputesWL<OwnerFn, T> WL;
  };

  template<typename nTy>
  struct retype {
    typedef OwnerComputesWL<OwnerFn, nTy> WL;
  };

  typedef T value_type;

  void push(const value_type& val)  {
    unsigned int index = Fn(val);
    unsigned int mindex = Items.effectiveIDFor(index);
    //std::cerr << "[" << index << "," << index % active << "]\n";
    if (mindex == Items.myEffectiveID())
      Items.get(mindex).push(val);
    else
      pushBuffer.get(mindex).push(val);
  }

  template<typename ItTy>
  void push(ItTy b, ItTy e) {
    while (b != e)
      push(*b++);
  }

  template<typename ItTy>
  void push_initial(ItTy b, ItTy e) {
    fill_work(*this, b, e);
    for (unsigned int x = 0; x < pushBuffer.size(); ++x)
      pushBuffer.get(x).flush();
  }

  boost::optional<value_type> pop() {
    cWL& wl = Items.get();
    boost::optional<value_type> retval = wl.pop();
    if (retval)
      return retval;
    pWL& p = pushBuffer.get();
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
  GaloisRuntime::PerThreadStorage<TLD> tlds;
  GaloisRuntime::GBarrier barrier1;
  GaloisRuntime::GBarrier barrier2;
  GaloisRuntime::LL::CacheLineStorage<volatile long> some;
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

  template<typename ItTy>
  void push_initial(ItTy b, ItTy e) {
    fill_work(*this, b, e);
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
      if (GaloisRuntime::LL::getTID() == 0) {
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

#endif

