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

#include "Galois/Runtime/mem.h"
#include "Galois/Runtime/WorkListHelpers.h"
#include "Galois/Runtime/ll/PaddedLock.h"
#include "Galois/Runtime/PerCPU.h"
#include "Galois/Runtime/Threads.h"
#include "Galois/util/GAlgs.h"

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
  void push(value_type val);

  //! push a range onto the queue
  template<typename Iter>
  void push(Iter b, Iter e);

  //! push range onto the queue
  template<typename Iter>
  void push_initial(Iter b, Iter e);

  //! pop a value from the queue.
  boost::optional<value_type> pop();
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

  void push(value_type val) {
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
    push(b,e);
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
WLCOMPILECHECK(LIFO);

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

  void push(value_type val) {
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
WLCOMPILECHECK(FIFO);

//#define ASDF 1
template<class Indexer = DummyIndexer<int>, typename ContainerTy = FIFO<>, typename T = int, bool concurrent = true >
class OrderedByIntegerMetric : private boost::noncopyable {

  typedef typename ContainerTy::template rethread<concurrent>::WL CTy;

  struct perItem {
    CTy* current;
    unsigned int curVersion;
    unsigned int lastMasterVersion;
    std::map<unsigned int, CTy*> local;
#if ASDF
    std::map<unsigned int, bool> cache;
#endif
  };

  std::vector<std::pair<unsigned int, CTy*> > masterLog;
  LL::PaddedLock<concurrent> masterLock;
  unsigned int masterVersion;

  Indexer I;

  PerCPU<perItem> current;

  void updateLocal_i(perItem& p) {
    //ASSERT masterLock
    for (; p.lastMasterVersion < masterVersion; ++p.lastMasterVersion) {
      std::pair<unsigned int, CTy*> logEntry = masterLog[p.lastMasterVersion];
      p.local[logEntry.first] = logEntry.second;
      assert(logEntry.second);
#if ASDF
      p.cache[logEntry.first] = true;
#endif
    }
  }

  void updateLocal(perItem& p) {
    if (p.lastMasterVersion != masterVersion) {
      masterLock.lock();
      updateLocal_i(p);
      masterLock.unlock();
    }
  }

  CTy* updateLocalOrCreate(perItem& p, unsigned int i) {
    //Try local then try update then find again or else create and update the master log
    CTy*& lC = p.local[i];
    if (lC)
      return lC;
    masterLock.lock();
    updateLocal_i(p);
    if (!lC) {
      lC = new CTy();
      ++masterVersion;
      masterLog.push_back(std::make_pair(i, lC));
    }
    masterLock.unlock();
    return lC;
  }

 public:
  template<bool newconcurrent>
  struct rethread {
    typedef OrderedByIntegerMetric<Indexer,ContainerTy,T,newconcurrent> WL;
  };
  template<typename Tnew>
  struct retype {
    typedef OrderedByIntegerMetric<Indexer,typename ContainerTy::template retype<Tnew>::WL,Tnew,concurrent> WL;
  };

  typedef T value_type;

  OrderedByIntegerMetric(const Indexer& x = Indexer())
    :masterVersion(0), I(x)
  {
    for (unsigned int i = 0; i < current.size(); ++i) {
      current.get(i).current = NULL;
      current.get(i).lastMasterVersion = 0;
    }
  }

  ~OrderedByIntegerMetric() {
    for (typename std::vector<std::pair<unsigned int, CTy*> >::iterator ii = masterLog.begin(), ee = masterLog.end(); ii != ee; ++ii) {
      delete ii->second;
    }
  }

  void push(value_type val) {
    unsigned int index = I(val);
    perItem& p = current.get();
    //fastpath
    if (index == p.curVersion && p.current) {
      p.current->push(val);
      return;
    }
    //slow path
    CTy* lC = updateLocalOrCreate(p, index);
    //opportunistically move to higher priority work
#if ASDF
    if (index < p.curVersion) {
      p.curVersion = index;
      p.current = lC;
    }
    p.cache[index] = true;
#endif
    lC->push(val);
  }

  template<typename Iter>
  void push(Iter b, Iter e) {
    while (b != e)
      push(*b++);
  }

  template<typename Iter>
  void push_initial(Iter b, Iter e) {
    push(b, e);
  }

  boost::optional<value_type> pop() {
    //Find a successful pop
    perItem& p = current.get();
    CTy*& C = p.current;
    boost::optional<value_type> retval;
    if (C && (retval = C->pop()))
      return retval;
    //Failed, find minimum bin
#if ASDF
    {
      //ltbb-style
      typename std::map<unsigned int, bool>::iterator ii = p.cache.begin(), ee = p.cache.end(), old;
      while (ii != ee) {
        p.curVersion = ii->first;
        C = p.local[ii->first];
        // why can C be null?
        if (C && (retval = C->pop()).first) {
          return retval;
        }
        old = ii;
        ++ii;
        p.cache.erase(old);
      }
    }
#endif

    updateLocal(p);
    for (typename std::map<unsigned int, CTy*>::iterator ii = p.local.begin(),
        ee = p.local.end(); ii != ee; ++ii) {
      p.curVersion = ii->first;
      C = ii->second;
      if ((retval = C->pop())) {
#if ASDF
        p.cache[ii->first] = true;
#endif
	return retval;
      } else {
#if ASDF
        p.cache.erase(ii->first);
#endif
      }
    }
    return boost::optional<value_type>();
  }
};
WLCOMPILECHECK(OrderedByIntegerMetric);

template<typename GlobalQueueTy = FIFO<>, typename LocalQueueTy = FIFO<>, typename T = int >
class LocalQueues : private boost::noncopyable {

  PerCPU<typename LocalQueueTy::template rethread<false>::WL> local;
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

  LocalQueues() {}

  void push(value_type val) {
    local.get().push(val);
  }

  template<typename Iter>
  void push(Iter b, Iter e) {
    local.get().push(b,e);
  }

  template<typename Iter>
  void push_initial(Iter b, Iter e) {
    global.push_initial(b,e);
  }

  boost::optional<value_type> pop() {
    boost::optional<value_type> ret = local.get().pop();
    if (ret)
      return ret;
    return global.pop();
  }
};
WLCOMPILECHECK(LocalQueues);

template<typename ContainerTy = FIFO<>, typename T = int >
class LocalStealing : private boost::noncopyable {

  PerCPU<typename ContainerTy::template rethread<true>::WL> local;

 public:
  template<bool newconcurrent>
  struct rethread {
    typedef LocalStealing<ContainerTy, T> WL;
  };
  template<typename Tnew>
  struct retype {
    typedef LocalStealing<typename ContainerTy::template retype<Tnew>::WL, Tnew> WL;
  };

  typedef T value_type;
  
  LocalStealing() {}

  void push(value_type val) {
    local.get().push(val);
  }

  template<typename Iter>
  void push(Iter b, Iter e) {
    local.get().push(b,e);
  }

  template<typename Iter>
  void push_initial(Iter b, Iter e) {
    local.get().push_initial(b,e);
  }

  boost::optional<value_type> pop() {
    boost::optional<value_type> ret = local.get().pop();
    if (ret)
      return ret;
    return local.getNext(ThreadPool::getActiveThreads()).pop();
  }
};
WLCOMPILECHECK(LocalStealing);

template<typename ContainerTy = FIFO<>, typename T = int >
class LevelStealing : private boost::noncopyable {

  PerLevel<typename ContainerTy::template rethread<true>::WL> local;

 public:
  template<bool newconcurrent>
  struct rethread {
    typedef LevelStealing<ContainerTy, T> WL;
  };
  template<typename Tnew>
  struct retype {
    typedef LevelStealing<typename ContainerTy::template retype<Tnew>::WL, Tnew> WL;
  };

  typedef T value_type;
  
  LevelStealing() {}

  void push(value_type val) {
    local.get().push(val);
  }

  template<typename Iter>
  void push(Iter b, Iter e) {
    local.get().push(b,e);
  }

  template<typename Iter>
  void push_initial(Iter b, Iter e) {
    local.get().push_initial(b,e);
  }

  boost::optional<value_type> pop() {
    boost::optional<value_type> ret = local.get().pop();
    if (ret)
      return ret;

    int mp = LL::getMaxPackageForThread(ThreadPool::getActiveThreads() - 1);
    int id = local.myEffectiveID();
    for (unsigned i = 0; i < local.size(); ++i) {
      ++id;
      id %= local.size();
      if (id <= mp) {
	ret = local.get(id).pop();
	if (ret)
	  return ret;
      }
    }
    return ret;
  }
};
WLCOMPILECHECK(LevelStealing);

//This overly complex specialization avoids a pointer indirection for non-distributed WL when accessing PerLevel
template<bool d, typename TQ>
struct squeues;

template<typename TQ>
struct squeues<true,TQ> {
  PerLevel<TQ> queues;
  TQ& get(int i) { return queues.get(i); }
  TQ& get() { return queues.get(); }
  int myEffectiveID() { return queues.myEffectiveID(); }
  int size() { return queues.size(); }
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
  class Chunk : public FixedSizeRing<T, chunksize, false>, public QT<Chunk, concurrent>::ListNode {};

  MM::FixedSizeAllocator heap;

  struct p {
    Chunk* cur;
    Chunk* next;
  };

  typedef QT<Chunk, concurrent> LevelItem;

  PerCPU<p> data;
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
    
    // int mp = LL::getMaxPackageForThread(ThreadPool::getActiveThreads() - 1);
    // for (int i = 0; i < Q.size(); ++i) {
    //   ++id;
    //   id %= Q.size();
    //   if (id <= mp) {
    // 	r = popChunkByID(id);
    // 	if (r)
    // 	  return r;
    //   }
    // }

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

  ChunkedMaster() : heap(sizeof(Chunk)) {
    for (unsigned int i = 0; i < data.size(); ++i) {
      p& r = data.get(i);
      r.cur = 0;
      r.next = 0;
    }
  }

  void push(value_type val)  {
    p& n = data.get();
    if (n.next && n.next->push_back(val))
      return;
    if (n.next)
      pushChunk(n.next);
    n.next = mkChunk();
    bool worked = n.next->push_back(val);
    assert(worked);
  }

  template<typename Iter>
  void push(Iter b, Iter e) {
    while (b != e)
      push(*b++);
  }

  template<typename Iter>
  void push_initial(Iter b, Iter e) {
    push(b,e);
  }

  boost::optional<value_type> pop()  {
    p& n = data.get();
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
WLCOMPILECHECK(ChunkedFIFO);

template<int chunksize=64, typename T = int, bool concurrent=true>
class ChunkedLIFO : public ChunkedMaster<T, ConExtLinkedStack, false, true, chunksize, concurrent> {};
WLCOMPILECHECK(ChunkedLIFO);

template<int chunksize=64, typename T = int, bool concurrent=true>
class dChunkedFIFO : public ChunkedMaster<T, ConExtLinkedQueue, true, false, chunksize, concurrent> {};
WLCOMPILECHECK(dChunkedFIFO);

template<int chunksize=64, typename T = int, bool concurrent=true>
class dChunkedLIFO : public ChunkedMaster<T, ConExtLinkedStack, true, true, chunksize, concurrent> {};
WLCOMPILECHECK(dChunkedLIFO);

//Weird WorkList where push and pop are different types
template<typename IterTy, bool concurrent = true>
class TileAdaptor {
  typedef typename std::iterator_traits<IterTy>::value_type T;
  typedef typename boost::reference_wrapper<T> wlT;

  std::deque<wlT> wl;
  LL::SimpleLock<concurrent> Lock;

  typename T::iterator ii, ee;

public:
  typedef typename std::iterator_traits<typename T::iterator>::value_type value_type;

  template<bool newconcurrent>
  struct rethread {
    typedef TileAdaptor<IterTy, newconcurrent> WL;
  };

  template<typename Tnew>
  struct retype {
    typedef TileAdaptor<IterTy, concurrent> WL;
  };

  void push(wlT val) {
    Lock.lock();
    wl.push_front(val);
    Lock.unlock();
  }

  template<typename Iter>
  void push(Iter b, Iter e) {
    while (b != e)
      push(boost::ref(*b++));
  }

  template<typename Iter>
  void push_initial(Iter b, Iter e) {
    push(b,e);
  }

  boost::optional<value_type> pop() {
    Lock.lock();
    while (ii == ee) {
      if (wl.empty()) {
	Lock.unlock();
	return boost::optional<value_type>();
      } else {
	T t = wl.back();
	ii = t.begin();
	ee = t.end();
	wl.pop_back();
      }
    }
    boost::optional<value_type> retval = boost::optional<value_type>(*ii++);
    Lock.unlock();
    return retval;
  }
};

//FIXME: make this valid for input iterators (e.g. no default constructor)
template<typename IterTy = int*>
class ForwardAccessRange {
  //! Thread-local data
  struct TLD {
    IterTy begin;
    IterTy end;
    bool init;
  };

  PerCPU<TLD> tlds;
  unsigned num;
  IterTy gbegin;
  IterTy gend;

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
  void push(value_type val) {
    abort();
  }

  //! push a range onto the queue
  template<typename Iter>
  void push(Iter b, Iter e) {
    abort();
  }

  //stager each thread's start item
  void push_initial(IterTy b, IterTy e) {
    num = ThreadPool::getActiveThreads();
    gbegin = b;
    gend = e;
    for (unsigned i = 0; i < tlds.size(); ++i)
      tlds.get(i).init = false;
  }

  //! pop a value from the queue.
  // move through range in num thread strides
  boost::optional<value_type> pop() {
    TLD& tld = tlds.get();
    if (!tld.init) {
      tld.begin = Galois::safe_advance(gbegin, gend, tlds.myEffectiveID());
      tld.end = gend;
      tld.init = true;
    }
    if (tld.begin != tld.end) {
      boost::optional<value_type> retval = *tld.begin;
      tld.begin = Galois::safe_advance(tld.begin, tld.end, num);
      assert(*retval);
      return retval;
    }
    return boost::optional<value_type>();
  }
};
WLCOMPILECHECK(ForwardAccessRange);

template<bool Stealing, typename IterTy = int*>
class RandomAccessRange {
  //! Thread-local data
  struct TLD {
    IterTy begin;
    IterTy end;
    IterTy stealBegin;
    IterTy stealEnd;
    bool init;
    LL::SimpleLock<true> stealLock;
  };

  PerCPU<TLD> tlds;

  IterTy gbegin;
  IterTy gend;

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
    TLD& otld = tlds.getNext(ThreadPool::getActiveThreads());
    if (do_steal(tld, tlds.getNext(ThreadPool::getActiveThreads()))) {
      //redistribute stolen items for neighbor to steal too
      if (std::distance(tld.begin, tld.end) > 1) {
	tld.stealLock.lock();
	tld.stealEnd = tld.end;
	tld.stealBegin = tld.end = Galois::split_range(tld.begin, tld.end);
	tld.stealLock.unlock();
      }
    }
  }

public:

  //! T is the value type of the WL
  typedef typename std::iterator_traits<IterTy>::value_type value_type;

  template<bool newconcurrent>
  struct rethread {
    typedef RandomAccessRange<Stealing,IterTy> WL;
  };

  template<typename Tnew>
  struct retype {
    typedef RandomAccessRange<Stealing,IterTy> WL;
  };

  //! push a value onto the queue
  void push(value_type val) {
    abort();
  }

  //! push a range onto the queue
  template<typename Iter>
  void push(Iter b, Iter e) {
    abort();
  }

  //stager each thread's start item
  void push_initial(IterTy b, IterTy e) {
    gbegin = b;
    gend = e;
    for (unsigned i = 0; i < tlds.size(); ++i)
      tlds.get(i).init = false;
  }

  //! pop a value from the queue.
  // move through range in num thread strides
  boost::optional<value_type> pop() {
    TLD& tld = tlds.get();
    if (!tld.init) {
      unsigned num = ThreadPool::getActiveThreads();
      unsigned len = std::distance(gbegin,gend);
      unsigned per = (len + num - 1) / num;
      unsigned i = tlds.myEffectiveID();
      tld.begin = gbegin + per * i;
      tld.end = tld.begin + std::min(per, (unsigned)std::distance(tld.begin, gend));
      if (Stealing) {
	tld.stealEnd = tld.end;
	tld.stealBegin = tld.end = Galois::split_range(tld.begin, tld.end);
      }
      tld.init = true;
    }
    if (Stealing && tld.begin == tld.end)
      try_steal(tld);
    if (tld.begin != tld.end)
      return boost::optional<value_type>(*tld.begin++);
    return boost::optional<value_type>();
  }
};
WLCOMPILECHECK(RandomAccessRange);

template<typename OwnerFn, typename T = int>
class OwnerComputesWL : private boost::noncopyable {
  typedef ChunkedLIFO<256, T> cWL;
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
  
  void push(value_type val)  {
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
    push(b,e);
  }

  boost::optional<value_type> pop() {
    cWL& wl = Items.get();
    boost::optional<value_type> retval = wl.pop();
    if (retval)
      return retval;
    pWL& p = pushBuffer.get();
    while (retval = p.pop())
      wl.push(*retval);
    return wl.pop();
  }
};
//WLCOMPILECHECK(OwnerComputesWL);

//End namespace
}
}

#endif
