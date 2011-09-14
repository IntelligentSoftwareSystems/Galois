// Scalable Local worklists -*- C++ -*-
/*
Galois, a framework to exploit amorphous data-parallelism in irregular
programs.

Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS SOFTWARE
AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY, FITNESS FOR ANY
PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF PERFORMANCE, AND ANY
WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF DEALING OR USAGE OF TRADE.
NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH RESPECT TO THE USE OF THE
SOFTWARE OR DOCUMENTATION. Under no circumstances shall University be liable
for incidental, special, indirect, direct or consequential damages or loss of
profits, interruption of business, or related expenses which may arise from use
of Software or Documentation, including but not limited to those resulting from
defects in Software and/or Documentation, or loss or inaccuracy of data of any
kind.
*/

#ifndef __WORKLIST_H_
#define __WORKLIST_H_

#include <queue>
#include <stack>
#include <limits>
#include <map>
#include <set>
#include <algorithm>
#include <boost/utility.hpp>

#include "Galois/Runtime/PaddedLock.h"
#include "Galois/Runtime/PerCPU.h"
//#include "Galois/Runtime/QueuingLock.h"
#include "Galois/Queue.h"

#include <boost/utility.hpp>

#include "mem.h"
#include "WorkListHelpers.h"

#ifndef WLCOMPILECHECK
#define WLCOMPILECHECK(name) //
#endif

#ifdef GALOIS_TBB
#define TBB_PREVIEW_CONCURRENT_PRIORITY_QUEUE 1
#include <tbb/concurrent_priority_queue.h>
#include <tbb/concurrent_queue.h>
#endif

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
  bool push(value_type val);

  //! push a range onto the queue
  template<typename Iter>
  bool push(Iter b, Iter e);

  //! pop a value from the queue.
  std::pair<bool, value_type> pop();
  
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


template<class Compare = std::less<int>, typename T = int, bool concurrent = true>
class PriQueue : private boost::noncopyable, private PaddedLock<concurrent> {

  std::priority_queue<T, std::vector<T>, Compare> wl;

  using PaddedLock<concurrent>::lock;
  using PaddedLock<concurrent>::try_lock;
  using PaddedLock<concurrent>::unlock;

public:
  template<bool newconcurrent>
  struct rethread {
    typedef PriQueue<Compare, T, newconcurrent> WL;
  };
  template<typename Tnew>
  struct retype {
    typedef PriQueue<Compare, Tnew, concurrent> WL;
  };

  typedef T value_type;

  bool push(value_type val) {
    lock();
    wl.push(val);
    unlock();
    return true;
  }

  template<typename Iter>
  bool push(Iter b, Iter e) {
    lock();
    while (b != e)
      wl.push(*b++);
    unlock();
    return true;
  }

  std::pair<bool, value_type> pop() {
    lock();
    if (wl.empty()) {
      unlock();
      return std::make_pair(false, value_type());
    } else {
      value_type retval = wl.top();
      wl.pop();
      unlock();
      return std::make_pair(true, retval);
    }
  }
};
WLCOMPILECHECK(PriQueue);

template<class Compare = std::less<int>, typename T = int, bool concurrent = true>
class PriQueueTree : private boost::noncopyable, private PaddedLock<concurrent> {

  std::multiset<T, Compare> wl;

  using PaddedLock<concurrent>::lock;
  using PaddedLock<concurrent>::try_lock;
  using PaddedLock<concurrent>::unlock;

public:
  template<bool newconcurrent>
  struct rethread {
    typedef PriQueueTree<Compare, T, newconcurrent> WL;
  };
  template<typename Tnew>
  struct retype {
    typedef PriQueueTree<Compare, Tnew, concurrent> WL;
  };

  typedef T value_type;

  bool push(value_type val) {
    lock();
    wl.insert(val);
    unlock();
    return true;
  }

  template<typename Iter>
  bool push(Iter b, Iter e) {
    lock();
    while (b != e)
      wl.insert(*b++);
    unlock();
    return true;
  }

  std::pair<bool, value_type> pop() {
    lock();
    if (wl.empty()) {
      unlock();
      return std::make_pair(false, value_type());
    } else {
      value_type retval = *wl.begin();
      wl.erase(wl.begin());
      unlock();
      return std::make_pair(true, retval);
    }
  }
};
WLCOMPILECHECK(PriQueueTree);

#ifdef GALOIS_TBB
template<class Compare = std::less<int>, typename T = int>
class TbbPriQueue : private boost::noncopyable {
  tbb::concurrent_priority_queue<T,Compare> wl;

public:
  template<bool newconcurrent>
  struct rethread {
    typedef TbbPriQueue<Compare, T> WL;
  };
  template<typename Tnew>
  struct retype {
    typedef TbbPriQueue<Compare, Tnew> WL;
  };

  typedef T value_type;

  bool push(value_type val) {
    wl.push(val);
    return true;
  }

  template<typename Iter>
  bool push(Iter b, Iter e) {
    while (b != e)
      wl.push(*b++);
    return true;
  }

  std::pair<bool, value_type> pop() {
    value_type retval;
    if (wl.try_pop(retval)) {
      return std::make_pair(true, retval);
    } else {
      return std::make_pair(false, value_type());
    }
  }
};
WLCOMPILECHECK(TbbPriQueue);

template<typename T = int>
class TbbFIFO : private boost::noncopyable  {
  tbb::concurrent_bounded_queue<T> wl;

public:
  template<bool newconcurrent>
  struct rethread {
    typedef TbbFIFO<T> WL;
  };
  template<typename Tnew>
  struct retype {
    typedef TbbFIFO<Tnew> WL;
  };

  typedef T value_type;

  bool push(value_type val) {
    wl.push(val);
    return true;
  }

  template<typename Iter>
  bool push(Iter b, Iter e) {
    while (b != e)
      wl.push(*b++);
    return true;
  }

  std::pair<bool, value_type> pop() {
    T V = T();
    bool B = wl.try_pop(V);
    return std::make_pair(B, V);
  }
};
WLCOMPILECHECK(TbbFIFO);

#endif

template<typename T = int, bool concurrent = true>
class LIFO : private boost::noncopyable, private PaddedLock<concurrent> {
  std::deque<T> wl;

  using PaddedLock<concurrent>::lock;
  using PaddedLock<concurrent>::try_lock;
  using PaddedLock<concurrent>::unlock;

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

  bool push(value_type val) {
    lock();
    wl.push_back(val);
    unlock();
    return true;
  }

  template<typename Iter>
  bool push(Iter b, Iter e) {
    lock();
    while (b != e)
      wl.push_back(*b++);
    unlock();
    return true;
  }

  std::pair<bool, value_type> pop()  {
    lock();
    if (wl.empty()) {
      unlock();
      return std::make_pair(false, value_type());
    } else {
      value_type retval = wl.back();
      wl.pop_back();
      unlock();
      return std::make_pair(true, retval);
    }
  }
};
WLCOMPILECHECK(LIFO);

template<typename T = int, bool concurrent = true>
class FIFO : private boost::noncopyable, private PaddedLock<concurrent>  {
  std::deque<T> wl;

  using PaddedLock<concurrent>::lock;
  using PaddedLock<concurrent>::try_lock;
  using PaddedLock<concurrent>::unlock;

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

  bool push(value_type val) {
    lock();
    wl.push_back(val);
    unlock();
    return true;
  }

  template<typename Iter>
  bool push(Iter b, Iter e) {
    lock();
    while (b != e)
      wl.push_back(*b++);
    unlock();
    return true;
  }

  std::pair<bool, value_type> pop() {
    lock();
    if (wl.empty()) {
      unlock();
      return std::make_pair(false, value_type());
    } else {
      value_type retval = wl.front();
      wl.pop_front();
      unlock();
      return std::make_pair(true, retval);
    }
  }
};
WLCOMPILECHECK(FIFO);

template<class Indexer = DummyIndexer, typename ContainerTy = FIFO<>, typename T = int, bool concurrent = true >
class OrderedByIntegerMetric : private boost::noncopyable {

  typedef typename ContainerTy::template rethread<concurrent>::WL CTy;

  struct perItem {
    CTy* current;
    unsigned int curVersion;
    unsigned int lastMasterVersion;
    std::map<int, CTy*> local;
  };

  std::vector<std::pair<int, CTy*> > masterLog;
  PaddedLock<concurrent> masterLock;
  unsigned int masterVersion;

  Indexer I;

  PerCPU<perItem> current;

  void updateLocal_i(perItem& p) {
    //ASSERT masterLock
    for (; p.lastMasterVersion < masterVersion; ++p.lastMasterVersion) {
      std::pair<int, CTy*> logEntry = masterLog[p.lastMasterVersion];
      CTy*& x = p.local[logEntry.first];
      // XXX: check this race
      assert(x == NULL || x == logEntry.second);
      p.local[logEntry.first] = logEntry.second;
    }
  }

  void updateLocal(perItem& p) {
    if (p.lastMasterVersion != masterVersion) {
      masterLock.lock();
      updateLocal_i(p);
      masterLock.unlock();
    }
  }

  CTy* updateLocalOrCreate(perItem& p, int i) {
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
      current.get(i).current = 0;
      current.get(i).lastMasterVersion = 0;
    }
  }

  ~OrderedByIntegerMetric() {
    for (typename std::vector<std::pair<int, CTy*> >::iterator ii = masterLog.begin(), ee = masterLog.end(); ii != ee; ++ii) {
      delete ii->second;
    }
  }

  bool push(value_type val) {
    unsigned int index = I(val);
    perItem& pI = current.get();
    //fastpath
    if (index == pI.curVersion && pI.current)
      return pI.current->push(val);
    //slow path
    CTy* lC = updateLocalOrCreate(pI, index);
    return lC->push(val);
  }

  template<typename Iter>
  bool push(Iter b, Iter e) {
    while (b != e)
      push(*b++);
    return true;
  }

  std::pair<bool, value_type> pop() {
    //Find a successful pop
    perItem& pI = current.get();
    CTy*& C = pI.current;
    std::pair<bool, value_type> retval;
    if (C && (retval = C->pop()).first)
      return retval;
    //Failed, find minimum bin
    updateLocal(pI);
    for (typename std::map<int, CTy*>::iterator ii = pI.local.begin(), ee = pI.local.end(); ii != ee; ++ii) {
      C = ii->second;
      pI.curVersion = ii->first;
      if ((retval = C->pop()).first)
	return retval;
    }
    retval.first = false;
    return retval;
  }
};
WLCOMPILECHECK(OrderedByIntegerMetric);

template<
  class Indexer = DummyIndexer,
  typename ContainerTy = FIFO<>,
  typename T = int,
  bool concurrent = true>
class LocalizedOrderedByIntegerMetric : private boost::noncopyable {

  typedef typename ContainerTy::template rethread<concurrent>::WL CTy;

  struct perItem {
    CTy* current;
    unsigned int curVersion;
    unsigned int lastMasterVersion;
    std::map<unsigned int, CTy*> local;
  };

  std::vector<std::pair<unsigned int, CTy*> > masterLog;
  PaddedLock<concurrent> masterLock;
  unsigned int masterVersion;

  Indexer I;
  const unsigned int numThreads;

  PerCPU<perItem> current;

  void updateLocal_i(perItem& p) {
    //ASSERT masterLock
    for (; p.lastMasterVersion < masterVersion; ++p.lastMasterVersion) {
      std::pair<unsigned int, CTy*> logEntry = masterLog[p.lastMasterVersion];
      CTy*& x = p.local[logEntry.first];
      // XXX: check this race
      assert(x == NULL || x == logEntry.second);
      x = logEntry.second;
    }
  }

  void updateLocal(perItem& p) {
    if (p.lastMasterVersion != masterVersion) {
      masterLock.lock();
      updateLocal_i(p);
      masterLock.unlock();
    }
  }

  CTy* updateLocalOrCreate(perItem& p, unsigned int i, unsigned int id) {
    //Try local then try update then find again or else create and update the master log
    CTy*& lC = p.local[i];
    if (lC)
      return lC;
    masterLock.lock();
    updateLocal_i(p);
    if (!lC) {
      CTy* bins = new CTy[numThreads];
      lC = &bins[id];
      ++masterVersion;
      masterLog.push_back(std::make_pair(i, bins));
    }
    masterLock.unlock();
    return lC;
  }

 public:
  template<bool newconcurrent>
  struct rethread {
    typedef LocalizedOrderedByIntegerMetric<Indexer,ContainerTy,T,newconcurrent> WL;
  };
  template<typename Tnew>
  struct retype {
    typedef LocalizedOrderedByIntegerMetric<Indexer,typename ContainerTy::template retype<Tnew>::WL,Tnew,concurrent> WL;
  };

  typedef T value_type;

  LocalizedOrderedByIntegerMetric(const Indexer& x = Indexer()):
    masterVersion(0),
    I(x),
    numThreads(getSystemThreadPolicy().getNumThreads())
  {
    for (unsigned int i = 0; i < current.size(); ++i) {
      current.get(i).current = 0;
      current.get(i).lastMasterVersion = 0;
    }
  }

  ~LocalizedOrderedByIntegerMetric() {
    for (typename std::vector<std::pair<unsigned, CTy*> >::iterator ii = masterLog.begin(), ee = masterLog.end(); ii != ee; ++ii) {
      delete [] ii->second;
    }
  }

  bool push(value_type val) {
    unsigned int index = I(val);
    unsigned int id = current.myEffectiveID();
    perItem& pI = current.get(id);
    //fastpath
    if (index == pI.curVersion && pI.current)
      return pI.current->push(val);
    //slow path
    CTy* lC = updateLocalOrCreate(pI, index, 0);
    return lC->push(val);
  }

  std::pair<bool, value_type> pop() {
    //Find a successful pop
    unsigned int id = current.myEffectiveID();
    perItem& pI = current.get(id);
    CTy*& C = pI.current;
    std::pair<bool, value_type> retval;
    if (C && (retval = C->pop()).first)
      return retval;
    //Failed, find minimum bin
    updateLocal(pI);
    for (typename std::map<unsigned int, CTy*>::iterator ii = pI.local.begin(), ee = pI.local.end(); ii != ee; ++ii) {
      pI.curVersion = ii->first;
      for (unsigned int i = 0; i < 1; ++i) {
        C = &ii->second[(0 + i) % numThreads];
        if ((retval = C->pop()).first)
          return retval;
      }
    }
    retval.first = false;
    return retval;
  }

  template<typename Iter>
  bool push(Iter ii, Iter ee) {
    while (ii != ee) {
      push(*ii++);
    }
    return true;
  }
};
WLCOMPILECHECK(LocalizedOrderedByIntegerMetric);

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

  bool push(value_type val) {
    return local.get().push(val);
  }

  template<typename Iter>
  bool push(Iter b, Iter e) {
    return local.get().push(b,e);
  }

  std::pair<bool, value_type> pop() {
    std::pair<bool, value_type> ret = local.get().pop();
    if (ret.first)
      return ret;
    ret = global.pop();
    return ret;
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

  bool push(value_type val) {
    return local.get().push(val);
  }

  template<typename Iter>
  bool push(Iter b, Iter e) {
    return local.get().push(b,e);
  }

  std::pair<bool, value_type> pop() {
    std::pair<bool, value_type> ret = local.get().pop();
    if (ret.first)
      return ret;
    return local.getNext().pop();
  }
};
WLCOMPILECHECK(LocalStealing);

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
    
    for (int i = 0; i < Q.size(); ++i) {
      ++id;
      id %= Q.size();
      r = popChunkByID(id);
      if (r) {
	//Chunk* r2 = popChunkByID(id);
	//if (r2)
	//pushChunk(r2);
	return r;
      }
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

  bool push(value_type val)  {
    p& n = data.get();
    if (n.next && n.next->push_back(val))
      return true;
    if (n.next)
      pushChunk(n.next);
    n.next = mkChunk();
    bool worked = n.next->push_back(val);
    assert(worked);
    return true;
  }

  template<typename Iter>
  bool push(Iter b, Iter e) {
    while (b != e)
      push(*b++);
    return true;
  }

  std::pair<bool, value_type> pop()  {
    p& n = data.get();
    std::pair<bool, value_type> retval;
    if (isStack) {
      if (n.next && (retval = n.next->pop_back()).first)
	return retval;
      if(n.next)
	delChunk(n.next);
      n.next = popChunk();
      if (n.next)
	return n.next->pop_back();
      return std::make_pair(false, value_type());
    } else {
      if (n.cur && (retval = n.cur->pop_front()).first)
	return retval;
      if(n.cur)
	delChunk(n.cur);
      n.cur = popChunk();
      if (!n.cur) {
	n.cur = n.next;
	n.next = 0;
      }
      if (n.cur)
	return n.cur->pop_front();
      return std::make_pair(false, value_type());
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

template<typename Partitioner = DummyPartitioner, typename T = int, typename ChildWLTy = dChunkedFIFO<>, bool concurrent=true>
class PartitionedWL : private boost::noncopyable {

  Partitioner P;
  PerCPU<ChildWLTy> Items;
  int active;

public:
  template<bool newconcurrent>
  struct rethread {
    typedef PartitionedWL<T, Partitioner, ChildWLTy, newconcurrent> WL;
  };

  typedef T value_type;
  
  PartitionedWL(const Partitioner& p = Partitioner()) :P(p), active(getSystemThreadPool().getActiveThreads()) {
    //std::cerr << active << "\n";
  }

  bool push(value_type val)  {
    unsigned int index = P(val);
    //std::cerr << "[" << index << "," << index % active << "]\n";
    return Items.get(index % active).push(val);
  }

  template<typename Iter>
  bool push(Iter b, Iter e) {
    while (b != e)
      push(*b++);
  }

  std::pair<bool, value_type> pop()  {
    std::pair<bool, value_type> r = Items.get().pop();
    // std::cerr << "{" << Items.myEffectiveID() << "}";
    // if (r.first)
    //   std::cerr << r.first;
    return r;
  }
  
  std::pair<bool, value_type> try_pop() {
    return pop();
  }
};

template<class Compare = std::less<int>, typename T = int>
class SkipListQueue : private boost::noncopyable {

  Galois::ConcurrentSkipListMap<T,int,Compare> wl;
  int magic;

public:
  template<bool newconcurrent>
  struct rethread {
    typedef SkipListQueue<Compare, T> WL;
  };
  template<typename Tnew>
  struct retype {
    typedef SkipListQueue<Compare, Tnew> WL;
  };

  typedef T value_type;

  bool push(value_type val) {
    wl.putIfAbsent(val, &magic);
    return true;
  }

  template<typename Iter>
  bool push(Iter b, Iter e) {
    while (b != e)
      push(*b++);
  }

  std::pair<bool, value_type> pop() {
    return wl.pollFirstKey();
  }
};
WLCOMPILECHECK(SkipListQueue);

template<class Compare = std::less<int>, typename T = int>
class FCPairingHeapQueue : private boost::noncopyable {

  Galois::FCPairingHeap<T,Compare> wl;

public:

  template<bool newconcurrent>
  struct rethread {
    typedef FCPairingHeapQueue<Compare, T> WL;
  };
  template<typename Tnew>
  struct retype {
    typedef FCPairingHeapQueue<Compare, Tnew> WL;
  };

  typedef T value_type;

  bool push(value_type val) {
    wl.add(val);
    return true;
  }

  template<typename Iter>
  bool push(Iter b, Iter e) {
    while (b != e)
      push(*b++);
  }

  std::pair<bool, value_type> pop() {
    return wl.pollMin();
  }
};
WLCOMPILECHECK(FCPairingHeapQueue);

template<typename Iter = int*, typename T = int>
class InitialIterator {
  struct Range {
    Iter ii;
    Iter ee;
  };
  Range Master;
  PaddedLock<true> L;
  PerCPU<Range> Ranges;
public:
  typedef T value_type;

  //! change the concurrency flag
  template<bool newconcurrent>
  struct rethread {
    typedef InitialIterator<Iter, T> WL;
  };

  //! change the type the worklist holds
  template<typename Tnew>
  struct retype {
    typedef InitialIterator<Iter, Tnew> WL;
  };

  bool push(value_type val) {
    assert(0 && "cannot push into InitialIterator Worklist");
    abort();
  }

  template<typename Iter2>
  bool push(Iter2 b, Iter2 e) {
    assert(0 && "cannot push into InitialIterator Worklist");
    abort();
  }

  std::pair<bool, value_type> pop() {
    Range& myR = Ranges.get();
    if (myR.ii != myR.ee)
      return std::make_pair(true, *myR.ii++);

    L.lock();
    myR.ii = Master.ii;
    //FIXME: specialize for random access iterators
    //for (int i = 0; i < 1024 && Master.ii != Master.ee; ++i, ++Master.ii) ;
    //int num = std::min(256, (int)std::distance(Master.ii, Master.ee));
    int d = (int)std::distance(Master.ii, Master.ee);
    int num = std::min(std::max(16, d / 32), d);
    std::advance(Master.ii, num);
    myR.ee = Master.ii;
    L.unlock();

    if (myR.ii == myR.ee)
      return std::make_pair(false, value_type());
    else
      return std::make_pair(true, *myR.ii++);
  }

  void fill_initial(Iter b, Iter e) {
    Master.ii = b;
    Master.ee = e;
  }
};
WLCOMPILECHECK(InitialIterator);

//End namespace
}
}

#endif
