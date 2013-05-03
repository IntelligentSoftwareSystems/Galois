/** Simple Worklists that do not adhere to the general worklist contract -*- C++ -*-
 * @file
 * This is the only file to include for basic Galois functionality.
 *
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
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#ifndef GALOIS_RUNTIME_PARALLELWORKINLINE_H
#define GALOIS_RUNTIME_PARALLELWORKINLINE_H

#include "Galois/Runtime/ParallelWork.h"
#include <cstdio>

namespace Galois {
namespace Runtime {
namespace {

template<typename T, bool isLIFO, unsigned ChunkSize>
struct FixedSizeRingAdaptor: public Galois::FixedSizeRing<T,ChunkSize> {
  typedef typename FixedSizeRingAdaptor::reference reference;

  reference cur() { return isLIFO ? this->front() : this->back();  }

  template<typename U>
  void push(U&& val) {
    this->push_front(std::forward<U>(val));
  }

  void pop()  {
    if (isLIFO) this->pop_front();
    else this->pop_back();
  }
};

struct WID {
  unsigned tid;
  unsigned pid;
  WID(unsigned t): tid(t) {
    pid = LL::getLeaderForThread(tid);
  }
  WID() {
    tid = LL::getTID();
    pid = LL::getLeaderForThread(tid);
  }
};

template<typename T,template<typename,bool> class OuterTy, bool isLIFO,int ChunkSize>
class dChunkedMaster : private boost::noncopyable {
  class Chunk : public FixedSizeRingAdaptor<T,isLIFO,ChunkSize>, public OuterTy<Chunk,true>::ListNode {};

  MM::FixedSizeAllocator heap;

  struct p {
    Chunk* next;
  };

  typedef OuterTy<Chunk, true> LevelItem;

  PerThreadStorage<p> data;
  PerPackageStorage<LevelItem> Q;

  Chunk* mkChunk() {
    return new (heap.allocate(sizeof(Chunk))) Chunk();
  }
  
  void delChunk(Chunk* C) {
    C->~Chunk();
    heap.deallocate(C);
  }

  void pushChunk(const WID& id, Chunk* C)  {
    LevelItem& I = *Q.getLocal(id.pid);
    I.push(C);
  }

  Chunk* popChunkByID(unsigned int i)  {
    LevelItem* I = Q.getRemote(i);
    if (I)
      return I->pop();
    return 0;
  }

  Chunk* popChunk(const WID& id)  {
    Chunk* r = popChunkByID(id.pid);
    if (r)
      return r;
    
    for (unsigned int i = id.pid + 1; i < Q.size(); ++i) {
      r = popChunkByID(i);
      if (r) 
	return r;
    }

    for (unsigned int i = 0; i < id.pid; ++i) {
      r = popChunkByID(i);
      if (r)
	return r;
    }

    return 0;
  }

  void pushSP(const WID& id, p& n, const T& val);
  bool emptySP(const WID& id, p& n);
  void popSP(const WID& id, p& n);

public:
  typedef T value_type;

  dChunkedMaster() : heap(sizeof(Chunk)) {
    for (unsigned int i = 0; i < data.size(); ++i) {
      p& r = *data.getRemote(i);
      r.next = 0;
    }
  }

  void push(const WID& id, const value_type& val)  {
    p& n = *data.getLocal(id.tid);
    if (n.next && !n.next->full()) {
      n.next->push(val);
      return;
    }
    pushSP(id, n, val);
  }

  unsigned currentChunkSize(const WID& id) {
    p& n = *data.getLocal(id.tid);
    if (n.next) {
      return n.next->size();
    }
    return 0;
  }

  template<typename Iter>
  void push(const WID& id, Iter b, Iter e) {
    while (b != e)
      push(id, *b++);
  }

  template<typename Iter>
  void push_initial(const WID& id, Iter b, Iter e) {
    push(id, b, e);
  }

  value_type& cur(const WID& id) {
    p& n = *data.getLocal(id.tid);
    return n.next->cur();
  }

  bool empty(const WID& id) {
    p& n = *data.getRemote(id.tid);
    if (n.next && !n.next->empty())
      return false;
    return emptySP(id, n);
  }

  bool sempty() {
    WID id;
    for (unsigned i = 0; i < data.size(); ++i) {
      id.tid = i;
      id.pid = LL::getLeaderForThread(i);
      if (!empty(id))
        return false;
    }
    return true;
  }

  void pop(const WID& id)  {
    p& n = *data.getLocal(id.tid);
    if (n.next && !n.next->empty()) {
      n.next->pop();
      return;
    }
    popSP(id, n);
  }
};

template<typename T,template<typename,bool> class OuterTy, bool isLIFO,int ChunkSize>
void dChunkedMaster<T,OuterTy,isLIFO,ChunkSize>::popSP(const WID& id, p& n) {
  while (true) {
    if (n.next && !n.next->empty()) {
      n.next->pop();
      return;
    }
    if (n.next)
      delChunk(n.next);
    n.next = popChunk(id);
    if (!n.next)
      return;
  }
}

template<typename T,template<typename,bool> class OuterTy, bool isLIFO,int ChunkSize>
bool dChunkedMaster<T,OuterTy,isLIFO,ChunkSize>::emptySP(const WID& id, p& n) {
  while (true) {
    if (n.next && !n.next->empty())
      return false;
    if (n.next)
      delChunk(n.next);
    n.next = popChunk(id);
    if (!n.next)
      return true;
  }
}

template<typename T,template<typename,bool> class OuterTy, bool isLIFO,int ChunkSize>
void dChunkedMaster<T,OuterTy,isLIFO,ChunkSize>::pushSP(const WID& id, p& n, const T& val) {
  if (n.next)
    pushChunk(id, n.next);
  n.next = mkChunk();
  n.next->push(val);
}

template<typename T,int ChunkSize>
class Worklist: public dChunkedMaster<T, WorkList::ConExtLinkedQueue, true, ChunkSize> { };

template<class T, class FunctionTy>
class BSInlineExecutor {
  typedef T value_type;
  typedef Worklist<value_type,256> WLTy;

  struct ThreadLocalData {
    Galois::Runtime::UserContextAccess<value_type> facing;
    SimpleRuntimeContext cnx;
    LoopStatistics<ForEachTraits<FunctionTy>::NeedsStats> stat;
    ThreadLocalData(const char* ln): stat(ln) { }
  };

  WLTy wls[2];
  FunctionTy& function;
  const char* loopname;
  Galois::Runtime::Barrier& barrier;
  LL::CacheLineStorage<volatile long> done;

  bool empty(WLTy* wl) {
    return wl->sempty();
  }

  GALOIS_ATTRIBUTE_NOINLINE
  void abortIteration(ThreadLocalData& tld, const WID& wid, WLTy* cur, WLTy* next) {
    tld.cnx.cancel_iteration();
    tld.stat.inc_conflicts();
    if (ForEachTraits<FunctionTy>::NeedsPush) {
      tld.facing.resetPushBuffer();
    }
    value_type& val = cur->cur(wid);
    next->push(wid, val);
    cur->pop(wid);
  }

  void processWithAborts(ThreadLocalData& tld, const WID& wid, WLTy* cur, WLTy* next) {
    int result = 0;
#if GALOIS_USE_EXCEPTION_HANDLER
    try {
      process(tld, wid, cur, next);
    } catch (ConflictFlag const& flag) {
      clearConflictLock();
      result = flag;
    }
#else
    if ((result = setjmp(hackjmp)) == 0) {
      process(tld, wid, cur, next);
    }
#endif
    switch (result) {
    case 0: break;
    case Galois::Runtime::CONFLICT:
      abortIteration(tld, wid, cur, next);
      break;
    case Galois::Runtime::BREAK:
    default:
      abort();
    }
  }

  void process(ThreadLocalData& tld, const WID& wid, WLTy* cur, WLTy* next) {
    int cs = std::max(cur->currentChunkSize(wid), 1U);
    for (int i = 0; i < cs; ++i) {
      value_type& val = cur->cur(wid);
      tld.stat.inc_iterations();
      function(val, tld.facing.data());
      if (ForEachTraits<FunctionTy>::NeedsPush) {
        next->push(wid,
            tld.facing.getPushBuffer().begin(),
            tld.facing.getPushBuffer().end());
        tld.facing.resetPushBuffer();
      }
      if (ForEachTraits<FunctionTy>::NeedsAborts)
        tld.cnx.commit_iteration();
      cur->pop(wid);
    }
  }

  void go() {
    ThreadLocalData tld(loopname);
    setThreadContext(&tld.cnx);
    unsigned tid = LL::getTID();
    WID wid;

    WLTy* cur = &wls[0];
    WLTy* next = &wls[1];

    while (true) {
      while (!cur->empty(wid)) {
        if (ForEachTraits<FunctionTy>::NeedsAborts) {
          processWithAborts(tld, wid, cur, next);
        } else {
          process(tld, wid, cur, next);
        }
        if (ForEachTraits<FunctionTy>::NeedsPIA)
          tld.facing.resetAlloc();
      }

      std::swap(next, cur);

      barrier.wait();

      if (tid == 0) {
        if (empty(cur))
          done.data = true;
      }
      
      barrier.wait();

      if (done.data)
        break;
    }

    setThreadContext(0);
  }

public:
  BSInlineExecutor(FunctionTy& f, const char* ln): function(f), loopname(ln), barrier(getSystemBarrier()) { 
    if (ForEachTraits<FunctionTy>::NeedsBreak) {
      assert(0 && "not supported by this executor");
      abort();
    }
  }

  template<typename RangeTy>
  void AddInitialWork(RangeTy range) {
    wls[0].push_initial(WID(), range.local_begin(), range.local_end());
  }

  void initThread() {}

  void operator()() {
    go();
  }
};


} // end anonymouse
} // end runtime

namespace WorkList {
  template<class T=int>
  class BulkSynchronousInline { };
}

namespace Runtime {
namespace {

template<class T,class FunctionTy>
struct ForEachWork<WorkList::BulkSynchronousInline<>,T,FunctionTy>:
  public BSInlineExecutor<T,FunctionTy> {
  typedef BSInlineExecutor<T,FunctionTy> SuperTy;
  ForEachWork(FunctionTy& f, const char* ln): SuperTy(f, ln) { }
};

}
} // runtime


} //galois

#endif
