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

namespace GaloisRuntime {
namespace HIDDEN {

//! Alternative implementation of FixedSizeRing which is non-concurrent and
//! supports pushX/empty/popX rather than pushX/popX with boost::optional

template<typename T, unsigned ChunkSize>
struct FixedSizeBase: private boost::noncopyable {
  char datac[sizeof(T[ChunkSize])] __attribute__ ((aligned (__alignof__(T))));

  T* data() { return reinterpret_cast<T*>(&datac[0]); }
  T* at(unsigned i) { return &data()[i]; }
  void create(int i, const T& val) { new (at(i)) T(val); }
  void destroy(unsigned i) { (at(i))->~T(); }
  inline unsigned chunksize() const { return ChunkSize; }
  typedef T value_type;
};

template<typename T, bool isLIFO, int ChunkSize>
class FixedSizeRing: public FixedSizeBase<T,ChunkSize> {
  unsigned start;
  unsigned count;

public:
  FixedSizeRing(): start(0), count(0) { }

  unsigned size() const { return count; }
  bool empty() const { return count == 0; }
  bool full() const { return count == this->chunksize(); }

  void push(const T& val) {
    start = (start + this->chunksize() - 1) % this->chunksize();
    ++count;
    this->create(start, val);
  }

  void pop() {
    unsigned end = (start + count - 1) % this->chunksize();
    this->destroy(end);
    --count;
  }

  T& cur() { 
    unsigned end = (start + count - 1) % this->chunksize();
    return *this->at(end); 
  }
};


template<typename T, int ChunkSize>
class FixedSizeRing<T,true,ChunkSize>: public FixedSizeBase<T,ChunkSize>  {
  unsigned end;

public:
  FixedSizeRing(): end(0) { }

  unsigned size() const { return end; }
  bool empty() const { return end == 0; }
  bool full() const { return end >= this->chunksize(); }
  void pop() { this->destroy(--end); }
  T& cur() { return *this->at(end-1); }
  void push(const T& val) { this->create(end, val); ++end; }
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
  class Chunk : public FixedSizeRing<T,isLIFO,ChunkSize>, public OuterTy<Chunk,true>::ListNode {};

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
class dChunkedLIFO: public dChunkedMaster<T, WorkList::ConExtLinkedStack, true, ChunkSize> { };

template<typename T,int ChunkSize>
class dChunkedFIFO: public dChunkedMaster<T, WorkList::ConExtLinkedQueue, false, ChunkSize> { };

template<class T, class FunctionTy, template<typename,int> class WorklistTy>
class BSInlineExecutor {
  typedef T value_type;
  typedef WorklistTy<value_type,256> WLTy;

  struct ThreadLocalData {
    GaloisRuntime::UserContextAccess<value_type> facing;
    SimpleRuntimeContext cnx;
    LoopStatistics<ForEachTraits<FunctionTy>::NeedsStats> stat;
    ThreadLocalData(const char* ln): stat(ln) { }
  };

  GaloisRuntime::GBarrier barrier1;
  GaloisRuntime::GBarrier barrier2;
  WLTy wls[2];
  FunctionTy& function;
  const char* loopname;
  LL::CacheLineStorage<volatile long> done;
  unsigned numActive;

  bool empty(WLTy* wl) {
    return wl->sempty();
  }

  GALOIS_ATTRIBUTE_NOINLINE
  void abortIteration(ThreadLocalData& tld, const HIDDEN::WID& wid, WLTy* cur, WLTy* next) {
    tld.cnx.cancel_iteration();
    tld.stat.inc_conflicts();
    if (ForEachTraits<FunctionTy>::NeedsPush) {
      tld.facing.resetPushBuffer();
    }
    value_type& val = cur->cur(wid);
    next->push(wid, val);
    cur->pop(wid);
  }

  void processWithAborts(ThreadLocalData& tld, const HIDDEN::WID& wid, WLTy* cur, WLTy* next) {
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
    case GaloisRuntime::CONFLICT:
      abortIteration(tld, wid, cur, next);
      break;
    case GaloisRuntime::BREAK:
    default:
      abort();
    }
  }

  void process(ThreadLocalData& tld, const HIDDEN::WID& wid, WLTy* cur, WLTy* next) {
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
    HIDDEN::WID wid;

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

      barrier1.wait();

      if (tid == 0) {
        if (empty(cur))
          done.data = true;
      }
      
      barrier2.wait();

      if (done.data)
        break;
    }

    setThreadContext(0);
  }

public:
  BSInlineExecutor(FunctionTy& f, const char* ln): function(f), loopname(ln) { 
    if (ForEachTraits<FunctionTy>::NeedsBreak) {
      assert(0 && "not supported by this executor");
      abort();
    }

    numActive = galoisActiveThreads;
    barrier1.reinit(numActive);
    barrier2.reinit(numActive);
  }

  template<typename IterTy>
  bool AddInitialWork(IterTy b, IterTy e) {
    unsigned int a = numActive;
    unsigned int id = LL::getTID();
    unsigned dist = std::distance(b, e);
    unsigned num = (dist + a - 1) / a; //round up
    unsigned int A = std::min(num * id, dist);
    unsigned int B = std::min(num * (id + 1), dist);
    IterTy b2 = b;
    IterTy e2 = b;
    std::advance(b2, A);
    std::advance(e2, B);
    wls[0].push_initial(HIDDEN::WID(), b2, e2);
    return true;
  }

  void operator()() {
    go();
  }
};


} // end HIDDEN

namespace WorkList {
  template<bool isLIFO=true, class T=int>
  class BulkSynchronousInline { };
}

template<class T,class FunctionTy>
struct ForEachWork<WorkList::BulkSynchronousInline<true>,T,FunctionTy>:
  public HIDDEN::BSInlineExecutor<T,FunctionTy,HIDDEN::dChunkedLIFO> {
  typedef HIDDEN::BSInlineExecutor<T,FunctionTy,HIDDEN::dChunkedLIFO> SuperTy;
  ForEachWork(FunctionTy& f, const char* ln): SuperTy(f, ln) { }
};

template<class T,class FunctionTy>
struct ForEachWork<WorkList::BulkSynchronousInline<false>,T,FunctionTy>:
  public HIDDEN::BSInlineExecutor<T,FunctionTy,HIDDEN::dChunkedFIFO> {
  typedef HIDDEN::BSInlineExecutor<T,FunctionTy,HIDDEN::dChunkedFIFO> SuperTy;
  ForEachWork(FunctionTy& f, const char* ln): SuperTy(f, ln) { }
};

}
#endif
