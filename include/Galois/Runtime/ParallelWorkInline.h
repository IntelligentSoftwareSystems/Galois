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

namespace GaloisRuntime {
namespace HIDDEN {

//! Alternative implementation of FixedSizeRing which is non-concurrent and
//! supports pushX/empty/popX rather than pushX/popX with boost::optional
template<typename T, int __chunksize = 64>
class FixedSizeLIFO : private boost::noncopyable {
  int end;

  char datac[sizeof(T[__chunksize])] __attribute__ ((aligned (__alignof__(T))));

  T* at(int i) {
    T* s = reinterpret_cast<T*>(&datac[0]);
    return &s[i];
  }

  inline int chunksize() const { return __chunksize; }

public:
  typedef T value_type;

  FixedSizeLIFO(): end(0) { }

  int size() const {
    return end;
  }

  bool empty() const {
    return end == 0;
  }

  bool full() const {
    return end >= chunksize();
  }

  void push_back(value_type val) {
    new (at(end++)) T(val);
  }

  T& back() {
    return *at(end-1);
  }

  void pop_back() {
    at(--end)->~T();
  }
};

struct WID {
  unsigned tid;
  unsigned pid;
  WID() {
    tid = LL::getTID();
    pid = LL::getPackageForThreadInternal(tid);
  }
};

template<typename T,int chunksize,bool concurrent=true>
class dChunkedLIFO : private boost::noncopyable {
private:
  class Chunk : public FixedSizeLIFO<T, chunksize>, public WorkList::ConExtLinkedStack<Chunk, concurrent>::ListNode {};

  MM::FixedSizeAllocator heap;

  struct p {
    Chunk* next;
  };

  typedef WorkList::ConExtLinkedStack<Chunk, concurrent> LevelItem;

  PerCPU<p,concurrent> data;
  PerLevel<LevelItem,concurrent> Q;

  Chunk* mkChunk() {
    return new (heap.allocate(sizeof(Chunk))) Chunk();
  }
  
  __attribute__((noinline))
  void delChunk(Chunk* C) {
    C->~Chunk();
    heap.deallocate(C);
  }

  __attribute__((noinline))
  void pushChunk(Chunk* C)  {
    LevelItem& I = Q.get();
    I.push(C);
  }

  Chunk* popChunkByID(unsigned int i)  {
    LevelItem& I = Q.get(i);
    return I.pop();
  }

  __attribute__((noinline))
  Chunk* popChunk()  {
    int pid = Q.myEffectiveID();
    Chunk* r = popChunkByID(pid);
    if (r)
      return r;
    
    for (int i = pid + 1; i < (int) Q.size(); ++i) {
      r = popChunkByID(i);
      if (r) 
	return r;
    }

    for (int i = 0; i < pid; ++i) {
      r = popChunkByID(i);
      if (r)
	return r;
    }

    return 0;
  }

public:
  typedef T value_type;

  dChunkedLIFO() : heap(sizeof(Chunk)) {
    for (unsigned int i = 0; i < data.size(); ++i) {
      p& r = data.get(i);
      r.next = 0;
    }
  }

  __attribute__((noinline))
  void push_backSP(p& n, value_type val) {
    if (n.next)
      pushChunk(n.next);
    n.next = mkChunk();
    n.next->push_back(val);
  }

  void push_back(const WID&, value_type val)  {
    p& n = data.get();
    if (n.next && !n.next->full()) {
      n.next->push_back(val);
      return;
    }
    push_backSP(n, val);
  }

  template<typename Iter>
  void push_back(const WID& i, Iter b, Iter e) {
    while (b != e)
      push_back(i, *b++);
  }

  template<typename Iter>
  void push_initial(Iter b, Iter e) {
    push_back(b, e);
  }

  value_type& back(const WID&) {
    p& n = data.get();
    return n.next->back();
  }

  __attribute__((noinline))
  bool emptySP(p& n) {
    while (true) {
      if (n.next && !n.next->empty())
        return false;
      if (n.next)
        delChunk(n.next);
      n.next = popChunk();
      if (!n.next)
        return true;
    }
  }

  bool sempty() {
    WID id;
    for (unsigned i = 0; i < data.size(); ++i) {
      id.tid = i;
      id.pid = LL::getPackageForThreadInternal(i);
      if (!empty(id))
        return false;
    }
    return true;
  }

  bool empty(const WID&) {
    p& n = data.get();
    if (n.next && !n.next->empty())
      return false;
    return
      emptySP(n);
  }

  __attribute__((noinline))
  void pop_backSP(p& n) {
    while (true) {
      if (n.next && !n.next->empty()) {
        n.next->pop_back();
        return;
      }
      if (n.next)
        delChunk(n.next);
      n.next = popChunk();
      if (!n.next)
        return;
    }
  }

  void pop_back(const WID&)  {
    p& n = data.get();
    if (n.next && !n.next->empty()) {
      n.next->pop_back();
      return;
    }
    pop_backSP(n);
  }
};

template<typename T,int chunksize,bool concurrent=true>
class dChunkedLIFOThreaded : private boost::noncopyable {
  class Chunk : public FixedSizeLIFO<T, chunksize>, public WorkList::ConExtLinkedStack<Chunk, concurrent>::ListNode {};

  MM::FixedSizeAllocator heap;

  struct p {
    Chunk* next;
  };

  typedef WorkList::ConExtLinkedStack<Chunk, concurrent> LevelItem;

  PerCPU<p> data;
  PerLevel<LevelItem> Q;

  Chunk* mkChunk() {
    return new (heap.allocate(sizeof(Chunk))) Chunk();
  }
  
  __attribute__((noinline))
  void delChunk(Chunk* C) {
    C->~Chunk();
    heap.deallocate(C);
  }

  __attribute__((noinline))
  void pushChunk(const WID& id, Chunk* C)  {
    LevelItem& I = Q.get(id.pid);
    I.push(C);
  }

  Chunk* popChunkByID(unsigned int i)  {
    LevelItem& I = Q.get(i);
    return I.pop();
  }

  __attribute__((noinline))
  Chunk* popChunk(const WID& id)  {
    int pid = id.pid;
    Chunk* r = popChunkByID(pid);
    if (r)
      return r;
    
    for (int i = pid + 1; i < (int) Q.size(); ++i) {
      r = popChunkByID(i);
      if (r) 
	return r;
    }

    for (int i = 0; i < pid; ++i) {
      r = popChunkByID(i);
      if (r)
	return r;
    }

    return 0;
  }

public:
  typedef T value_type;

  dChunkedLIFOThreaded() : heap(sizeof(Chunk)) {
    for (unsigned int i = 0; i < data.size(); ++i) {
      p& r = data.get(i);
      r.next = 0;
    }
  }

  __attribute__((noinline))
  void push_backSP(const WID& id, p& n, value_type val) {
    if (n.next)
      pushChunk(id, n.next);
    n.next = mkChunk();
    n.next->push_back(val);
  }

  void push_back(const WID& id, value_type val)  {
    p& n = data.get(id.tid);
    if (n.next && !n.next->full()) {
      n.next->push_back(val);
      return;
    }
    push_backSP(id, n, val);
  }

  template<typename Iter>
  void push_back(const WID& id, Iter b, Iter e) {
    while (b != e)
      push_back(id, *b++);
  }

  template<typename Iter>
  void push_initial(const WID& id, Iter b, Iter e) {
    push_back(id, b, e);
  }

  value_type& back(const WID& id) {
    p& n = data.get(id.tid);
    return n.next->back();
  }

  __attribute__((noinline))
  bool emptySP(const WID& id, p& n) {
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

  bool empty(const WID& id) {
    p& n = data.get(id.tid);
    if (n.next && !n.next->empty())
      return false;
    return
      emptySP(id, n);
  }

  bool sempty() {
    WID id;
    for (unsigned i = 0; i < data.size(); ++i) {
      id.tid = i;
      id.pid = LL::getPackageForThreadInternal(i);
      if (!empty(id))
        return false;
    }
    return true;
  }

  __attribute__((noinline))
  void pop_backSP(const WID& id, p& n) {
    while (true) {
      if (n.next && !n.next->empty()) {
        n.next->pop_back();
        return;
      }
      if (n.next)
        delChunk(n.next);
      n.next = popChunk(id);
      if (!n.next)
        return;
    }
  }

  void pop_back(const WID& id)  {
    p& n = data.get(id.tid);
    if (n.next && !n.next->empty()) {
      n.next->pop_back();
      return;
    }
    pop_backSP(id, n);
  }
};
} // end HIDDEN

template<class T, class FunctionTy>
class ForEachWork<WorkList::BulkSynchronous<>,T,FunctionTy,true>: public ForEachWorkBase<T,  FunctionTy> {
  typedef ForEachWorkBase<T, FunctionTy> Super;
  typedef typename Super::value_type value_type;
  typedef typename Super::ThreadLocalData ThreadLocalData;
  typedef HIDDEN::dChunkedLIFOThreaded<T,256> BagTy;

  GaloisRuntime::FastBarrier barrier1;
  GaloisRuntime::FastBarrier barrier2;
  BagTy bags[2];
  LL::CacheLineStorage<volatile long> done;
  unsigned numActive;

  bool empty(ThreadLocalData& tld, unsigned round) {
    return bags[round].sempty();
  }

  void go() {
    unsigned round = 0;
    ThreadLocalData& tld = Super::initWorker();
    unsigned tid = LL::getTID();

    HIDDEN::WID wid;

    BagTy* cur = &bags[round];
    BagTy* next = &bags[round+1];

    while (true) {
      while (!cur->empty(wid)) {
        value_type& p = cur->back(wid);
        function(p, tld.facing);
        
        Super::incrementIterations(tld);

        if (ForeachTraits<FunctionTy>::NeedsPush) {
          for (typename std::vector<value_type>::iterator ii = tld.facing.__getPushBuffer().begin(), 
              ei = tld.facing.__getPushBuffer().end(); ii != ei; ++ii)
            next->push_back(wid, *ii);
          tld.facing.__resetPushBuffer();
        }

        if (ForeachTraits<FunctionTy>::NeedsPIA)
          tld.facing.__resetAlloc();

        cur->pop_back(wid);
      }

      round = (round + 1) & 1;
      next = cur;
      cur = &bags[round];

      barrier1.wait();
      if (tid == 0) {
        if (empty(tld, round))
          done.data = true;
      }
      barrier2.wait();

      if (done.data)
        break;
    }
    Super::deinitWorker();
  }

public:
  ForEachWork(FunctionTy& f, const char* loopname): Super(f, loopname) { 
    numActive = GaloisRuntime::getSystemThreadPool().getActiveThreads();
    barrier1.reinit(numActive);
    barrier2.reinit(numActive);
  }

  template<typename IterTy>
  bool AddInitialWork(IterTy b, IterTy e) {
    bags[0].push_initial(HIDDEN::WID(), b, e);
    return true;
  }

  void operator()() {
    go();
  }
};

}
#endif
