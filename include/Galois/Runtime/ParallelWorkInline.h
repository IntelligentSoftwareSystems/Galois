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
  
  void delChunk(Chunk* C) {
    C->~Chunk();
    heap.deallocate(C);
  }

  void pushChunk(const WID& id, Chunk* C)  {
    LevelItem& I = Q.get(id.pid);
    I.push(C);
  }

  Chunk* popChunkByID(unsigned int i)  {
    LevelItem& I = Q.get(i);
    return I.pop();
  }

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

  dChunkedLIFO() : heap(sizeof(Chunk)) {
    for (unsigned int i = 0; i < data.size(); ++i) {
      p& r = data.get(i);
      r.next = 0;
    }
  }

  void push_backSP(const WID& id, p& n, value_type val);

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

  bool emptySP(const WID& id, p& n);

  bool empty(const WID& id) {
    p& n = data.get(id.tid);
    if (n.next && !n.next->empty())
      return false;
    return emptySP(id, n);
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

  void pop_backSP(const WID& id, p& n);

  void pop_back(const WID& id)  {
    p& n = data.get(id.tid);
    if (n.next && !n.next->empty()) {
      n.next->pop_back();
      return;
    }
    pop_backSP(id, n);
  }
};

template<typename T,int chunksize,bool concurrent>
void dChunkedLIFO<T,chunksize,concurrent>::pop_backSP(const WID& id, p& n) {
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

template<typename T,int chunksize,bool concurrent>
bool dChunkedLIFO<T,chunksize,concurrent>::emptySP(const WID& id, p& n) {
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

template<typename T,int chunksize,bool concurrent>
void dChunkedLIFO<T,chunksize,concurrent>::push_backSP(const WID& id, p& n, value_type val) {
  if (n.next)
    pushChunk(id, n.next);
  n.next = mkChunk();
  n.next->push_back(val);
}

} // end HIDDEN

template<class T, class FunctionTy>
class ForEachWork<WorkList::BulkSynchronous<>,T,FunctionTy,true>: public ForEachWorkBase<T,  FunctionTy> {
  typedef ForEachWorkBase<T, FunctionTy> Super;
  typedef typename Super::value_type value_type;
  typedef typename Super::ThreadLocalData ThreadLocalData;
  typedef HIDDEN::dChunkedLIFO<T,256> WLTy;

  GaloisRuntime::FastBarrier barrier1;
  GaloisRuntime::FastBarrier barrier2;
  WLTy wls[2];
  LL::CacheLineStorage<volatile long> done;
  unsigned numActive;

  bool empty(ThreadLocalData& tld, unsigned round) {
    return wls[round].sempty();
  }

  void go() {
    unsigned round = 0;
    ThreadLocalData& tld = Super::initWorker();
    unsigned tid = LL::getTID();

    HIDDEN::WID wid;

    WLTy* cur = &wls[round];
    WLTy* next = &wls[round+1];

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
      cur = &wls[round];

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
    wls[0].push_initial(HIDDEN::WID(), b, e);
    return true;
  }

  void operator()() {
    go();
  }
};

#ifdef GALOIS_DET

class ContextPool: private boost::noncopyable {
  template<typename T> struct LinkedNode { void *x,*y; T z; };
  typedef Galois::GFixedAllocator<LinkedNode<SimpleRuntimeContext> > Allocator;
  typedef std::list<SimpleRuntimeContext,Allocator> List;
  typedef List::iterator iterator;

  List list;
  iterator end;

public:
  ContextPool(): end(list.end()) { }

  SimpleRuntimeContext* next() {
    SimpleRuntimeContext* r;
    if (end != list.end()) {
      r = &*end;
      ++end;
    } else {
      list.push_back(SimpleRuntimeContext());
      r = &list.back();
      end = list.end();
    }
    return r;
  }

  void commit() {
    for (iterator ii = list.begin(); ii != end; ++ii)
      ii->commit_iteration();
    end = list.begin();
  }
};

template<class T, class FunctionTy, bool isSimple>
class ForEachWork<WorkList::Deterministic<>,T,FunctionTy,isSimple>: public ForEachWorkBase<T,  FunctionTy> {
  typedef ForEachWorkBase<T, FunctionTy> Super;
  typedef typename Super::value_type value_type;
  typedef typename Super::ThreadLocalData ThreadLocalData;
  typedef std::pair<T,unsigned long> WorkItem;
  typedef std::pair<WorkItem,SimpleRuntimeContext*> PendingItem;
  typedef std::pair<T,std::pair<unsigned long,unsigned> > NewItem;
  typedef HIDDEN::dChunkedLIFO<WorkItem,256> WL;
  typedef HIDDEN::dChunkedLIFO<PendingItem,256> Pending;
  typedef HIDDEN::dChunkedLIFO<NewItem,256> New;

  PerCPU<ContextPool> pool;
  FastBarrier barrier1;
  FastBarrier barrier2;
  FastBarrier barrier3;
  WL wl;
  Pending pending;
  New new_;
  LL::CacheLineStorage<volatile long> done;
  unsigned numActive;

  bool renumber();
  void pendingLoop(const HIDDEN::WID& wid, ThreadLocalData& tld);
  void commitLoop(const HIDDEN::WID& wid, ThreadLocalData& tld);
  void go();

  struct TotalOrder {
    bool operator()(const NewItem& a, const NewItem& b) const {
      if (a.second.first < b.second.first)
        return true;
      else if (a.second.first == b.second.first) 
        return a.second.second < b.second.second;
      else
        return false;
    }
  };

public:
  ForEachWork(FunctionTy& f, const char* loopname): Super(f, loopname) { 
    numActive = GaloisRuntime::getSystemThreadPool().getActiveThreads();
    barrier1.reinit(numActive);
    barrier2.reinit(numActive);
    barrier3.reinit(numActive);
    // TODO(ddn): support break
    if (ForeachTraits<FunctionTy>::NeedsBreak) abort();
  }

  template<typename IterTy>
  bool AddInitialWork(IterTy b, IterTy e) {
    HIDDEN::WID wid;
    WorkItem a[1];
    unsigned long id = 0;
    while (b != e) {
      a[0] = std::make_pair(*b, ++id);
      wl.push_initial(wid, &a[0], &a[1]);
      ++b;
    }
    return true;
  }

  void operator()() {
    go();
  }
};

template<class T,class FunctionTy,bool isSimple>
void ForEachWork<WorkList::Deterministic<>,T,FunctionTy,isSimple>::go() {
  unsigned round = 0;
  ThreadLocalData& tld = Super::initWorker();
  unsigned tid = LL::getTID();

  HIDDEN::WID wid;

  while (true) {
    setPending(true);
    pendingLoop(wid, tld);

    barrier1.wait();

    setPending(false);
    commitLoop(wid, tld);

    barrier2.wait();

    pool.get(wid.tid).commit();
    // TODO generate unique ids
    // TODO: !needsPush specialization
    if (tid == 0) {
      if (renumber())
        done.data = true;
    }
    barrier3.wait();

    if (done.data)
      break;
  }
  Super::deinitWorker();
}

template<class T,class FunctionTy,bool isSimple>
bool ForEachWork<WorkList::Deterministic<>,T,FunctionTy,isSimple>::renumber()
{
  std::vector<NewItem> buf;
  HIDDEN::WID wid;

  while (!new_.empty(wid)) {
    NewItem& p = new_.back(wid);
    buf.push_back(p);
    new_.pop_back(wid);
  }
  std::sort(buf.begin(), buf.end(), TotalOrder());
  unsigned long id = 0;
  bool retval = !buf.empty();
  for (typename std::vector<NewItem>::iterator ii = buf.begin(), ei = buf.end(); ii != ei; ++ii) {
    wl.push_back(wid, std::make_pair(ii->first, ++id));
  }

  return retval;
}

template<class T,class FunctionTy,bool isSimple>
void ForEachWork<WorkList::Deterministic<>,T,FunctionTy,isSimple>::pendingLoop(
    const HIDDEN::WID& wid,
    ThreadLocalData& tld)
{
  SimpleRuntimeContext* cnx = pool.get(wid.tid).next();
  setThreadContext(cnx);

  while (!wl.empty(wid)) {
    WorkItem& p = wl.back(wid);
    bool commit = true;
    try {
      cnx->setId(p.second);
      cnx->start_iteration();
      function(p.first, tld.facing);
    } catch (ConflictFlag i) {
      switch (i) {
        case CONFLICT: commit = false; break;
        case REACHED_FAILSAFE: break;
        default: assert(0 && "Unknown conflict flag"); abort(); break;
      }
    }

    if (tld.facing.__getPushBuffer().begin() != tld.facing.__getPushBuffer().end()) {
      assert(0 && "Pushed before failsafe");
      abort();
    }

    if (ForeachTraits<FunctionTy>::NeedsPIA)
      tld.facing.__resetAlloc();

    if (commit)
      pending.push_back(wid, std::make_pair(p, cnx));
    else
      new_.push_back(wid, std::make_pair(p.first, std::make_pair(p.second, 0)));

    wl.pop_back(wid);
    cnx = pool.get(wid.tid).next();
    setThreadContext(cnx);
  }
}

template<class T,class FunctionTy,bool isSimple>
void ForEachWork<WorkList::Deterministic<>,T,FunctionTy,isSimple>::commitLoop(
    const HIDDEN::WID& wid,
    ThreadLocalData& tld) 
{
  while (!pending.empty(wid)) {
    PendingItem& p = pending.back(wid);
    bool commit = true;
    try {
      setThreadContext(p.second);
      function(p.first.first, tld.facing);
    } catch (ConflictFlag i) {
      switch (i) {
        case CONFLICT: commit = false; break;
        default: assert(0 && "Unknown exception"); abort(); break;
      }
    }
    
    if (commit) {
      if (ForeachTraits<FunctionTy>::NeedsPush) {
        unsigned long parent = p.first.second;
        unsigned count = 0;
        typedef typename Galois::UserContext<value_type>::pushBufferTy::iterator iterator;
        for (iterator ii = tld.facing.__getPushBuffer().begin(), 
            ei = tld.facing.__getPushBuffer().end(); ii != ei; ++ii) {
          new_.push_back(wid, std::make_pair(*ii, std::make_pair(parent, ++count)));
        }
      }
      Super::incrementIterations(tld);
    } else {
      new_.push_back(wid, std::make_pair(p.first.first, std::make_pair(p.first.second, 0)));
    }

    if (ForeachTraits<FunctionTy>::NeedsPIA)
      tld.facing.__resetAlloc();

    tld.facing.__resetPushBuffer();

    pending.pop_back(wid);
  }

  setThreadContext(0);
}

#endif

}
#endif
