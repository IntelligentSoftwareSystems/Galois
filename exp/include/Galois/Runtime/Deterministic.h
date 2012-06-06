/** Deterministic execution -*- C++ -*-
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
 * @section Description
 *
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#ifndef GALOIS_RUNTIME_DETERMINISTIC_H
#define GALOIS_RUNTIME_DETERMINISTIC_H

#ifdef GALOIS_DET

#include "Galois/Runtime/DualLevelIterator.h"

#include <boost/iterator/iterator_facade.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/counting_iterator.hpp>

namespace GaloisRuntime {
namespace Deterministic {

//! Wrapper around WorkList::ChunkedFIFO to allow peek() and empty() and still have FIFO order
template<int chunksize,typename T>
struct FIFO {
  WorkList::ChunkedFIFO<chunksize,T,false> m_data;
  WorkList::ChunkedLIFO<16,T,false> m_buffer;
  size_t m_size;

  FIFO(): m_size(0) { }

  boost::optional<T> pop() {
    boost::optional<T> p;
    if ((p = m_buffer.pop()) || (p = m_data.pop())) {
      --m_size;
    }
    return p;
  }

  boost::optional<T> peek() {
    boost::optional<T> p;
    if ((p = m_buffer.pop())) {
      m_buffer.push(*p);
    } else if ((p = m_data.pop())) {
      m_buffer.push(*p);
    }
    return p;
  }

  void push(const T& item) {
    m_data.push(item);
    ++m_size;
  }

  bool empty() const {
    return m_size == 0;
  }
};

template<typename T>
struct WorkItem {
  T item;
  unsigned long id;
  SimpleRuntimeContext* cnx;
  void* localState;

  WorkItem(const T& _item, unsigned long _id): item(_item), id(_id), cnx(NULL), localState(NULL) { }
};

template<typename T>
struct has_local_state {
  typedef char yes[1];
  typedef char no[2];
  template<typename C> static yes& test(typename C::LocalState*);
  template<typename> static no& test(...);
  static const bool value = sizeof(test<T>(0)) == sizeof(yes);
};

template<typename T,typename FunctionTy,bool hasLocalState = false>
struct StateManager { 
  void alloc(GaloisRuntime::UserContextAccess<T>&, FunctionTy* self) { }
  void dealloc(GaloisRuntime::UserContextAccess<T>&) { }
  void save(GaloisRuntime::UserContextAccess<T>&, void*&) { }
  void restore(GaloisRuntime::UserContextAccess<T>&, void* ) { } 
};

template<typename T,typename FunctionTy>
struct StateManager<T,FunctionTy,true> {
  typedef typename FunctionTy::LocalState LocalState;
  void alloc(GaloisRuntime::UserContextAccess<T>& c,FunctionTy* self) {
    void *p = c.data().getPerIterAlloc().allocate(sizeof(LocalState));
    new (p) LocalState(self, c.data().getPerIterAlloc());
    c.setLocalState(p, false);
  }
  void dealloc(GaloisRuntime::UserContextAccess<T>& c) {
    bool dummy;
    LocalState *p = (LocalState*) c.data().getLocalState(dummy);
    p->~LocalState();
  }
  void save(GaloisRuntime::UserContextAccess<T>& c, void*& localState) { 
    bool dummy;
    localState = c.data().getLocalState(dummy);
  }
  void restore(GaloisRuntime::UserContextAccess<T>& c, void* localState) { 
    c.setLocalState(localState, true);
  }
};

template<typename T,typename Function1Ty,typename Function2Ty,bool useLocalState>
class Executor {
  typedef T value_type;
  typedef WorkItem<T> Item;
  typedef std::pair<T,std::pair<unsigned long,unsigned> > NewItem;
  typedef WorkList::dChunkedFIFO<32,Item> WL;
  typedef WorkList::dChunkedFIFO<32,Item> PendingWork;
  typedef WorkList::ChunkedFIFO<32,Item,false> LocalPendingWork;
  typedef WorkList::dChunkedFIFO<32,NewItem> NewWork;
  typedef WorkList::dChunkedLIFO<32,SimpleRuntimeContext> ContextPool;
  typedef WorkList::dChunkedLIFO<32,SimpleRuntimeContext*> ContextPtrPool;

  // Truly thread-local
  struct ThreadLocalData: private boost::noncopyable {
    LocalPendingWork localPending;
    GaloisRuntime::UserContextAccess<value_type> facing;
    LoopStatistics<ForEachTraits<Function1Ty>::NeedsStats || ForEachTraits<Function2Ty>::NeedsStats> stat;
    size_t newSize;
    WL* wlcur;
    WL* wlnext;
    size_t rounds;
    size_t outerRounds;
    ThreadLocalData(const char* loopname): stat(loopname), rounds(0), outerRounds(0) { reset(); }
    void reset() {
      newSize = 0;
    }
  };

  // Mostly local but shared sometimes
  struct MergeInfo {
    typedef std::vector<NewItem> NewItemsTy;
    typedef FIFO<256,Item> ReserveTy;
    NewItemsTy newItems;
    ReserveTy reserve;

    size_t size;
    size_t window;
    size_t delta;
    size_t committed;
    size_t iterations;

    MergeInfo() { reset(); }
    
    void calculateWindow(PerCPU<MergeInfo>&, unsigned numActive);

    void nextWindow(WL* wl, bool renumbered) {
      if (renumbered)
        window = delta;
      else
        window += delta;
      boost::optional<Item> p;
      while ((p = reserve.peek())) {
        if (p->id > window)
          break;
        wl->push(*p);
        reserve.pop();
      }
    }

    void reset() {
      committed = 0;
      iterations = 0;
    }
  };

  typedef typename MergeInfo::NewItemsTy NewItemsTy;
  typedef typename NewItemsTy::iterator NewItemsIterator;

  struct GetNewItem: public std::unary_function<int,NewItemsTy&> {
    PerCPU<MergeInfo>* base;
    GetNewItem() { }
    GetNewItem(PerCPU<MergeInfo>* b): base(b) { }
    NewItemsTy& operator()(int i) const { return base->get(i).newItems; }
  };

  typedef boost::transform_iterator<GetNewItem, boost::counting_iterator<int> > MergeOuterIt;
  typedef DualLevelIterator<MergeOuterIt> MergeIt;

  PerCPU<MergeInfo> mergeInfo;
  typename MergeInfo::NewItemsTy mergeBuf;
  std::vector<value_type> distributeBuf;
  GBarrier barrier[8];
  WL worklists[2];
  NewWork new_;
  PendingWork pending;
  ContextPool contextPool;
  ContextPtrPool contextPtrPool;
  Function1Ty& function1;
  Function2Ty& function2;
  StateManager<T,Function1Ty,useLocalState> stateManager;
  const char* loopname;
  LL::CacheLineStorage<volatile long> innerDone;
  LL::CacheLineStorage<volatile long> outerDone;
  int numActive;

  bool merge(int begin, int end);
  bool renumber(ThreadLocalData& tld);
  bool pendingLoop(ThreadLocalData& tld);
  bool commitLoop(ThreadLocalData& tld);
  void go();

  struct LessThan {
    bool operator()(const NewItem& a, const NewItem& b) const {
      if (a.second.first < b.second.first)
        return true;
      else if (a.second.first == b.second.first) 
        return a.second.second < b.second.second;
      else
        return false;
    }
  };

  struct EqualTo {
    bool operator()(const NewItem& a, const NewItem& b) const {
      return a.second.first == b.second.first && a.second.second == b.second.second;
    }
  };

  struct GetFirst: public std::unary_function<NewItem,const value_type&> {
    const value_type& operator()(const NewItem& x) const {
      return x.first;
    }
  };

  SimpleRuntimeContext* nextContext() {
    contextPool.push(SimpleRuntimeContext());
    SimpleRuntimeContext* retval = contextPool.unsafePeek();
    if (useLocalState)
      contextPtrPool.push(retval);
    return retval;
  }

  void commitContexts() {
    boost::optional<SimpleRuntimeContext> p;
    while ((p = contextPool.pop())) {
      p->commit_iteration();
    }
  }

  template<typename InputIteratorTy>
  void safe_advance(InputIteratorTy& it, size_t d, size_t& cur, size_t dist) {
    if (d + cur >= dist) {
      d = dist - cur;
    }
    std::advance(it, d);
    cur += d;
  }

  //! Slightly complicated reindexing to separate out continuous elements in InputIterator
  template<typename InputIteratorTy>
  void redistribute(InputIteratorTy b, InputIteratorTy e, size_t dist) {
    unsigned int tid = LL::getTID();
    //const size_t numBlocks = 1 << 7;
    //const size_t mask = numBlocks - 1;
    //size_t blockSize = dist / numBlocks; // round down
    MergeInfo& minfo = mergeInfo.get();
    size_t blockSize = minfo.delta;
    size_t numBlocks = dist / minfo.delta;
    
    size_t cur = 0;
    safe_advance(b, tid, cur, dist);
    while (b != e) {
      unsigned long id;
      if (cur < blockSize * numBlocks)
        //id = (cur & mask) * blockSize + (cur / numBlocks);
        id = (cur % numBlocks) * blockSize + (cur / numBlocks);
      else
        id = cur;
      distributeBuf[id] = *b;
      safe_advance(b, numActive, cur, dist);
    }
  }

  template<typename InputIteratorTy>
  void copyIn(InputIteratorTy b, InputIteratorTy e, size_t dist, WL* wl) {
    unsigned int tid = LL::getTID();
    MergeInfo& minfo = mergeInfo.get();
    size_t cur = 0;
    size_t k = 0;
    safe_advance(b, tid, cur, dist);
    while (b != e) {
      unsigned long id = k * numActive + tid + 1;
      if (id > minfo.delta)
        minfo.reserve.push(Item(*b, id));
      else
        wl->push(Item(*b, id));
      ++k;
      safe_advance(b, numActive, cur, dist);
    }
  }

  template<typename InputIteratorTy>
  void distribute(InputIteratorTy b, InputIteratorTy e, size_t dist, WL* wl) {
    unsigned int tid = LL::getTID();
#if 1
    if (tid == 0) {
      distributeBuf.resize(dist);
    }
    barrier[4].wait();
    redistribute(b, e, dist);
    barrier[5].wait();
    copyIn(distributeBuf.begin(), distributeBuf.end(), dist, wl);
#else
    copyIn(b, e, dist, wl);
#endif
  }

public:
  Executor(Function1Ty& f1, Function2Ty& f2, const char* ln): function1(f1), function2(f2), loopname(ln) { 
    numActive = (int) Galois::getActiveThreads();
    for (int i = 0; i < sizeof(barrier)/sizeof(*barrier); ++i)
      barrier[i].reinit(numActive);
    if (ForEachTraits<Function1Ty>::NeedsBreak || ForEachTraits<Function2Ty>::NeedsBreak) {
      assert(0 && "Break not supported");
      abort();
    }
  }

  template<typename IterTy>
  bool AddInitialWork(IterTy b, IterTy e) {
    unsigned int dist = std::distance(b, e);
    MergeInfo& minfo = mergeInfo.get();
    minfo.window = minfo.delta = std::max(dist / 100U, 1U);
    distribute(b, e, dist, &worklists[1]);

    return true;
  }

  void operator()() {
    go();
  }
};

template<typename T,typename Function1Ty,typename Function2Ty,bool useLocalState>
void Executor<T,Function1Ty,Function2Ty,useLocalState>::MergeInfo::calculateWindow(PerCPU<MergeInfo>& mergeInfo, unsigned numActive) 
{
  // Accumulate all threads' info
  size_t committed = 0;
  size_t iterations = 0;
  for (int i = 0; i < numActive; ++i) {
    MergeInfo& minfo = mergeInfo.get(i);
    committed += minfo.committed;
    iterations += minfo.iterations;
  }

  const float target = 0.90;
  float commitRatio = iterations > 0 ? committed / (float) iterations : 0.0;
  if (commitRatio > target)
    delta += delta;
  else
    delta = commitRatio / target * delta;
  delta = std::max(delta, (size_t) 1024);
  // XXX set max when we have local state to bound allocations....?
  //if (LL::getTID() == 0) {
  //  printf("%.3f (%zu/%zu) window: %zu delta: %zu\n", 
  //      commitRatio, committed, iterations, window, delta);
  //}
}

template<typename T,typename Function1Ty,typename Function2Ty,bool useLocalState>
void Executor<T,Function1Ty,Function2Ty,useLocalState>::go() {
  ThreadLocalData tld(loopname);
  MergeInfo& minfo = mergeInfo.get();
  tld.wlcur = &worklists[0];
  tld.wlnext = &worklists[1];

  while (true) {
    ++tld.outerRounds;

    while (true) {
      ++tld.rounds;
      //barrier[0].wait();

      std::swap(tld.wlcur, tld.wlnext);
      setPending(PENDING);
      bool nextPending = pendingLoop(tld);
      innerDone.data = true;

      barrier[1].wait();

      setPending(COMMITTING);
      bool nextCommit = commitLoop(tld);
      outerDone.data = true;
      if (nextPending || nextCommit)
        innerDone.data = false;

      barrier[2].wait();

      commitContexts();

      if (innerDone.data)
        break;

      barrier[0].wait();
    } 

    if (!minfo.reserve.empty()) {
      outerDone.data = false;
    }

    minfo.calculateWindow(mergeInfo, numActive);

    barrier[3].wait();

    minfo.reset();

    bool renumbered = false;
    if (outerDone.data) {
      if (!ForEachTraits<Function1Ty>::NeedsPush && !ForEachTraits<Function2Ty>::NeedsPush)
        break;
      if (renumber(tld))
        break;
      renumbered = true;
    }

    minfo.nextWindow(tld.wlnext, renumbered);
  }

  setPending(NON_DET);

  if (ForEachTraits<Function1Ty>::NeedsStats || ForEachTraits<Function2Ty>::NeedsStats) {
    if (LL::getTID() == 0) {
      reportStat(loopname, "RoundsExecuted", tld.rounds);
      reportStat(loopname, "OuterRoundsExecuted", tld.outerRounds);
    }
  }
}

template<typename T,typename Function1Ty,typename Function2Ty,bool useLocalState>
bool Executor<T,Function1Ty,Function2Ty,useLocalState>::merge(int begin, int end)
{
  if (begin == end)
    return false;
  else if (begin + 1 == end)
    return !mergeInfo.get(begin).newItems.empty();
  
  bool retval = false;
  int mid = (end - begin) / 2 + begin;
  retval |= merge(begin, mid);
  retval |= merge(mid, end);

  LessThan lessThan;
  MergeOuterIt bbegin(boost::make_counting_iterator(begin), GetNewItem(&mergeInfo));
  MergeOuterIt mmid(boost::make_counting_iterator(mid), GetNewItem(&mergeInfo));
  MergeOuterIt eend(boost::make_counting_iterator(end), GetNewItem(&mergeInfo));
  MergeIt aa(bbegin, mmid), ea(mmid, mmid);
  MergeIt bb(mmid, eend), eb(eend, eend);
  MergeIt cc(bbegin, eend), ec(eend, eend);

  while (aa != ea && bb != eb) {
    if (lessThan(*aa, *bb))
      mergeBuf.push_back(*aa++);
    else
      mergeBuf.push_back(*bb++);
  }

  for (; aa != ea; ++aa)
    mergeBuf.push_back(*aa);

  for (; bb != eb; ++bb)
    mergeBuf.push_back(*bb);

  for (NewItemsIterator ii = mergeBuf.begin(), ei = mergeBuf.end(); ii != ei; ++ii) 
    *cc++ = *ii; 

  mergeBuf.clear();

  assert(cc == ec);

  return retval;
}

template<typename T,typename Function1Ty,typename Function2Ty,bool useLocalState>
bool Executor<T,Function1Ty,Function2Ty,useLocalState>::renumber(ThreadLocalData& tld)
{
  MergeInfo& minfo = mergeInfo.get();

#if 1
  minfo.newItems.clear();
  minfo.newItems.reserve(tld.newSize * 2);
  boost::optional<NewItem> p;
  while ((p = new_.pop())) {
    minfo.newItems.push_back(*p);
  }

  std::sort(minfo.newItems.begin(), minfo.newItems.end(), LessThan());
  
  barrier[6].wait();

  unsigned tid = LL::getTID();
  if (tid == 0) {
    size_t size = 0;
    for (int i = 0; i < numActive; ++i)
      size += mergeInfo.get(i).newItems.size();

    mergeBuf.reserve(size);
    
    for (int i = 0; i < numActive; ++i)
      mergeInfo.get(i).size = size;

    outerDone.data = !merge(0, numActive);
  }

  barrier[7].wait();

  MergeOuterIt bbegin(boost::make_counting_iterator(0), GetNewItem(&mergeInfo));
  MergeOuterIt eend(boost::make_counting_iterator(numActive), GetNewItem(&mergeInfo));
  MergeIt ii(bbegin, eend), ei(eend, eend);

  distribute(boost::make_transform_iterator(ii, GetFirst()),
      boost::make_transform_iterator(ei, GetFirst()),
      minfo.size, tld.wlnext);
#else
  new_.flush();

  barrier[6].wait();
  
  if (LL::getTID() == 0) {
    mergeBuf.clear();
    mergeBuf.reserve(tld.newSize * numActive);
    boost::optional<NewItem> p;
    while ((p = new_.pop())) {
      mergeBuf.push_back(*p);
    }

    std::sort(mergeBuf.begin(), mergeBuf.end(), LessThan());

    unsigned long id = 0;
    outerDone.data = mergeBuf.empty();

    printf("R %ld\n", mergeBuf.size());
  }

  barrier[7].wait();

  distribute(boost::make_transform_iterator(mergeBuf.begin(), GetFirst()),
      boost::make_transform_iterator(mergeBuf.end(), GetFirst()),
      mergeBuf.size(), tld.wlnext);
#endif

  tld.reset();
  return outerDone.data;
}

template<typename T,typename Function1Ty,typename Function2Ty,bool useLocalState>
bool Executor<T,Function1Ty,Function2Ty,useLocalState>::pendingLoop(ThreadLocalData& tld)
{
  SimpleRuntimeContext* cnx = nextContext();
  MergeInfo& minfo = mergeInfo.get();
  bool retval = false;
  boost::optional<Item> p;
  while ((p = tld.wlcur->pop())) {
    ++minfo.iterations;
    bool commit = true;
    cnx->set_id(p->id);
    cnx->start_iteration();
    tld.stat.inc_iterations();
    setThreadContext(cnx);
    stateManager.alloc(tld.facing, &function1);
    try {
      function1(p->item, tld.facing.data());
    } catch (ConflictFlag i) {
      clearConflictLock();
      stateManager.dealloc(tld.facing);
      switch (i) {
        case CONFLICT: commit = false; break;
        case REACHED_FAILSAFE: break;
        default: assert(0 && "Unknown conflict flag"); abort(); break;
      }
    }

    if (ForEachTraits<Function1Ty>::NeedsPIA && !useLocalState)
      tld.facing.resetAlloc();

    if (commit) {
      p->cnx = cnx;
      if (useLocalState) {
        stateManager.save(tld.facing, p->localState);
        tld.localPending.push(*p);
      } else {
        pending.push(*p);
      }
    } else {
      tld.wlnext->push(*p);
      tld.stat.inc_conflicts();
      retval = true;
    }

    cnx = nextContext();
  }

  return retval;
}

template<typename T,typename Function1Ty,typename Function2Ty,bool useLocalState>
bool Executor<T,Function1Ty,Function2Ty,useLocalState>::commitLoop(ThreadLocalData& tld) 
{
  bool retval = false;
  MergeInfo& minfo = mergeInfo.get();
  boost::optional<Item> p;

  while ((p = (useLocalState) ? tld.localPending.pop() : pending.pop())) {
    bool commit = true;
    if (useLocalState && !p->cnx->is_ready())
      commit = false;

    if (commit) {
      try {
        setThreadContext(p->cnx);
        stateManager.restore(tld.facing, p->localState);
        function2(p->item, tld.facing.data());
      } catch (ConflictFlag i) {
        clearConflictLock();
        switch (i) {
          case CONFLICT: commit = false; break;
          default: assert(0 && "Unknown exception"); abort(); break;
        }
      }
    }

    stateManager.dealloc(tld.facing);
    
    if (commit) {
      ++minfo.committed;
      if (ForEachTraits<Function2Ty>::NeedsPush) {
        unsigned long parent = p->id;
        typedef typename UserContextAccess<value_type>::pushBufferTy::iterator iterator;
        unsigned count = 0;
        for (iterator ii = tld.facing.getPushBuffer().begin(), 
            ei = tld.facing.getPushBuffer().end(); ii != ei; ++ii) {
          new_.push(std::make_pair(*ii, std::make_pair(parent, ++count)));
          ++tld.newSize;
          if (count == 0) {
            assert(0 && "Counter overflow");
            abort();
          }
        }
      }
      assert(ForEachTraits<Function2Ty>::NeedsPush
          || tld.facing.getPushBuffer().begin() == tld.facing.getPushBuffer().end());
    } else {
      p->cnx = NULL;
      //if (useLocalState) p->localState = NULL;
      tld.wlnext->push(*p);
      tld.stat.inc_conflicts();
      retval = true;
    }

    if (ForEachTraits<Function2Ty>::NeedsPIA && !useLocalState)
      tld.facing.resetAlloc();

    tld.facing.resetPushBuffer();
  }

  if (ForEachTraits<Function2Ty>::NeedsPIA && useLocalState)
    tld.facing.resetAlloc();

  setThreadContext(0);
  return retval;
}

}
}

namespace Galois {
template<typename InitTy, typename WorkTy>
static inline void for_each_det_impl(InitTy& init, WorkTy& W) {
  using namespace GaloisRuntime;

  assert(!inGaloisForEach);

  inGaloisForEach = true;
  RunCommand w[4] = {Config::ref(init), 
		     Config::ref(getSystemBarrier()),
		     Config::ref(W),
		     Config::ref(getSystemBarrier())};
  getSystemThreadPool().run(&w[0], &w[4]);
  runAllLoopExitHandlers();
  inGaloisForEach = false;
}

template<bool useLocalState,typename IterTy, typename Function1Ty, typename Function2Ty>
static inline void for_each_det(IterTy b, IterTy e, Function1Ty f1, Function2Ty f2, const char* loopname = 0) {
  using namespace GaloisRuntime;

  const bool S = useLocalState && Deterministic::has_local_state<Function1Ty>::value;
  typedef typename std::iterator_traits<IterTy>::value_type T;
  typedef Deterministic::Executor<T,Function1Ty,Function2Ty,S> WorkTy;

  WorkTy W(f1, f2, loopname);
  Initializer<IterTy, WorkTy> init(b, e, W);

  for_each_det_impl(init, W);
}

template<bool useLocalState,typename IterTy, typename FunctionTy>
static inline void for_each_det(IterTy b, IterTy e, FunctionTy f, const char* loopname = 0) {
  Galois::for_each_det<useLocalState>(b, e, f, f, loopname);
}
}
#endif

#endif
