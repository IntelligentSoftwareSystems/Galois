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

#ifdef GALOIS_USE_DET

#include "Galois/Runtime/DualLevelIterator.h"

#include <boost/static_assert.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/counting_iterator.hpp>

#include <deque>

namespace GaloisRuntime {
namespace Deterministic {

static const int ChunkSize = 32;
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
struct DItem {
  T item;
  unsigned long id;
  SimpleRuntimeContext* cnx;
  void* localState;

  DItem(const T& _item, unsigned long _id): item(_item), id(_id), cnx(NULL), localState(NULL) { }
  DItem(const DItem<T>& o): item(o.item), id(o.id), cnx(o.cnx), localState(o.localState) { }
};

//! Some template meta programming
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
  void alloc(GaloisRuntime::UserContextAccess<T>&, FunctionTy& self) { }
  void dealloc(GaloisRuntime::UserContextAccess<T>&) { }
  void save(GaloisRuntime::UserContextAccess<T>&, void*&) { }
  void restore(GaloisRuntime::UserContextAccess<T>&, void*) { } 
};

template<typename T,typename FunctionTy>
struct StateManager<T,FunctionTy,true> {
  typedef typename FunctionTy::LocalState LocalState;
  void alloc(GaloisRuntime::UserContextAccess<T>& c,FunctionTy& self) {
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

template<typename T>
struct has_break_fn {
  typedef char yes[1];
  typedef char no[2];
  template<typename C> static yes& test(typename C::BreakFn*);
  template<typename> static no& test(...);
  static const bool value = sizeof(test<T>(0)) == sizeof(yes);
};

template<typename FunctionTy,bool hasBreak = false>
struct BreakManager {
  BreakManager(FunctionTy&) { }
  bool checkBreak() { return false; }
};

template<typename FunctionTy>
class BreakManager<FunctionTy,true> {
  GBarrier barrier[1];
  LL::CacheLineStorage<volatile long> done;
  typename FunctionTy::BreakFn breakFn;

public:
  BreakManager(FunctionTy& fn): breakFn(fn) { 
    int numActive = (int) Galois::getActiveThreads();
    for (int i = 0; i < sizeof(barrier)/sizeof(*barrier); ++i)
      barrier[i].reinit(numActive);
  }

  bool checkBreak() {
    runAllLoopExitHandlers();
    if (LL::getTID() == 0)
      done.data = breakFn();
    barrier[0].wait();
    return done.data;
  }
};

class ContextPool {
  typedef WorkList::dChunkedLIFO<ChunkSize,SimpleRuntimeContext> Pool;
  Pool pool;
public:
  SimpleRuntimeContext* next() {
    pool.push(SimpleRuntimeContext());
    SimpleRuntimeContext* retval = pool.unsafePeek();
    return retval;
  }

  void commitAll() {
    boost::optional<SimpleRuntimeContext> p;
    while ((p = pool.pop())) {
      p->commit_iteration();
    }
  }
};

template<typename T>
struct DNewItem { 
  T item;
  unsigned long parent;
  unsigned count;

  DNewItem(const T& _item, unsigned long _parent, unsigned _count): item(_item), parent(_parent), count(_count) { }
  DNewItem(const DNewItem& o): item(o.item), parent(o.parent), count(o.count) { }

  bool operator<(const DNewItem<T>& o) const {
    if (parent < o.parent)
      return true;
    else if (parent == o.parent)
      return count < o.count;
    else
      return false;
  }

  bool operator==(const DNewItem<T>& o) const {
    return parent == o.parent && count == o.count;
  }

  struct GetFirst: public std::unary_function<DNewItem<T>,const T&> {
    const T& operator()(const DNewItem<T>& x) const {
      return x.item;
    }
  };
};

template<typename> class DMergeManagerBase;
template<typename,typename,bool> class DMergeManager;

template<typename T>
class DMergeLocal {
  template<typename> friend class DMergeManagerBase;
  template<typename,typename,bool> friend class DMergeManager;

  typedef DItem<T> Item;
  typedef DNewItem<T> NewItem;
  typedef std::vector<NewItem,typename Galois::PerIterAllocTy::rebind<NewItem>::other> NewItemsTy;
  typedef FIFO<ChunkSize*8,Item> ReserveTy;

  Galois::IterAllocBaseTy heap;
  Galois::PerIterAllocTy alloc;
  size_t window;
  size_t delta;
  size_t committed;
  size_t iterations;
  size_t aborted;
  size_t size;
  NewItemsTy newItems;
  ReserveTy reserve;
  unsigned long minId;
  unsigned long maxId;
  
  void initialWindow(size_t w) {
    window = delta = w;
  }

  //! Update min and max id from sorted iterator
  template<typename BiIteratorTy>
  void updateMinMax(BiIteratorTy ii, BiIteratorTy ei) {
    minId = std::numeric_limits<unsigned long>::max();
    maxId = 0;
    if (ii != ei) {
      if (ii + 1 == ei) {
        minId = maxId = ii->parent;
      } else {
        minId = ii->parent;
        maxId = (ei-1)->parent;
      }
    }
  }

public:
  DMergeLocal(): alloc(&heap), newItems(alloc) { reset(); }

  void incrementIterations(bool firstRound) {
    if (firstRound)
      ++iterations;
  }

  void incrementCommitted(bool firstRound) {
    if (firstRound)
      ++committed;
  }

  template<typename WL>
  void nextWindow(WL* wl, bool newWork) {
    if (newWork)
      window = delta;
    else
//      window += (delta - aborted);
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
    aborted = 0;
  }

  bool empty() {
    return reserve.empty();
  }
};

template<typename T>
class DMergeManagerBase {
protected:
  //static const int MinDelta = ChunkSize;
  static const int MinDelta = ChunkSize * 40;

  typedef DItem<T> Item;
  typedef DNewItem<T> NewItem;
  typedef WorkList::dChunkedFIFO<ChunkSize,NewItem> NewWork;
  typedef DMergeLocal<T> MergeLocal;
  typedef typename MergeLocal::NewItemsTy NewItemsTy;
  typedef typename NewItemsTy::iterator NewItemsIterator;

  Galois::IterAllocBaseTy heap;
  Galois::PerIterAllocTy alloc;
  PerCPU<MergeLocal> data;

  NewWork new_;
  int numActive;

  template<typename InputIteratorTy>
  void safe_advance(InputIteratorTy& it, size_t d, size_t& cur, size_t dist) {
    if (d + cur >= dist) {
      d = dist - cur;
    }
    std::advance(it, d);
    cur += d;
  }

  template<typename InputIteratorTy,typename WL>
  void copyIn(InputIteratorTy b, InputIteratorTy e, size_t dist, WL* wl) {
    unsigned int tid = LL::getTID();
    MergeLocal& mlocal = this->data.get();
    size_t cur = 0;
    size_t k = 0;
    safe_advance(b, tid, cur, dist);
    while (b != e) {
      unsigned long id = k * this->numActive + tid + 1;
      if (id > mlocal.delta)
        mlocal.reserve.push(Item(*b, id));
      else
        wl->push(Item(*b, id));
      ++k;
      safe_advance(b, this->numActive, cur, dist);
    }
  }

public:
  DMergeManagerBase(): alloc(&heap) {
    numActive = (int) Galois::getActiveThreads();
  }

  MergeLocal& get() {
    return data.get();
  }

  void calculateWindow(bool inner) {
    MergeLocal& mlocal = data.get();

    // Accumulate all threads' info
    size_t allcommitted = 0;
    size_t alliterations = 0;
    for (int i = 0; i < numActive; ++i) {
      DMergeLocal<T>& mlocal = data.get(i);
      allcommitted += mlocal.committed;
      alliterations += mlocal.iterations;
    }

//    mlocal.aborted = alliterations - allcommitted;

    const float target = 0.95;
    float commitRatio = alliterations > 0 ? allcommitted / (float) alliterations : 0.0;

    if (commitRatio >= target)
      mlocal.delta += mlocal.delta;
    else if (allcommitted == 0) // special case when we don't execute anything
      mlocal.delta += mlocal.delta;
    else
      mlocal.delta = commitRatio / target * mlocal.delta;

    if (!inner)
      mlocal.delta = std::max(mlocal.delta, (size_t) MinDelta);
    else if (mlocal.delta < MinDelta)
      mlocal.delta = 0; //mlocal.aborted; // XXX;
//    if (LL::getTID() == 0) {
//      printf("%.3f (%zu/%zu) window: %zu delta: %zu\n", 
//          commitRatio, allcommitted, alliterations, mlocal.window, mlocal.delta);
//    }
  }
};

template<typename T>
struct has_id_fn {
  typedef char yes[1];
  typedef char no[2];
  template<typename C> static yes& test(typename C::IdFn*);
  template<typename> static no& test(...);
  static const bool value = sizeof(test<T>(0)) == sizeof(yes);
};

template<typename T,typename,bool = false>
class DMergeManager: public DMergeManagerBase<T> {
  typedef DMergeManagerBase<T> Base;
  typedef typename Base::Item Item;
  typedef typename Base::NewItem NewItem;
  typedef typename Base::MergeLocal MergeLocal;
  typedef typename Base::NewItemsTy NewItemsTy;
  typedef typename Base::NewItemsIterator NewItemsIterator;

  struct GetNewItem: public std::unary_function<int,NewItemsTy&> {
    PerCPU<MergeLocal>* base;
    GetNewItem() { }
    GetNewItem(PerCPU<MergeLocal>* b): base(b) { }
    NewItemsTy& operator()(int i) const { return base->get(i).newItems; }
  };

  typedef boost::transform_iterator<GetNewItem, boost::counting_iterator<int> > MergeOuterIt;
  typedef DualLevelIterator<MergeOuterIt> MergeIt;

  std::vector<NewItem,typename Galois::PerIterAllocTy::rebind<NewItem>::other> mergeBuf;
  std::vector<T,typename Galois::PerIterAllocTy::rebind<T>::other> distributeBuf;

  GBarrier barrier[4];

  bool merge(int begin, int end) {
    if (begin == end)
      return false;
    else if (begin + 1 == end)
      return !this->data.get(begin).newItems.empty();
    
    bool retval = false;
    int mid = (end - begin) / 2 + begin;
    retval |= merge(begin, mid);
    retval |= merge(mid, end);

    MergeOuterIt bbegin(boost::make_counting_iterator(begin), GetNewItem(&this->data));
    MergeOuterIt mmid(boost::make_counting_iterator(mid), GetNewItem(&this->data));
    MergeOuterIt eend(boost::make_counting_iterator(end), GetNewItem(&this->data));
    MergeIt aa(bbegin, mmid), ea(mmid, mmid);
    MergeIt bb(mmid, eend), eb(eend, eend);
    MergeIt cc(bbegin, eend), ec(eend, eend);

    while (aa != ea && bb != eb) {
      if (*aa < *bb)
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

  //! Slightly complicated reindexing to separate out continuous elements in InputIterator
  template<typename InputIteratorTy>
  void redistribute(InputIteratorTy b, InputIteratorTy e, size_t dist) {
    unsigned int tid = LL::getTID();
    //const size_t numBlocks = 1 << 7;
    //const size_t mask = numBlocks - 1;
    //size_t blockSize = dist / numBlocks; // round down
    MergeLocal& mlocal = this->data.get();
    //size_t blockSize = std::max((size_t) (0.9*minfo.delta), (size_t) 1);
    size_t blockSize = mlocal.delta;
    size_t numBlocks = dist / blockSize;
    
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
      safe_advance(b, this->numActive, cur, dist);
    }
  }

  template<typename InputIteratorTy,typename WL>
  void distribute(InputIteratorTy b, InputIteratorTy e, size_t dist, WL* wl) {
    unsigned int tid = LL::getTID();
    MergeLocal& mlocal = this->data.get();
    mlocal.initialWindow(std::max(dist / 100, (size_t) Base::MinDelta));
#if 1
    if (tid == 0) {
      distributeBuf.resize(dist);
    }
    barrier[0].wait();
    redistribute(b, e, dist);
    barrier[1].wait();
    copyIn(distributeBuf.begin(), distributeBuf.end(), dist, wl);
#else
    copyIn(b, e, dist, wl);
#endif
  }

public:
  DMergeManager(): mergeBuf(this->alloc), distributeBuf(this->alloc) {
    for (int i = 0; i < sizeof(barrier)/sizeof(*barrier); ++i)
      barrier[i].reinit(this->numActive);
  }

  template<typename InputIteratorTy, typename WL>
  void addInitialWork(InputIteratorTy b, InputIteratorTy e, WL* wl) {
    size_t dist = std::distance(b, e);
    distribute(b, e, dist, wl);
  }

  void pushNew(const T& item, unsigned long parent, unsigned count) {
    this->new_.push(NewItem(item, parent, count));
  }

  template<typename WL>
  void distributeNewWork(WL* wl) {
    MergeLocal& mlocal = this->data.get();

    // XXX take ids and dump
#if 1
    mlocal.newItems.clear();
    //minfo.newItems.reserve(tld.newSize * 2);
    boost::optional<NewItem> p;
    while ((p = this->new_.pop())) {
      mlocal.newItems.push_back(*p);
    }

    std::sort(mlocal.newItems.begin(), mlocal.newItems.end());
    
    barrier[2].wait();

    unsigned tid = LL::getTID();
    if (tid == 0) {
      size_t size = 0;
      for (int i = 0; i < this->numActive; ++i)
        size += this->data.get(i).newItems.size();

      mergeBuf.reserve(size);
      
      for (int i = 0; i < this->numActive; ++i)
        this->data.get(i).size = size;

      //!merge(0, numActive);
      merge(0, this->numActive);
    }

    barrier[3].wait();

    MergeOuterIt bbegin(boost::make_counting_iterator(0), GetNewItem(&this->data));
    MergeOuterIt eend(boost::make_counting_iterator(this->numActive), GetNewItem(&this->data));
    MergeIt ii(bbegin, eend), ei(eend, eend);

//    mlocal.initialWindow(std::max(mlocal.size / 100, (size_t) Base::MinDelta)); // XXX
    distribute(boost::make_transform_iterator(ii, typename Base::NewItem::GetFirst()),
        boost::make_transform_iterator(ei, typename Base::NewItem::GetFirst()),
        mlocal.size, wl);
#else
    this->new_.flush();

    barrier[2].wait();
    
    if (LL::getTID() == 0) {
      mergeBuf.clear();
      mergeBuf.reserve(tld.newSize * this->numActive);
      boost::optional<NewItem> p;
      while ((p = this->new_.pop())) {
        mergeBuf.push_back(*p);
      }

      std::sort(mergeBuf.begin(), mergeBuf.end());

      unsigned long id = 0;
      outerDone.data = mergeBuf.empty();

      printf("R %ld\n", mergeBuf.size());
    }

    barrier[3].wait();

    distribute(boost::make_transform_iterator(mergeBuf.begin(), typename NewItem::GetFirst()),
        boost::make_transform_iterator(mergeBuf.end(), typename NewItem::GetFirst()),
        mergeBuf.size(), tld.wlnext);
#endif
  }
};

// this specialization only selected when FunctionTy::IdFn exists
template<typename T,typename FunctionTy>
class DMergeManager<T,FunctionTy,true>: public DMergeManagerBase<T> {
  typedef DMergeManagerBase<T> Base;
  typedef typename Base::Item Item;
  typedef typename Base::NewItem NewItem;
  typedef typename Base::MergeLocal MergeLocal;
  typedef typename Base::NewItemsTy NewItemsTy;
  typedef typename Base::NewItemsIterator NewItemsIterator;
  typedef typename FunctionTy::IdFn IdFn;

  std::vector<NewItem,typename Galois::PerIterAllocTy::rebind<NewItem>::other> mergeBuf;

  GBarrier barrier[4];
  IdFn idFunction;

  void broadcastMinMax(MergeLocal& mlocal, unsigned int tid) {
    for (int i = 0; i < this->numActive; ++i) {
      if (i == tid) continue;
      DMergeLocal<T>& mother = this->data.get(i);
      mother.minId = mlocal.minId;
      mother.maxId = mlocal.maxId;
    }
  }

  template<typename InputIteratorTy,typename WL>
  void distribute(InputIteratorTy b, InputIteratorTy e, size_t dist, WL* wl) {
    unsigned int tid = LL::getTID();
    MergeLocal& mlocal = this->data.get();
    if (tid == 0) {
      mergeBuf.clear();
      mergeBuf.reserve(dist);
      for (; b != e; ++b) {
        unsigned long id = idFunction(*b);
        mergeBuf.push_back(NewItem(*b, id, 1));
      }
      std::sort(mergeBuf.begin(), mergeBuf.end());
      mlocal.updateMinMax(mergeBuf.begin(), mergeBuf.end());
      broadcastMinMax(mlocal, tid);
    }
    mlocal.initialWindow(std::max((mlocal.maxId - mlocal.minId) / 100, (size_t) Base::MinDelta));
    barrier[0].wait();
    copyIn(boost::make_transform_iterator(mergeBuf.begin(), typename Base::NewItem::GetFirst()),
        boost::make_transform_iterator(mergeBuf.end(), typename Base::NewItem::GetFirst()),
        dist, wl);
  }

public:
  DMergeManager(): mergeBuf(this->alloc) {
    for (int i = 0; i < sizeof(barrier)/sizeof(*barrier); ++i)
      barrier[i].reinit(this->numActive);
  }

  template<typename InputIteratorTy, typename WL>
  void addInitialWork(InputIteratorTy b, InputIteratorTy e, WL* wl) {
    size_t dist = std::distance(b, e);
    distribute(b, e, dist, wl);
  }

  void pushNew(const T& item, unsigned long parent, unsigned count) {
    this->new_.push(NewItem(item, idFunction(item), 1));
  }

  template<typename WL>
  void distributeNewWork(WL* wl) {
    unsigned int tid = LL::getTID();
    MergeLocal& mlocal = this->data.get();

    mlocal.newItems.clear();
    boost::optional<NewItem> p;
    while ((p = this->new_.pop())) {
      mlocal.newItems.push_back(*p);
    }

    std::sort(mlocal.newItems.begin(), mlocal.newItems.end());
    NewItemsIterator ii = mlocal.newItems.begin(), ei = mlocal.newItems.end();
    mlocal.updateMinMax(ii, ei);

    barrier[1].wait();
    
    if (tid == 0) {
      for (int i = 0; i < this->numActive; ++i) {
        DMergeLocal<T>& mother = this->data.get(i);
        mlocal.minId = std::min(mother.minId, mlocal.minId);
        mlocal.maxId = std::max(mother.maxId, mlocal.maxId);
      }
      broadcastMinMax(mlocal, tid);
    }

    barrier[2].wait();

    mlocal.initialWindow(std::max((mlocal.maxId - mlocal.minId) / 100, (size_t) Base::MinDelta));

    for (; ii != ei; ++ii) {
      unsigned long id = ii->parent;
      if (id > mlocal.delta)
        mlocal.reserve.push(Item(ii->item, id));
      else
        wl->push(Item(ii->item, id));
    }
  }
};

template<typename T,typename Function1Ty,typename Function2Ty,bool useLocalState>
class Executor {
  typedef T value_type;
  typedef DItem<T> Item;
  typedef DNewItem<T> NewItem;
  typedef DMergeLocal<T> MergeLocal;
  typedef WorkList::dChunkedFIFO<ChunkSize,Item> WL;
  typedef WorkList::dChunkedFIFO<ChunkSize,Item> PendingWork;
  typedef WorkList::ChunkedFIFO<ChunkSize,Item,false> LocalPendingWork;

  // Truly thread-local
  struct ThreadLocalData: private boost::noncopyable {
    LocalPendingWork localPending;
    GaloisRuntime::UserContextAccess<value_type> facing;
    LoopStatistics<ForEachTraits<Function1Ty>::NeedsStats || ForEachTraits<Function2Ty>::NeedsStats> stat;
    WL* wlcur;
    WL* wlnext;
    size_t rounds;
    size_t outerRounds;
    bool firstRound;
    bool hasNewWork;
    ThreadLocalData(const char* loopname): stat(loopname), rounds(0), outerRounds(0) { }
  };

  GBarrier barrier[4];
  WL worklists[2];
  DMergeManager<T,Function1Ty,has_id_fn<Function1Ty>::value> mergeManager;
  BreakManager<Function1Ty,has_break_fn<Function1Ty>::value> breakManager;
  PendingWork pending;
  ContextPool contextPool;
  Function1Ty& function1;
  Function2Ty& function2;
  StateManager<T,Function1Ty,useLocalState> stateManager;
  const char* loopname;
  LL::CacheLineStorage<volatile long> innerDone;
  LL::CacheLineStorage<volatile long> outerDone;
  LL::CacheLineStorage<volatile long> hasNewWork;
  int numActive;

  bool pendingLoop(ThreadLocalData& tld);
  bool commitLoop(ThreadLocalData& tld);
  void go();

public:
  Executor(Function1Ty& f1, Function2Ty& f2, const char* ln):
    breakManager(f1), function1(f1), function2(f2), loopname(ln)
  { 
    numActive = (int) Galois::getActiveThreads();
    for (int i = 0; i < sizeof(barrier)/sizeof(*barrier); ++i)
      barrier[i].reinit(numActive);
    //BOOST_STATIC_ASSERT(!ForEachTraits<Function1Ty>::NeedsBreak
    //    && !ForEachTraits<Function2Ty>::NeedsBreak
    //    || has_break_fn<Function1Ty>::value);
  }

  template<typename IterTy>
  bool AddInitialWork(IterTy b, IterTy e) {
    mergeManager.addInitialWork(b, e, &worklists[1]);

    return true;
  }

  void operator()() {
    go();
  }
};

template<typename T,typename Function1Ty,typename Function2Ty,bool useLocalState>
void Executor<T,Function1Ty,Function2Ty,useLocalState>::go() {
  ThreadLocalData tld(loopname);
  MergeLocal& mlocal = mergeManager.get();
  tld.wlcur = &worklists[0];
  tld.wlnext = &worklists[1];

  tld.hasNewWork = false;

  while (true) {
    ++tld.outerRounds;

    tld.firstRound = true;

    while (true) {
      ++tld.rounds;

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

      contextPool.commitAll();
      tld.firstRound = false;

      if (innerDone.data) {
        break;
      }

      mergeManager.calculateWindow(true);

      barrier[0].wait();

      mlocal.nextWindow(tld.wlnext, false);
      mlocal.reset();
      tld.firstRound = true;
    }

    if (!mlocal.empty())
      outerDone.data = false;

    if (tld.hasNewWork)
      hasNewWork.data = true;

    mergeManager.calculateWindow(false);

    if (breakManager.checkBreak()) {
      break;
    }

    barrier[3].wait();

    bool newWork = false;
    if (outerDone.data) {
      if (!ForEachTraits<Function1Ty>::NeedsPush && !ForEachTraits<Function2Ty>::NeedsPush)
        break;
      if (!hasNewWork.data)
        break;
      mergeManager.distributeNewWork(tld.wlnext);
      tld.hasNewWork = false;
      hasNewWork.data = false;
      newWork = true;
    }

    mlocal.nextWindow(tld.wlnext, newWork);
    mlocal.reset();
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
bool Executor<T,Function1Ty,Function2Ty,useLocalState>::pendingLoop(ThreadLocalData& tld)
{
  SimpleRuntimeContext* cnx = contextPool.next();
  MergeLocal& mlocal = mergeManager.get();
  bool retval = false;
  boost::optional<Item> p;
  while ((p = tld.wlcur->pop())) {
    mlocal.incrementIterations(tld.firstRound);
    bool commit = true;
    cnx->set_id(p->id);
    cnx->start_iteration();
    tld.stat.inc_iterations();
    setThreadContext(cnx);
    stateManager.alloc(tld.facing, function1);
    int result = 0;
#if GALOIS_USE_EXCEPTION_HANDLER
    try {
      function1(p->item, tld.facing.data());
    } catch (ConflictFlag flag) {
      clearConflictLock();
      result = flag;
    }
#else
    if ((result = setjmp(hackjmp)) == 0) {
      function1(p->item, tld.facing.data());
    }
#endif
    switch (result) {
      case 0: break;
      case CONFLICT: stateManager.dealloc(tld.facing); commit = false; break;
      case REACHED_FAILSAFE: stateManager.dealloc(tld.facing); break;
      default: assert(0 && "Unknown conflict flag"); abort(); break;
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

    cnx = contextPool.next();
  }

  return retval;
}

template<typename T,typename Function1Ty,typename Function2Ty,bool useLocalState>
bool Executor<T,Function1Ty,Function2Ty,useLocalState>::commitLoop(ThreadLocalData& tld) 
{
  bool retval = false;
  MergeLocal& mlocal = mergeManager.get();
  boost::optional<Item> p;

  while ((p = (useLocalState) ? tld.localPending.pop() : pending.pop())) {
    bool commit = true;
    if (useLocalState && !p->cnx->is_ready())
      commit = false;

    if (commit) {
      setThreadContext(p->cnx);
      stateManager.restore(tld.facing, p->localState);
      int result = 0;
#if GALOIS_USE_EXCEPTION_HANDLER
      try {
        function2(p->item, tld.facing.data());
      } catch (ConflictFlag flag) {
        clearConflictLock();
        result = flag;
      }
#else
      if ((result = setjmp(hackjmp)) == 0) {
        function2(p->item, tld.facing.data());
      }
#endif
      switch (result) {
        case 0: break;
        case CONFLICT: commit = false; break;
        default: assert(0 && "Unknown conflict flag"); abort(); break;
      }
    }

    stateManager.dealloc(tld.facing);
    
    if (commit) {
      mlocal.incrementCommitted(tld.firstRound);
      if (ForEachTraits<Function2Ty>::NeedsPush) {
        unsigned long parent = p->id;
        typedef typename UserContextAccess<value_type>::pushBufferTy::iterator iterator;
        unsigned count = 0;
        for (iterator ii = tld.facing.getPushBuffer().begin(), 
            ei = tld.facing.getPushBuffer().end(); ii != ei; ++ii) {
          mergeManager.pushNew(*ii, parent, ++count);
          tld.hasNewWork = true;
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
