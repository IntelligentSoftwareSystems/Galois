/** Deterministic execution -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
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
#ifndef GALOIS_RUNTIME_DETERMINISTICWORK_H
#define GALOIS_RUNTIME_DETERMINISTICWORK_H

#include "Galois/config.h"
#include "Galois/Threads.h"

#include "Galois/ParallelSTL/ParallelSTL.h"
#include "Galois/TwoLevelIteratorA.h"
#include "Galois/Runtime/ll/gio.h"

#include <boost/iterator/iterator_facade.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/counting_iterator.hpp>

#include GALOIS_CXX11_STD_HEADER(type_traits)
#include <deque>
#include <queue>

namespace Galois {
namespace Runtime {
//! Implementation of deterministic execution
namespace DeterministicImpl {

template<typename T>
struct DItem {
  T val;
  unsigned long id;
  void *localState;

  DItem(const T& _val, unsigned long _id): val(_val), id(_id), localState(NULL) { }
};

template<typename T>
struct DeterministicContext: public SimpleRuntimeContext {
  typedef DItem<T> Item;

  Item item;
  bool not_ready;

  DeterministicContext(const Item& _item): 
      SimpleRuntimeContext(true), 
      item(_item),
      not_ready(false)
  { }

  virtual void subAcquire(Lockable* lockable) {
    if (getPending() == COMMITTING)
      return;

    if (this->tryLock(lockable)) {
      this->addToNhood(lockable);
    }

    DeterministicContext* other;
    do {
      other = static_cast<DeterministicContext*>(this->getOwner(lockable));
      if (other == this)
        return;
      if (other) {
        bool conflict = other->item.id < this->item.id;
        if (conflict) {
          // A lock that I want but can't get
          not_ready = true;
          return; 
        }
      }
    } while (!this->stealByCAS(lockable, other));

    // Disable loser
    if (other) {
      other->not_ready = true; // Only need atomic write
    }

    return;
  }
};

namespace {

template<typename T>
struct DNewItem { 
  T val;
  unsigned long parent;
  unsigned count;

  DNewItem(const T& _val, unsigned long _parent, unsigned _count): val(_val), parent(_parent), count(_count) { }

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

  bool operator!=(const DNewItem<T>& o) const {
    return !(*this == o);
  }

  struct GetValue: public std::unary_function<DNewItem<T>,const T&> {
    const T& operator()(const DNewItem<T>& x) const {
      return x.val;
    }
  };
};

template<typename InputIteratorTy>
void safe_advance(InputIteratorTy& it, size_t d, size_t& cur, size_t dist) {
  if (d + cur >= dist) {
    d = dist - cur;
  }
  std::advance(it, d);
  cur += d;
}

//! Wrapper around WorkList::ChunkedFIFO to allow peek() and empty() and still have FIFO order
template<int ChunkSize,typename T>
struct FIFO {
  WorkList::ChunkedFIFO<ChunkSize,T,false> m_data;
  WorkList::ChunkedLIFO<16,T,false> m_buffer;
  size_t m_size;

  FIFO(): m_size(0) { }

  ~FIFO() {
    Galois::optional<T> p;
    while ((p = m_buffer.pop()))
      ;
    while ((p = m_data.pop()))
      ;
  }

  Galois::optional<T> pop() {
    Galois::optional<T> p;
    if ((p = m_buffer.pop()) || (p = m_data.pop())) {
      --m_size;
    }
    return p;
  }

  Galois::optional<T> peek() {
    Galois::optional<T> p;
    if ((p = m_buffer.pop())) {
      m_buffer.push(*p);
    } else if ((p = m_data.pop())) {
      m_buffer.push(*p);
    }
    return p;
  }

  void push(const T& val) {
    m_data.push(val);
    ++m_size;
  }

  size_t size() const {
    return m_size;
  }

  bool empty() const {
    return m_size == 0;
  }
};

template<typename T, typename Function1Ty, typename Function2Ty>
struct Options {
  typedef T value_type;
  typedef Function1Ty function1_type;
  typedef Function2Ty function2_type;

  static const bool needsStats = ForEachTraits<function1_type>::NeedsStats || ForEachTraits<function2_type>::NeedsStats;
  static const bool needsPush = ForEachTraits<function1_type>::NeedsPush || ForEachTraits<function2_type>::NeedsPush;
  static const bool needsBreak = ForEachTraits<function1_type>::NeedsBreak || ForEachTraits<function2_type>::NeedsBreak;
  static const bool hasBreak = has_deterministic_parallel_break<function1_type>::value;
  static const bool hasId = has_deterministic_id<function1_type>::value;
  static const bool useLocalState = has_deterministic_local_state<function1_type>::value;

  function1_type fn1;
  function2_type fn2;

  Options(const function1_type& fn1, const function2_type& fn2): fn1(fn1), fn2(fn2) { }
};

template<typename OptionsTy, bool Enable>
struct StateManagerBase {
  typedef typename OptionsTy::value_type value_type;
  typedef typename OptionsTy::function1_type function_type;
  void allocLocalState(UserContextAccess<value_type>&, function_type& self) { }
  void deallocLocalState(UserContextAccess<value_type>&) { }
  void saveLocalState(UserContextAccess<value_type>&, void*&) { }
  void restoreLocalState(UserContextAccess<value_type>&, void*) { } 
  void reuseItem(DItem<value_type>& item) { }

  template<typename LWL, typename GWL>
  typename GWL::value_type* emplaceContext(LWL& lwl, GWL& gwl, const DItem<value_type>& item) const {
    return gwl.emplace(item);
  }
  
  template<typename LWL, typename GWL>
  typename GWL::value_type* peekContext(LWL& lwl, GWL& gwl) const {
    return gwl.peek();
  }
  
  template<typename LWL, typename GWL>
  void popContext(LWL& lwl, GWL& gwl) const {
    gwl.pop_peeked();
  }
};

template<typename OptionsTy>
struct StateManagerBase<OptionsTy, true> {
  typedef typename OptionsTy::value_type value_type;
  typedef typename OptionsTy::function1_type function_type;
  typedef typename function_type::GaloisDeterministicLocalState LocalState;

  void allocLocalState(UserContextAccess<value_type>& c, function_type& self) {
    void *p = c.data().getPerIterAlloc().allocate(sizeof(LocalState));
    new (p) LocalState(self, c.data().getPerIterAlloc());
    c.setLocalState(p, false);
  }

  void deallocLocalState(UserContextAccess<value_type>& c) {
    bool dummy;
    LocalState *p = reinterpret_cast<LocalState*>(c.data().getLocalState(dummy));
    if (p)
      p->~LocalState();
  }

  void saveLocalState(UserContextAccess<value_type>& c, void*& localState) { 
    bool dummy;
    localState = c.data().getLocalState(dummy);
  }

  void restoreLocalState(UserContextAccess<value_type>& c, void* localState) { 
    c.setLocalState(localState, true);
  }

  template<typename LWL, typename GWL>
  typename LWL::value_type* emplaceContext(LWL& lwl, GWL& gwl, const DItem<value_type>& item) const {
    return lwl.emplace(item);
  }

  template<typename LWL, typename GWL>
  typename LWL::value_type* peekContext(LWL& lwl, GWL& gwl) const {
    return lwl.peek();
  }

  template<typename LWL, typename GWL>
  void popContext(LWL& lwl, GWL& gwl) const {
    lwl.pop_peeked();
  }

  void reuseItem(DItem<value_type>& item) { item.localState = NULL; }
};

template<typename OptionsTy>
struct StateManager: public StateManagerBase<OptionsTy, OptionsTy::useLocalState> { };

template<typename OptionsTy, bool Enable>
struct BreakManagerBase {
  bool checkBreak(typename OptionsTy::function1_type&) { return false; }
};

template<typename OptionsTy>
class BreakManagerBase<OptionsTy, true> {
  Barrier& barrier;
  LL::CacheLineStorage<volatile long> done;

public:
  BreakManagerBase(): barrier(getSystemBarrier()) { }

  bool checkBreak(typename OptionsTy::function1_type& fn) {
    if (LL::getTID() == 0)
      done.get() = fn.galoisDeterministicParallelBreak();
    barrier.wait();
    return done.get();
  }
};

template<typename OptionsTy>
struct BreakManager: public BreakManagerBase<OptionsTy, OptionsTy::hasBreak> { };

constexpr int ChunkSize = 32;
constexpr unsigned InitialNumRounds = 100;
constexpr int MinDelta = ChunkSize * 40;

class WindowManager {
public:
  class ThreadLocalData {
    friend class WindowManager;
    size_t window;
    size_t delta;
    size_t committed;
    size_t iterations;
    size_t aborted;

  public:
    size_t nextWindow() {
      window += delta;
      committed = iterations = aborted = 0;
      return window;
    }

    void incrementIterations() { ++iterations; }
    void incrementCommitted() { ++committed; }
  };

private:
  PerThreadStorage<ThreadLocalData> data;
  unsigned numActive;

public:
  WindowManager() {
    numActive = getActiveThreads(); 
  }

  ThreadLocalData& getLocalWindowManager() {
    return *data.getLocal();
  }

  size_t initialWindow(size_t dist, size_t atleast, size_t base = 0) {
    ThreadLocalData& local = *data.getLocal();
    size_t w = std::max(dist / InitialNumRounds, atleast) + base;
    return (local.window = local.delta = w);
  }

  void calculateWindow(bool inner) {
    ThreadLocalData& local = *data.getLocal();

    // Accumulate all threads' info
    size_t allcommitted = 0;
    size_t alliterations = 0;
    for (unsigned i = 0; i < numActive; ++i) {
      ThreadLocalData& r = *data.getRemote(i);
      allcommitted += r.committed;
      alliterations += r.iterations;
    }

    float commitRatio = alliterations > 0 ? allcommitted / (float) alliterations : 0.0;
    const float target = 0.95;

    if (commitRatio >= target)
      local.delta += local.delta;
    else if (allcommitted == 0) // special case when we don't execute anything
      local.delta += local.delta;
    else
      local.delta = commitRatio / target * local.delta;

    if (!inner) {
      local.delta = std::max(local.delta, (size_t) MinDelta);
    } else if (local.delta < (size_t) MinDelta) {
      // Try to get some new work instead of increasing window
      local.delta = 0;
    }

    // Useful debugging info
    if (false) {
      if (LL::getTID() == 0) {
        char buf[1024];
        snprintf(buf, 1024, "%d %.3f (%zu/%zu) window: %zu delta: %zu\n", 
            inner, commitRatio, allcommitted, alliterations, local.window, local.delta);
        LL::gPrint(buf);
      }
    }
  }
};

template<typename OptionsTy>
class NewWorkManager {
  typedef typename OptionsTy::value_type value_type;
  typedef DItem<value_type> Item;
  typedef DNewItem<value_type> NewItem;
  typedef std::vector<NewItem, typename PerIterAllocTy::rebind<NewItem>::other> NewItemsTy;
  typedef typename NewItemsTy::iterator NewItemsIterator;
  typedef FIFO<1024,Item> ReserveTy;
  typedef WorkList::dChunkedFIFO<ChunkSize,NewItem> NewWork;

  struct GetNewItem: public std::unary_function<int,NewItemsTy&> {
    NewWorkManager* self;
    GetNewItem(NewWorkManager* s = 0): self(s) { }
    NewItemsTy& operator()(int i) const { return self->data.getRemote(i)->newItems; }
  };
  
  typedef boost::transform_iterator<GetNewItem, boost::counting_iterator<int> > MergeOuterIt;
  typedef std::vector<NewItem, typename PerIterAllocTy::rebind<NewItem>::other> MergeBuf;
  typedef std::vector<value_type, typename PerIterAllocTy::rebind<value_type>::other> DistributeBuf;

  struct ThreadLocalData {
    IterAllocBaseTy heap;
    PerIterAllocTy alloc;
    NewItemsTy newItems;
    ReserveTy reserve;
    size_t minId;
    size_t maxId;
    size_t size;

    ThreadLocalData(): alloc(&heap), newItems(alloc) { }
  };

  const OptionsTy& options;
  IterAllocBaseTy heap;
  PerIterAllocTy alloc;
  PerThreadStorage<ThreadLocalData> data;
  NewWork new_;
  MergeBuf mergeBuf;
  DistributeBuf distributeBuf;
  Barrier& barrier;
  unsigned numActive;

  bool merge(int begin, int end) {
    if (begin == end)
      return false;
    else if (begin + 1 == end)
      return !data.getRemote(begin)->newItems.empty();
    
    bool retval = false;
    int mid = (end - begin) / 2 + begin;
    retval |= merge(begin, mid);
    retval |= merge(mid, end);

    GetNewItem fn(this);

    MergeOuterIt bbegin(boost::make_counting_iterator(begin), fn);
    MergeOuterIt mmid(boost::make_counting_iterator(mid), fn);
    MergeOuterIt eend(boost::make_counting_iterator(end), fn);
    auto aa = make_two_level_iterator<std::forward_iterator_tag, MergeOuterIt, typename NewItemsTy::iterator, GetBegin, GetEnd>(bbegin, mmid);
    auto bb = make_two_level_iterator<std::forward_iterator_tag, MergeOuterIt, typename NewItemsTy::iterator, GetBegin, GetEnd>(mmid, eend);
    auto cc = make_two_level_iterator<std::forward_iterator_tag, MergeOuterIt, typename NewItemsTy::iterator, GetBegin, GetEnd>(bbegin, eend);

    while (aa.first != aa.second && bb.first != bb.second) {
      if (*aa.first < *bb.first)
        mergeBuf.push_back(*aa.first++);
      else
        mergeBuf.push_back(*bb.first++);
    }

    for (; aa.first != aa.second; ++aa.first)
      mergeBuf.push_back(*aa.first);

    for (; bb.first != bb.second; ++bb.first)
      mergeBuf.push_back(*bb.first);

    for (NewItemsIterator ii = mergeBuf.begin(), ei = mergeBuf.end(); ii != ei; ++ii)
      *cc.first++ = *ii; 

    mergeBuf.clear();

    assert(cc.first == cc.second);

    return retval;
  }

  /**
   * Slightly complicated reindexing to separate out continuous elements in InputIterator.
   * <pre>
   * Example:
   *
   * blocksize: 2
   * pos:  0 1 2 3 4 5
   * item: A B C D E F
   * new:  A D B E C F
   * </pre>
   */
  template<typename InputIteratorTy>
  void redistribute(InputIteratorTy ii, InputIteratorTy ei, size_t dist, size_t window, unsigned tid) {
    ThreadLocalData& local = *data.getLocal();
    size_t blockSize = window;
    size_t numBlocks = dist / blockSize;
    
    size_t cur = 0;
    safe_advance(ii, tid, cur, dist);
    while (ii != ei) {
      unsigned long id;
      if (cur < blockSize * numBlocks)
        id = (cur % numBlocks) * blockSize + (cur / numBlocks);
      else
        id = cur;
      distributeBuf[id] = *ii;
      safe_advance(ii, numActive, cur, dist);
    }
  }

  template<typename InputIteratorTy,typename WL>
  void copyMine(InputIteratorTy ii, InputIteratorTy ei, size_t dist, WL* wl, size_t window, unsigned tid) {
    ThreadLocalData& local = *data.getLocal();
    size_t cur = 0;
    size_t k = 0;
    safe_advance(ii, tid, cur, dist);
    while (ii != ei) {
      unsigned long id = k * numActive + tid;
      if (id < window)
        wl->push(Item(*ii, id));
      else
        break;
      ++k;
      safe_advance(ii, numActive, cur, dist);
    }
    
    while (ii != ei) {
      unsigned long id = k * numActive + tid;
      local.reserve.push(Item(*ii, id));
      ++k;
      safe_advance(ii, numActive, cur, dist);
    }
  }

  template<typename InputIteratorTy,typename WL>
  void copyAllWithIds(InputIteratorTy ii, InputIteratorTy ei, WL* wl, size_t window) {
    ThreadLocalData& local = *data.getLocal();
    for (; ii != ei; ++ii) {
      unsigned long id = ii->parent;
      if (id < window)
        wl->push(Item(ii->val, id));
      else
        break;
    }

    for (; ii != ei; ++ii) {
      unsigned long id = ii->parent;
      local.reserve.push(Item(ii->val, id));
    }
  }

  template<typename InputIteratorTy,typename WL>
  void copyMineAfterRedistribute(InputIteratorTy ii, InputIteratorTy ei, size_t dist, WL* wl, size_t window, unsigned tid) {
    if (tid == 0) {
      distributeBuf.resize(dist);
    }
    barrier.wait();
    redistribute(ii, ei, dist, window, tid);
    barrier.wait();
    copyMine(distributeBuf.begin(), distributeBuf.end(), dist, wl, window, tid);
  }

  template<typename WL>
  void parallelSort(WindowManager& wm, WL* wl, unsigned tid) {
    ThreadLocalData& local = *data.getLocal();

    local.newItems.clear();
    Galois::optional<NewItem> p;
    while ((p = this->new_.pop())) {
      local.newItems.push_back(*p);
    }

    NewItemsIterator ii = local.newItems.begin();
    NewItemsIterator ei = local.newItems.end();
    std::sort(ii, ei);
    initialLimits(ii, ei);
    local.size = local.newItems.size();
    
    barrier.wait();

    if (tid == 0) {
      receiveLimits(local);
      broadcastLimits(local);
      if (!OptionsTy::hasId) {
        mergeBuf.reserve(local.size);
        merge(0, numActive);
      }
    }

    barrier.wait();

    if (OptionsTy::hasId) {
      size_t window = wm.initialWindow(local.maxId - local.minId, MinDelta, local.minId);
      copyAllWithIds(ii, ei, wl, window);
    } else {
      // initial window
      GetNewItem fn(this);
      MergeOuterIt bbegin(boost::make_counting_iterator(0), fn);
      MergeOuterIt eend(boost::make_counting_iterator((int) numActive), fn);
      auto ii = make_two_level_iterator<std::forward_iterator_tag, MergeOuterIt, typename NewItemsTy::iterator, GetBegin, GetEnd>(bbegin, eend);

      size_t window = wm.initialWindow(local.size, MinDelta);
      copyMineAfterRedistribute(boost::make_transform_iterator(ii.first, typename NewItem::GetValue()),
          boost::make_transform_iterator(ii.second, typename NewItem::GetValue()),
          local.size, wl, window, tid);
    }
  }

  void broadcastLimits(ThreadLocalData& local) {
    for (unsigned i = 1; i < numActive; ++i) {
      ThreadLocalData& other = *data.getRemote(i);
      other.minId = local.minId;
      other.maxId = local.maxId;
      other.size = local.size;
    }
  }

  void receiveLimits(ThreadLocalData& local) {
    for (unsigned i = 1; i < numActive; ++i) {
      ThreadLocalData& other = *data.getRemote(i);
      local.minId = std::min(other.minId, local.minId);
      local.maxId = std::max(other.maxId, local.maxId);
      local.size += other.size;
    }
  }

  //! Update min and max from sorted iterator
  template<typename BiIteratorTy>
  void initialLimits(BiIteratorTy ii, BiIteratorTy ei) {
    ThreadLocalData& local = *data.getLocal();

    local.minId = std::numeric_limits<size_t>::max();
    local.maxId = std::numeric_limits<size_t>::min();
    local.size = std::distance(ii, ei);

    if (ii != ei) {
      if (ii + 1 == ei) {
        local.minId = local.maxId = ii->parent;
      } else {
        local.minId = ii->parent;
        local.maxId = (ei-1)->parent;
      }
    }
  }

public:
  NewWorkManager(const OptionsTy& o): 
    options(o), alloc(&heap), mergeBuf(alloc), distributeBuf(alloc), barrier(getSystemBarrier()) 
  {
    numActive = getActiveThreads();
  }

  bool emptyReserve() { return data.getLocal()->reserve.empty(); }

  template<typename WL>
  void pushNextWindow(WL* wl, size_t window) {
    ThreadLocalData& local = *data.getLocal();
    Galois::optional<Item> p;
    while ((p = local.reserve.peek())) {
      if (p->id >= window)
        break;
      wl->push(*p);
      local.reserve.pop();
    }
  }

  void clearNewWork() { data.getLocal()->heap.clear(); }

  template<typename InputIteratorTy, bool HasId = OptionsTy::hasId>
  auto sortInitialWork(InputIteratorTy ii, InputIteratorTy ei) 
  -> typename std::enable_if<!HasId, void>::type 
  { 
  }

  template<typename InputIteratorTy, bool HasId = OptionsTy::hasId>
  auto sortInitialWork(InputIteratorTy ii, InputIteratorTy ei) 
  -> typename std::enable_if<HasId, void>::type
  { 
    ThreadLocalData& local = *data.getLocal();
    size_t dist = std::distance(ii, ei);

    mergeBuf.reserve(dist);
    for (; ii != ei; ++ii)
      mergeBuf.emplace_back(*ii, options.fn1.galoisDeterministicId(*ii), 1);

    ParallelSTL::sort(mergeBuf.begin(), mergeBuf.end());

    initialLimits(mergeBuf.begin(), mergeBuf.end());
    broadcastLimits(local);
  }

  template<typename InputIteratorTy, typename WL>
  void addInitialWork(WindowManager& wm, InputIteratorTy b, InputIteratorTy e, WL* wl) {
    size_t dist = std::distance(b, e);
    if (OptionsTy::hasId) {
      ThreadLocalData& local = *data.getLocal();
      size_t window = wm.initialWindow(dist, MinDelta, local.minId);
      copyMine(
          boost::make_transform_iterator(mergeBuf.begin(), typename NewItem::GetValue()),
          boost::make_transform_iterator(mergeBuf.end(), typename NewItem::GetValue()),
          mergeBuf.size(), wl, window, LL::getTID());
    } else {
      size_t window = wm.initialWindow(dist, MinDelta);
      copyMineAfterRedistribute(b, e, dist, wl, window, LL::getTID());
    }
  }

  template<bool HasId = OptionsTy::hasId>
  auto pushNew(const value_type& val, unsigned long parent, unsigned count) 
  -> typename std::enable_if<HasId, void>::type
  {
    new_.push(NewItem(val, options.fn1.galoisDeterministicId(val), 1));
  }

  template<bool HasId = OptionsTy::hasId>
  auto pushNew(const value_type& val, unsigned long parent, unsigned count) 
  -> typename std::enable_if<!HasId, void>::type
  {
    new_.push(NewItem(val, parent, count));
  }

  template<typename WL>
  void distributeNewWork(WindowManager& wm, WL* wl) {
    parallelSort(wm, wl, LL::getTID());
  }
};

template<typename OptionsTy>
class Executor: public BreakManager<OptionsTy>, StateManager<OptionsTy>, NewWorkManager<OptionsTy>, WindowManager {
  typedef typename OptionsTy::value_type value_type;
  typedef DItem<value_type> Item;
  typedef DeterministicContext<value_type> Context;

  typedef WorkList::dChunkedFIFO<ChunkSize,Item> WL;
  typedef WorkList::dChunkedFIFO<ChunkSize,Context> PendingWork;
  typedef WorkList::ChunkedFIFO<ChunkSize,Context,false> LocalPendingWork;

  // Truly thread-local
  struct ThreadLocalData: private boost::noncopyable {
    OptionsTy options;
    LocalPendingWork localPending;
    UserContextAccess<value_type> facing;
    LoopStatistics<OptionsTy::needsStats> stat;
    WL* wlcur;
    WL* wlnext;
    size_t rounds;
    size_t outerRounds;
    bool hasNewWork;
    ThreadLocalData(const OptionsTy& o, const char* loopname): options(o), stat(loopname), rounds(0), outerRounds(0) { }
  };

  const OptionsTy& options;
  Barrier& barrier;
  const char* loopname;
  WL worklists[2];
  PendingWork pending;
  LL::CacheLineStorage<volatile long> innerDone;
  LL::CacheLineStorage<volatile long> outerDone;
  LL::CacheLineStorage<volatile long> hasNewWork;

  bool pendingLoop(ThreadLocalData& tld);
  bool commitLoop(ThreadLocalData& tld);
  void go();

public:
  Executor(const OptionsTy& o, const char* ln):
    NewWorkManager<OptionsTy>(o), options(o), barrier(getSystemBarrier()), loopname(ln) 
  { 
    static_assert(!OptionsTy::needsBreak || OptionsTy::hasBreak,
        "need to use break function to break loop");
  }

  template<typename RangeTy>
  void AddInitialWork(RangeTy range) {
    this->addInitialWork(*this, range.begin(), range.end(), &worklists[1]);
  }

  template<typename IterTy>
  void preprocess(IterTy ii, IterTy ei) {
    this->sortInitialWork(ii, ei);
  }

  void operator()() {
    go();
  }
};

template<typename OptionsTy>
void Executor<OptionsTy>::go() {
  ThreadLocalData tld(options, loopname);
  auto& local = this->getLocalWindowManager();
  tld.wlcur = &worklists[0];
  tld.wlnext = &worklists[1];

  tld.hasNewWork = false;

  while (true) {
    ++tld.outerRounds;

    while (true) {
      ++tld.rounds;

      std::swap(tld.wlcur, tld.wlnext);
      setPending(PENDING);
      bool nextPending = pendingLoop(tld);
      innerDone.get() = true;

      barrier.wait();

      setPending(COMMITTING);
      bool nextCommit = commitLoop(tld);
      outerDone.get() = true;
      if (nextPending || nextCommit)
        innerDone.get() = false;

      barrier.wait();

      if (innerDone.get())
        break;

      this->calculateWindow(true);

      barrier.wait();

      this->pushNextWindow(tld.wlnext, local.nextWindow());
    }

    if (!this->emptyReserve())
      outerDone.get() = false;

    if (tld.hasNewWork)
      hasNewWork.get() = true;

    if (this->checkBreak(tld.options.fn1))
      break;

    this->calculateWindow(false);

    barrier.wait();

    if (outerDone.get()) {
      if (!OptionsTy::needsPush)
        break;
      if (!hasNewWork.get()) // (1)
        break;
      this->distributeNewWork(*this, tld.wlnext);
      tld.hasNewWork = false;
      // NB: assumes that distributeNewWork has a barrier otherwise checking at (1) is erroneous
      hasNewWork.get() = false;
    } else {
      this->pushNextWindow(tld.wlnext, local.nextWindow());
    }
  }

  setPending(NON_DET);

  this->clearNewWork();

  if (OptionsTy::needsStats) {
    if (LL::getTID() == 0) {
      reportStat(loopname, "RoundsExecuted", tld.rounds);
      reportStat(loopname, "OuterRoundsExecuted", tld.outerRounds);
    }
  }
}

template<typename OptionsTy>
bool Executor<OptionsTy>::pendingLoop(ThreadLocalData& tld)
{
  auto& local = this->getLocalWindowManager();
  bool retval = false;
  Galois::optional<Item> p;
  while ((p = tld.wlcur->pop())) {
    // Use a new context for each item because there is a race when reusing
    // between aborted iterations.

    Context* ctx = this->emplaceContext(tld.localPending, pending, *p);
    assert(ctx != NULL);

    local.incrementIterations();
    bool commit = true;

    ctx->startIteration();
    tld.stat.inc_iterations();
    setThreadContext(ctx);

    this->allocLocalState(tld.facing, tld.options.fn1);
    int result = 0;
#ifdef GALOIS_USE_LONGJMP
    if ((result = setjmp(hackjmp)) == 0) {
#else
    try {
#endif
      tld.options.fn1(ctx->item.val, tld.facing.data());
#ifdef GALOIS_USE_LONGJMP
    } else { clearConflictLock(); }
#else
    } catch (const ConflictFlag& flag) { clearConflictLock(); result = flag; }
#endif
    clearReleasable();
    switch (result) {
      case 0: 
      case REACHED_FAILSAFE: break;
      case CONFLICT: commit = false; break;
      default: assert(0 && "Unknown conflict flag"); abort(); break;
    }

    if (ForEachTraits<typename OptionsTy::function1_type>::NeedsPIA && !OptionsTy::useLocalState)
      tld.facing.resetAlloc();

    if (commit) { 
      this->saveLocalState(tld.facing, ctx->item.localState);
    } else {
      retval = true;
    }
  }

  return retval;
}

template<typename OptionsTy>
bool Executor<OptionsTy>::commitLoop(ThreadLocalData& tld) 
{
  bool retval = false;
  auto& local = this->getLocalWindowManager();

  size_t ncommits = 0;
  size_t niter = 0;

  Context* ctx;
  while ((ctx = this->peekContext(tld.localPending, pending))) {
    ++niter;
    bool commit = true;
    // Can skip this check in prefix by repeating computations but eagerly
    // aborting seems more efficient
    if (ctx->not_ready)
      commit = false;

    setThreadContext(ctx);
    if (commit) {
      this->restoreLocalState(tld.facing, ctx->item.localState);
      int result = 0;
#ifdef GALOIS_USE_LONGJMP
      if ((result = setjmp(hackjmp)) == 0) {
#else
      try {
#endif
        tld.options.fn2(ctx->item.val, tld.facing.data());
#ifdef GALOIS_USE_LONGJMP
      } else { clearConflictLock(); }
#else
      } catch (const ConflictFlag& flag) { clearConflictLock(); result = flag; }
#endif
      clearReleasable();
      switch (result) {
        case 0: break;
        case CONFLICT: commit = false; break;
        default: assert(0 && "Unknown conflict flag"); abort(); break;
      }
    }

    this->deallocLocalState(tld.facing);
    
    if (commit) {
      ++ncommits;
      local.incrementCommitted();
      if (OptionsTy::needsPush) {
        unsigned long parent = ctx->item.id;
        typedef typename UserContextAccess<value_type>::PushBufferTy::iterator iterator;
        unsigned count = 0;
        for (auto& item : tld.facing.getPushBuffer()) {
          this->pushNew(item, parent, ++count);
          if (count == 0) {
            assert(0 && "Counter overflow");
            abort();
          }
        }
        if (count)
          tld.hasNewWork = true;
      }
      assert(OptionsTy::needsPush
          || tld.facing.getPushBuffer().begin() == tld.facing.getPushBuffer().end());
    } else {
      this->reuseItem(ctx->item);
      tld.wlnext->push(ctx->item);
      tld.stat.inc_conflicts();
      retval = true;
    }

    if (commit) {
      ctx->commitIteration();
    } else {
      ctx->cancelIteration();
    }

    if (ForEachTraits<typename OptionsTy::function2_type>::NeedsPIA && !OptionsTy::useLocalState)
      tld.facing.resetAlloc();

    tld.facing.resetPushBuffer();
    this->popContext(tld.localPending, pending);
  }

  if (ForEachTraits<typename OptionsTy::function2_type>::NeedsPIA && OptionsTy::useLocalState)
    tld.facing.resetAlloc();

  setThreadContext(0);

  return retval;
}

} // end namespace anonymous
} // end namespace DeterministicImpl

template<typename RangeTy, typename WorkTy>
static inline void for_each_det_impl(const RangeTy& range, WorkTy& W) {
  W.preprocess(range.begin(), range.end());

  assert(!inGaloisForEach);

  inGaloisForEach = true;
  RunCommand init(std::bind(&WorkTy::template AddInitialWork<RangeTy>, std::ref(W), std::ref(range)));
  getSystemThreadPool().run(activeThreads, std::ref(init), std::ref(getSystemBarrier()), std::ref(W));
  inGaloisForEach = false;
}


} // end namespace Runtime
} // end namespace Galois

namespace Galois {

/**
 * Deterministic execution with prefix operator.
 * The prefix of the operator should be exactly the same as the operator
 * but with execution returning at the failsafe point. The operator
 * should conform to a standard Galois unordered set operator {@link for_each()}.
 *
 * @param b begining of range of initial items
 * @param e end of range of initial items
 * @param prefix prefix of operator
 * @param fn operator
 * @param loopname string to identify loop in statistics output
 */
template<typename IterTy, typename Function1Ty, typename Function2Ty>
static inline void for_each_det(IterTy b, IterTy e, Function1Ty prefix, Function2Ty fn, const char* loopname = 0) {
  typedef Runtime::StandardRange<IterTy> Range;
  typedef typename Range::value_type T;
  typedef Runtime::DeterministicImpl::Options<T,Function1Ty,Function2Ty> OptionsTy;
  typedef Runtime::DeterministicImpl::Executor<OptionsTy> WorkTy;

  OptionsTy options(prefix, fn);
  WorkTy W(options, loopname);
  Runtime::for_each_det_impl(Runtime::makeStandardRange(b, e), W);
}

/**
 * Deterministic execution with prefix operator.
 * The prefix of the operator should be exactly the same as the operator
 * but with execution returning at the failsafe point. The operator
 * should conform to a standard Galois unordered set operator {@link for_each()}.
 *
 * @param i initial item
 * @param prefix prefix of operator
 * @param fn operator
 * @param loopname string to identify loop in statistics output
 */
template<typename T, typename Function1Ty, typename Function2Ty>
static inline void for_each_det(T i, Function1Ty prefix, Function2Ty fn, const char* loopname = 0) {
  T wl[1] = { i };
  for_each_det(&wl[0], &wl[1], prefix, fn, loopname);
}

/**
 * Deterministic execution with single operator.
 * The operator fn is used both for the prefix computation and for the
 * continuation of computation, c.f., the prefix operator version which
 * uses two different functions. The operator can distinguish between
 * the two uses by querying {@link UserContext.getLocalState()}.
 *
 * @param b begining of range of initial items
 * @param e end of range of initial items
 * @param fn operator
 * @param loopname string to identify loop in statistics output
 */
template<typename IterTy, typename FunctionTy>
static inline void for_each_det(IterTy b, IterTy e, FunctionTy fn, const char* loopname = 0) {
  for_each_det(b, e, fn, fn, loopname);
}

/**
 * Deterministic execution with single operator.
 * The operator fn is used both for the prefix computation and for the
 * continuation of computation, c.f., the prefix operator version which
 * uses two different functions. The operator can distinguish between
 * the two uses by querying {@link UserContext.getLocalState()}.
 *
 * @param i initial item
 * @param fn operator
 * @param loopname string to identify loop in statistics output
 */
template<typename T, typename FunctionTy>
static inline void for_each_det(T i, FunctionTy fn, const char* loopname = 0) {
  T wl[1] = { i };
  for_each_det(&wl[0], &wl[1], fn, fn, loopname);
}

} // end namespace Galois

#endif
