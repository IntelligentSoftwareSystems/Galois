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

//#ifdef GALOIS_USE_HTM
//#include "HTMDeterministicWork.h"
//#else

#include "Galois/config.h"
#include "Galois/Threads.h"

#include "Galois/ParallelSTL/ParallelSTL.h"
#include "Galois/TwoLevelIterator.h"
#include "Galois/Runtime/ll/gio.h"
#include "Galois/Runtime/ll/EnvCheck.h"

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

template<typename T, typename OptionsTy>
class DeterministicContext:
    public SimpleRuntimeContext,
    public StrictObject<typename boost::mpl::if_c<OptionsTy::useInOrderCommit, unsigned long, void>::type> 
{
  template<bool useInOrderCommit = OptionsTy::useInOrderCommit>
  void updateAborted(unsigned long id, typename std::enable_if<useInOrderCommit>::type* = 0) {
    this->get() = std::min(this->get(), id);
  }

  template<bool useInOrderCommit = OptionsTy::useInOrderCommit>
  void updateAborted(unsigned long id, typename std::enable_if<!useInOrderCommit>::type* = 0) { }

public:  
  DItem<T> item;
  bool not_ready;

  DeterministicContext(const DItem<T>& _item): 
      SimpleRuntimeContext(true), 
      item(_item),
      not_ready(false)
  { }

  bool notReady() const { 
    return not_ready;
  }

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
          updateAborted(this->item.id);
          return; 
        }
      }
    } while (!this->stealByCAS(lockable, other));

    // Disable loser
    if (other) {
      other->not_ready = true; // Only need atomic write
      updateAborted(other->item.id);
    }

    return;
  }
};

namespace {

template<typename T, typename Function1Ty, typename Function2Ty>
struct Options {
  typedef T value_type;
  typedef Function1Ty function1_type;
  typedef Function2Ty function2_type;

  static const bool needsStats = ForEachTraits<Function1Ty>::NeedsStats || ForEachTraits<Function2Ty>::NeedsStats;
  static const bool needsPush = ForEachTraits<Function1Ty>::NeedsPush || ForEachTraits<Function2Ty>::NeedsPush;
  static const bool needsBreak = ForEachTraits<Function1Ty>::NeedsBreak || ForEachTraits<Function2Ty>::NeedsBreak;
  static const bool hasBreak = has_deterministic_parallel_break<Function1Ty>::value;
  static const bool hasId = has_deterministic_id<Function1Ty>::value;
  static const bool useLocalState = has_deterministic_local_state<Function1Ty>::value;
#ifdef GALOIS_USE_DET_FIXED_WINDOW
  static const bool hasFixedWindow = true;
#else
  static const bool hasFixedWindow = false;
#endif
#ifdef GALOIS_USE_DET_INORDER
  static const bool useInOrderCommit = true;
#else
  static const bool useInOrderCommit = false;
#endif

  Function1Ty fn1;
  Function2Ty fn2;
  int defaultDelta;

  Options(const Function1Ty& fn1, const Function2Ty& fn2): fn1(fn1), fn2(fn2) { 
    if (!LL::EnvCheck("GALOIS_FIXED_DET_WINDOW_SIZE", defaultDelta))
      defaultDelta = 256;
  }
};

template<typename OptionsTy, bool Enable>
struct StateManagerBase {
  typedef typename OptionsTy::value_type value_type;
  typedef typename OptionsTy::function1_type function_type;
  void alloc(UserContextAccess<value_type>&, function_type& self) { }
  void dealloc(UserContextAccess<value_type>&) { }
  void save(UserContextAccess<value_type>&, void*&) { }
  void restore(UserContextAccess<value_type>&, void*) { } 
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

  void alloc(UserContextAccess<value_type>& c, function_type& self) {
    void *p = c.data().getPerIterAlloc().allocate(sizeof(LocalState));
    new (p) LocalState(self, c.data().getPerIterAlloc());
    c.setLocalState(p, false);
  }

  void dealloc(UserContextAccess<value_type>& c) {
    bool dummy;
    LocalState *p = reinterpret_cast<LocalState*>(c.data().getLocalState(dummy));
    if (p)
      p->~LocalState();
  }

  void save(UserContextAccess<value_type>& c, void*& localState) { 
    bool dummy;
    localState = c.data().getLocalState(dummy);
  }

  void restore(UserContextAccess<value_type>& c, void* localState) { 
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

template<typename OptionsTy, bool Enable>
struct NewWorkManagerBase {
  static const bool value = false;
  static const int ChunkSize = 32;
  static const int MinDelta = ChunkSize * 40;

  template<typename Arg>
  static uintptr_t id(const typename OptionsTy::function1_type& fn, Arg arg) { return 0; }

  template<typename WL, typename T>
  void pushNew(const OptionsTy& options, WL& wl, const T& val, unsigned long parent, unsigned count) const {
    typedef typename WL::value_type value_type;
    wl.push(value_type(val, parent, count));
  }
};

template<typename OptionsTy>
struct NewWorkManagerBase<OptionsTy, true> {
  static const bool value = true;
  static const int ChunkSize = 32;
  static const int MinDelta = ChunkSize * 40;

  template<typename Arg>
  static uintptr_t id(const typename OptionsTy::function1_type& fn, Arg arg) {
    return fn.galoisDeterministicId(std::forward<Arg>(arg));
  }

  template<typename WL, typename T>
  void pushNew(const OptionsTy& options, WL& wl, const T& val, unsigned long parent, unsigned count) const {
    typedef typename WL::value_type value_type;
    wl.push(value_type(val, id(options.fn1, val), 1));
  }
};

template<typename OptionsTy>
struct NewWorkManager: public NewWorkManagerBase<OptionsTy, OptionsTy::hasId> { };

template<typename OptionsTy, bool Enable>
struct InOrderManagerBase {
  typedef DeterministicContext<typename OptionsTy::value_type, OptionsTy> Context;

  void initializeContext(Context* ctx) { }
  void updateAborted(const Context* ctx) { }
  void resetAborted() { }
  bool shouldCommit(const Context* ctx) { return true; }
  void allReduceAborted() { }
};

template<typename OptionsTy>
struct InOrderManagerBase<OptionsTy, true> {
  typedef DeterministicContext<typename OptionsTy::value_type, OptionsTy> Context;
  
  PerThreadStorage<unsigned long> data;
  Barrier& barrier;
  InOrderManagerBase(): barrier(getSystemBarrier()) { }

  void initializeContext(Context* ctx) { ctx->get() = std::numeric_limits<unsigned long>::max(); }

  void updateAborted(const Context* ctx) { 
    unsigned long& r = *data.getLocal();
    r = std::min(r, ctx->get());
  }
  void resetAborted() { *data.getLocal() = std::numeric_limits<unsigned long>::max(); }
  bool shouldCommit(const Context* ctx) { 
    return ctx->item.id < *data.getLocal();
  }
  void allReduceAborted() {
    unsigned long r = std::numeric_limits<unsigned long>::max();
    for (unsigned i = 0; i < activeThreads; ++i)
      r = std::min(r, *data.getRemote(i));
    barrier.wait();
    *data.getLocal() = r;
  }
};

template<typename OptionsTy>
struct InOrderManager: public InOrderManagerBase<OptionsTy, OptionsTy::useInOrderCommit> { };

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

  struct GetFirst: public std::unary_function<DNewItem<T>,const T&> {
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
template<int chunksize,typename T>
struct FIFO {
  WorkList::ChunkedFIFO<chunksize,T,false> m_data;
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

template<typename> class DMergeManagerBase;
template<typename> class DMergeManager;

//! Thread-local data for merging
template<typename OptionsTy>
class DMergeLocal: private boost::noncopyable {
  template<typename> friend class DMergeManagerBase;
  template<typename> friend class DMergeManager;

  typedef typename OptionsTy::value_type value_type;

  typedef DItem<value_type> Item;
  typedef DNewItem<value_type> NewItem;
  typedef std::vector<NewItem, typename PerIterAllocTy::rebind<NewItem>::other> NewItemsTy;
  typedef FIFO<1024,Item> ReserveTy;

  IterAllocBaseTy heap;
  PerIterAllocTy alloc;
  ReserveTy reserve;
  size_t window;
  size_t delta;
  size_t committed;
  size_t iterations;
  size_t aborted;
  // For id based execution
  size_t minId;
  size_t maxId;
  // For general execution
  size_t size;

public:
  NewItemsTy newItems;

  DMergeLocal(): alloc(&heap), newItems(alloc) { 
    resetStats(); 
  }

private:
  //! Update min and max from sorted iterator
  template<typename BiIteratorTy>
  void initialLimits(BiIteratorTy ii, BiIteratorTy ei) {
    minId = std::numeric_limits<size_t>::max();
    maxId = std::numeric_limits<size_t>::min();
    size = std::distance(ii, ei);

    if (ii != ei) {
      if (ii + 1 == ei) {
        minId = maxId = ii->parent;
      } else {
        minId = ii->parent;
        maxId = (ei-1)->parent;
      }
    }
  }

  template<typename InputIteratorTy,typename WL>
  void copyIn(InputIteratorTy ii, InputIteratorTy ei, size_t dist, WL* wl, unsigned numActive) {
    unsigned int tid = LL::getTID();
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
      reserve.push(Item(*ii, id));
      ++k;
      safe_advance(ii, numActive, cur, dist);
    }
  }

  size_t fixedWindowSize(const OptionsTy& options) {
    if (!OptionsTy::hasFixedWindow) return 0;
    if (OptionsTy::hasId)
      return options.defaultDelta + minId;
    else
      return options.defaultDelta;
  }
  void initialWindow(const OptionsTy& options, size_t dist) {
    size_t w = fixedWindowSize(options);
    if (!w) {
      if (OptionsTy::hasId)
        w = std::max((maxId - minId) / 100, (size_t) NewWorkManager<OptionsTy>::MinDelta) + minId;
      else
        w = std::max(dist / 100, (size_t) NewWorkManager<OptionsTy>::MinDelta);
    }
    window = delta = w;
  }

  void receiveLimits(const DMergeLocal<OptionsTy>& other) {
    minId = other.minId;
    maxId = other.maxId;
    size = other.size;
  }

  void reduceLimits(const DMergeLocal<OptionsTy>& other) {
    minId = std::min(other.minId, minId);
    maxId = std::max(other.maxId, maxId);
    size += other.size;
  }

public:
  void clear() { heap.clear(); }
  void incrementIterations() { ++iterations; }
  void incrementCommitted() { ++committed; }
  void resetStats() { committed = iterations = aborted = 0; }
  bool emptyReserve() { return reserve.empty(); }

  template<typename WL>
  void nextWindow(WL* wl) {
    window += delta;
    Galois::optional<Item> p;
    while ((p = reserve.peek())) {
      if (p->id >= window)
        break;
      wl->push(*p);
      reserve.pop();
    }
  }
};

template<typename OptionsTy>
class DMergeManagerBase {
protected:
  typedef typename OptionsTy::value_type value_type;
  typedef DItem<value_type> Item;
  typedef DNewItem<value_type> NewItem;
  typedef WorkList::dChunkedFIFO<NewWorkManager<OptionsTy>::ChunkSize,NewItem> NewWork;
  typedef DMergeLocal<OptionsTy> MergeLocal;
  typedef typename MergeLocal::NewItemsTy NewItemsTy;
  typedef typename NewItemsTy::iterator NewItemsIterator;

  IterAllocBaseTy heap;
  PerIterAllocTy alloc;
  PerThreadStorage<MergeLocal> data;

  NewWork new_;
  unsigned numActive;

  void broadcastLimits(MergeLocal& mlocal, unsigned int tid) {
    for (unsigned i = 0; i < this->numActive; ++i) {
      if (i == tid) continue;
      MergeLocal& mother = *this->data.getRemote(i);
      mother.receiveLimits(mlocal);
    }
  }

  void reduceLimits(MergeLocal& mlocal, unsigned int tid) {
    for (unsigned i = 0; i < this->numActive; ++i) {
      if (i == tid) continue;
      MergeLocal& mother = *this->data.getRemote(i);
      mlocal.reduceLimits(mother);
    }
  }

public:
  DMergeManagerBase(): alloc(&heap) {
    numActive = getActiveThreads();
  }

  MergeLocal& get() {
    return *data.getLocal();
  }

  void calculateWindow(const OptionsTy& options, bool inner) {
    MergeLocal& mlocal = *data.getLocal();

    // Accumulate all threads' info
    size_t allcommitted = 0;
    size_t alliterations = 0;
    for (unsigned i = 0; i < numActive; ++i) {
      MergeLocal& mlocal = *data.getRemote(i);
      allcommitted += mlocal.committed;
      alliterations += mlocal.iterations;
    }

    float commitRatio = alliterations > 0 ? allcommitted / (float) alliterations : 0.0;
    if (OptionsTy::hasFixedWindow) {
      if (!inner || allcommitted == alliterations) {
        mlocal.delta = mlocal.fixedWindowSize(options);
      } else {
        mlocal.delta = 0;
      }
    } else {
      const float target = 0.95;

      if (commitRatio >= target)
        mlocal.delta += mlocal.delta;
      else if (allcommitted == 0) // special case when we don't execute anything
        mlocal.delta += mlocal.delta;
      else
        mlocal.delta = commitRatio / target * mlocal.delta;

      if (!inner) {
        mlocal.delta = std::max(mlocal.delta, (size_t) NewWorkManager<OptionsTy>::MinDelta);
      } else if (mlocal.delta < (size_t) NewWorkManager<OptionsTy>::MinDelta) {
        // Try to get some new work instead of increasing window
        mlocal.delta = 0;
      }
    }

    // Useful debugging info
    if (false) {
      if (LL::getTID() == 0) {
        char buf[1024];
        snprintf(buf, 1024, "%d %.3f (%zu/%zu) window: %zu delta: %zu\n", 
            inner, commitRatio, allcommitted, alliterations, mlocal.window, mlocal.delta);
        LL::gPrint(buf);
      }
    }
  }
};

//! Default implementation for merging
template<typename OptionsTy>
class DMergeManager: public DMergeManagerBase<OptionsTy> {
  typedef DMergeManagerBase<OptionsTy> Base;
  typedef typename Base::value_type value_type;
  typedef typename Base::Item Item;
  typedef typename Base::NewItem NewItem;
  typedef typename Base::MergeLocal MergeLocal;
  typedef typename Base::NewItemsTy NewItemsTy;
  typedef typename Base::NewItemsIterator NewItemsIterator;

  struct GetNewItem: public std::unary_function<int,NewItemsTy&> {
    PerThreadStorage<MergeLocal>* base;
    GetNewItem() { }
    GetNewItem(PerThreadStorage<MergeLocal>* b): base(b) { }
    NewItemsTy& operator()(int i) const { return base->getRemote(i)->newItems; }
  };

  typedef boost::transform_iterator<GetNewItem, boost::counting_iterator<int> > MergeOuterIt;
  typedef typename ChooseStlTwoLevelIterator<MergeOuterIt, typename NewItemsTy::iterator>::type MergeIt;
  typedef std::vector<NewItem, typename PerIterAllocTy::rebind<NewItem>::other> MergeBuf;
  typedef std::vector<value_type, typename PerIterAllocTy::rebind<value_type>::other> DistributeBuf;

  const OptionsTy& options;
  MergeBuf mergeBuf;
  DistributeBuf distributeBuf;
  NewWorkManager<OptionsTy> newWorkManager;
  Barrier& barrier;

  bool merge(int begin, int end) {
    if (begin == end)
      return false;
    else if (begin + 1 == end)
      return !this->data.getRemote(begin)->newItems.empty();
    
    bool retval = false;
    int mid = (end - begin) / 2 + begin;
    retval |= merge(begin, mid);
    retval |= merge(mid, end);

    MergeOuterIt bbegin(boost::make_counting_iterator(begin), GetNewItem(&this->data));
    MergeOuterIt mmid(boost::make_counting_iterator(mid), GetNewItem(&this->data));
    MergeOuterIt eend(boost::make_counting_iterator(end), GetNewItem(&this->data));
    // MergeIt aa(bbegin, mmid), ea(mmid, mmid);
    // MergeIt bb(mmid, eend), eb(eend, eend);
    // MergeIt cc(bbegin, eend), ec(eend, eend);
    MergeIt aa = stl_two_level_begin(bbegin, mmid);
    MergeIt ea = stl_two_level_end(bbegin, mmid);
    MergeIt bb = stl_two_level_begin(mmid, eend);
    MergeIt eb = stl_two_level_end(mmid, eend);
    MergeIt cc = stl_two_level_begin(bbegin, eend);
    MergeIt ec = stl_two_level_end(bbegin, eend);

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
  void redistribute(InputIteratorTy ii, InputIteratorTy ei, size_t dist) {
    unsigned int tid = LL::getTID();
    //const size_t numBlocks = 1 << 7;
    //const size_t mask = numBlocks - 1;
    //size_t blockSize = dist / numBlocks; // round down
    MergeLocal& mlocal = *this->data.getLocal();
    //size_t blockSize = std::max((size_t) (0.9*minfo.delta), (size_t) 1);
    size_t blockSize = mlocal.delta;
    size_t numBlocks = dist / blockSize;
    
    size_t cur = 0;
    safe_advance(ii, tid, cur, dist);
    while (ii != ei) {
      unsigned long id;
      if (cur < blockSize * numBlocks)
        //id = (cur & mask) * blockSize + (cur / numBlocks);
        id = (cur % numBlocks) * blockSize + (cur / numBlocks);
      else
        id = cur;
      distributeBuf[id] = *ii;
      safe_advance(ii, this->numActive, cur, dist);
    }
  }

  template<typename InputIteratorTy,typename WL>
  void distribute(InputIteratorTy ii, InputIteratorTy ei, size_t dist, WL* wl) {
    unsigned int tid = LL::getTID();
    MergeLocal& mlocal = *this->data.getLocal();
    mlocal.initialWindow(options, dist);
    if (true) {
      // Renumber to avoid pathological cases
      if (tid == 0) {
        distributeBuf.resize(dist);
      }
      barrier.wait();
      redistribute(ii, ei, dist);
      barrier.wait();
      mlocal.copyIn(distributeBuf.begin(), distributeBuf.end(), dist, wl, this->numActive);
    } else {
      mlocal.copyIn(ii, ei, dist, wl, this->numActive);
    }
  }

  template<typename WL>
  void parallelSort(WL* wl) {
    MergeLocal& mlocal = *this->data.getLocal();

    mlocal.newItems.clear();
    Galois::optional<NewItem> p;
    while ((p = this->new_.pop())) {
      mlocal.newItems.push_back(*p);
    }

    std::sort(mlocal.newItems.begin(), mlocal.newItems.end());
    mlocal.initialLimits(mlocal.newItems.begin(), mlocal.newItems.end());
    
    barrier.wait();

    unsigned tid = LL::getTID();
    if (tid == 0) {
      this->reduceLimits(mlocal, tid);
      mergeBuf.reserve(mlocal.size);
      this->broadcastLimits(mlocal, tid);
      merge(0, this->numActive);
    }

    barrier.wait();

    MergeOuterIt bbegin(boost::make_counting_iterator(0), GetNewItem(&this->data));
    MergeOuterIt eend(boost::make_counting_iterator((int) this->numActive), GetNewItem(&this->data));
    MergeIt ii = stl_two_level_begin(bbegin, eend);
    MergeIt ei = stl_two_level_end(eend, eend);

    distribute(boost::make_transform_iterator(ii, typename NewItem::GetFirst()),
        boost::make_transform_iterator(ei, typename NewItem::GetFirst()),
        mlocal.size, wl);
  }

public:
  DMergeManager(const OptionsTy& o):
    options(o), mergeBuf(this->alloc), distributeBuf(this->alloc), barrier(getSystemBarrier()) 
  { }

  template<typename InputIteratorTy>
  void presort(const OptionsTy& options, InputIteratorTy ii, InputIteratorTy ei) { 
    if (!OptionsTy::hasId)
      return;

    size_t dist = std::distance(ii, ei);
    mergeBuf.reserve(dist);
    for (; ii != ei; ++ii)
      mergeBuf.push_back(NewItem(*ii, newWorkManager.id(options.fn1, *ii), 1));
    ParallelSTL::sort(mergeBuf.begin(), mergeBuf.end());

    MergeLocal& mlocal = *this->data.getLocal();
    this->broadcastLimits(mlocal, 0);
  }

  template<typename InputIteratorTy, typename WL>
  void addInitialWork(InputIteratorTy b, InputIteratorTy e, WL* wl) {
    if (OptionsTy::hasId) {
      distribute(
          boost::make_transform_iterator(mergeBuf.begin(), typename NewItem::GetFirst()),
          boost::make_transform_iterator(mergeBuf.end(), typename NewItem::GetFirst()),
          std::distance(mergeBuf.begin(), mergeBuf.end()), wl);
      mergeBuf.clear();
    } else {
      distribute(b, e, std::distance(b, e), wl);
    }
  }

  template<typename WL>
  void pushNew(const OptionsTy& options, const value_type& val, unsigned long parent, unsigned count,
      WL* wl, bool& hasNewWork) {
    newWorkManager.pushNew(options, this->new_, val, parent, count);
    hasNewWork = true;
  }

  template<typename WL>
  bool distributeNewWork(WL* wl) {
    parallelSort(wl);
    return false;
  }
};

template<typename OptionsTy>
class Executor {
  typedef typename OptionsTy::value_type value_type;
  typedef DItem<value_type> Item;
  typedef DNewItem<value_type> NewItem;
  typedef DMergeManager<OptionsTy> MergeManager;
  typedef DMergeLocal<OptionsTy> MergeLocal;
  typedef DeterministicContext<value_type, OptionsTy> Context;

  typedef WorkList::dChunkedFIFO<NewWorkManager<OptionsTy>::ChunkSize,Item> WL;
  typedef WorkList::dChunkedFIFO<NewWorkManager<OptionsTy>::ChunkSize,Context> PendingWork;
  typedef WorkList::ChunkedFIFO<NewWorkManager<OptionsTy>::ChunkSize,Context,false> LocalPendingWork;

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

  PendingWork pending;
  MergeManager mergeManager;
  BreakManager<OptionsTy> breakManager;
  StateManager<OptionsTy> stateManager;
  InOrderManager<OptionsTy> inOrderManager;
  WL worklists[2];
  const OptionsTy& options;
  Barrier& barrier;
  const char* loopname;
  LL::CacheLineStorage<volatile long> innerDone;
  LL::CacheLineStorage<volatile long> outerDone;
  LL::CacheLineStorage<volatile long> hasNewWork;
  int numActive;

  bool pendingLoop(ThreadLocalData& tld);
  bool commitLoop(ThreadLocalData& tld);
  void go();

public:
  Executor(const OptionsTy& o, const char* ln):
    mergeManager(o), options(o), barrier(getSystemBarrier()), loopname(ln) 
  { 
    static_assert(!OptionsTy::needsBreak || OptionsTy::hasBreak,
        "need to use break function to break loop");
  }

  template<typename RangeTy>
  void AddInitialWork(RangeTy range) {
    mergeManager.addInitialWork(range.begin(), range.end(), &worklists[1]);
  }

  template<typename IterTy>
  void presort(IterTy ii, IterTy ei) {
    mergeManager.presort(options, ii, ei);
  }

  void operator()() {
    go();
  }
};

template<typename OptionsTy>
void Executor<OptionsTy>::go() {
  ThreadLocalData tld(options, loopname);
  MergeLocal& mlocal = mergeManager.get();
  tld.wlcur = &worklists[0];
  tld.wlnext = &worklists[1];

  tld.hasNewWork = false;

  while (true) {
    ++tld.outerRounds;

    while (true) {
      ++tld.rounds;

      inOrderManager.resetAborted();

      std::swap(tld.wlcur, tld.wlnext);
      setPending(PENDING);
      bool nextPending = pendingLoop(tld);
      innerDone.get() = true;

      barrier.wait();

      inOrderManager.allReduceAborted();

      setPending(COMMITTING);
      bool nextCommit = commitLoop(tld);
      outerDone.get() = true;
      if (nextPending || nextCommit)
        innerDone.get() = false;

      barrier.wait();

      if (innerDone.get())
        break;

      mergeManager.calculateWindow(tld.options, true);

      barrier.wait();

      mlocal.nextWindow(tld.wlnext);
      mlocal.resetStats();
    }

    if (!mlocal.emptyReserve())
      outerDone.get() = false;

    if (tld.hasNewWork)
      hasNewWork.get() = true;

    if (breakManager.checkBreak(tld.options.fn1))
      break;

    mergeManager.calculateWindow(tld.options, false);

    barrier.wait();

    if (outerDone.get()) {
      if (!OptionsTy::needsPush)
        break;
      if (!hasNewWork.get()) // (1)
        break;
      tld.hasNewWork = mergeManager.distributeNewWork(tld.wlnext);
      // NB: assumes that distributeNewWork has a barrier otherwise checking at (1) is erroneous
      hasNewWork.get() = false;
    } else {
      mlocal.nextWindow(tld.wlnext);
    }

    mlocal.resetStats();
  }

  setPending(NON_DET);

  mlocal.clear(); // parallelize clean up too

  if (OptionsTy::needsStats) {
    if (LL::getTID() == 0) {
      reportStat(loopname, "RoundsExecuted", tld.rounds);
      reportStat(loopname, "OuterRoundsExecuted", tld.outerRounds);
      if (OptionsTy::hasFixedWindow)
        reportStat(loopname, "FixedWindowSize", options.defaultDelta);
      if (OptionsTy::useInOrderCommit)
        reportStat(loopname, "InOrderCommit", 1);
    }
  }
}

enum ConflictFlag {
  CONFLICT = -1,
  NO_CONFLICT = 0,
  REACHED_FAILSAFE = 1,
  BREAK = 2
};

template<typename OptionsTy>
bool Executor<OptionsTy>::pendingLoop(ThreadLocalData& tld)
{
  MergeLocal& mlocal = mergeManager.get();
  bool retval = false;
  Galois::optional<Item> p;
  while ((p = tld.wlcur->pop())) {
    // Use a new context for each item because there is a race when reusing
    // between aborted iterations.

    Context* ctx = stateManager.emplaceContext(tld.localPending, pending, *p);
    inOrderManager.initializeContext(ctx);

    assert(ctx != NULL);

    mlocal.incrementIterations();
    bool commit = true;

    ctx->startIteration();
    tld.stat.inc_iterations();
    setThreadContext(ctx);

    stateManager.alloc(tld.facing, tld.options.fn1);
    int result = 0;
    try {
      tld.options.fn1(ctx->item.val, tld.facing.data());
    } catch (const ConflictFlag& flag) { clearConflictLock(); result = flag; }
    clearReleasable();
    switch (result) {
      case 0: 
      case REACHED_FAILSAFE: break;
      case CONFLICT: commit = false; break;
      default: assert(0 && "Unknown conflict flag"); abort(); break;
    }

    if (ForEachTraits<typename OptionsTy::function1_type>::NeedsPIA && !OptionsTy::useLocalState)
      tld.facing.resetAlloc();

    inOrderManager.updateAborted(ctx);

    if (commit) { 
      stateManager.save(tld.facing, ctx->item.localState);
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
  MergeLocal& mlocal = mergeManager.get();

  size_t ncommits = 0;
  size_t niter = 0;

  Context* ctx;
  while ((ctx = stateManager.peekContext(tld.localPending, pending))) {
    ++niter;
    bool commit = true;
    // Can skip this check in prefix by repeating computations but eagerly
    // aborting seems more efficient
    if (ctx->notReady())
      commit = false;
    else if (!inOrderManager.shouldCommit(ctx))
      commit = false;

    setThreadContext(ctx);
    if (commit) {
      stateManager.restore(tld.facing, ctx->item.localState);
      int result = 0;
      try {
        tld.options.fn2(ctx->item.val, tld.facing.data());
      } catch (const ConflictFlag& flag) { clearConflictLock(); result = flag; }
      clearReleasable();
      switch (result) {
        case 0: break;
        case CONFLICT: commit = false; break;
        default: assert(0 && "Unknown conflict flag"); abort(); break;
      }
    }

    stateManager.dealloc(tld.facing);
    
    if (commit) {
      ++ncommits;
      mlocal.incrementCommitted();
      if (ForEachTraits<typename OptionsTy::function2_type>::NeedsPush) {
        unsigned long parent = ctx->item.id;
        typedef typename UserContextAccess<value_type>::PushBufferTy::iterator iterator;
        unsigned count = 0;
        for (iterator ii = tld.facing.getPushBuffer().begin(), 
            ei = tld.facing.getPushBuffer().end(); ii != ei; ++ii) {
          mergeManager.pushNew(tld.options, *ii, parent, ++count, tld.wlnext, tld.hasNewWork);
          if (count == 0) {
            assert(0 && "Counter overflow");
            abort();
          }
        }
      }
      assert(ForEachTraits<typename OptionsTy::function2_type>::NeedsPush
          || tld.facing.getPushBuffer().begin() == tld.facing.getPushBuffer().end());
    } else {
      stateManager.reuseItem(ctx->item);
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
    stateManager.popContext(tld.localPending, pending);
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
  W.presort(range.begin(), range.end());

  assert(!inGaloisForEach);

  inGaloisForEach = true;
  RunCommand init(std::bind(&WorkTy::template AddInitialWork<RangeTy>, std::ref(W), std::ref(range)));
  RunCommand w[4] = {std::ref(init), 
		     std::ref(getSystemBarrier()),
		     std::ref(W),
		     std::ref(getSystemBarrier())};
  getSystemThreadPool().run(&w[0], &w[4], activeThreads);
  inGaloisForEach = false;
}


#if 0
/**
 * TODO(ddn): This executor only properly works for ordered algorithms that do
 * not create new work; otherwise, the behavior is deterministic (and clients
 * have some control over the order enforced via comp), but this executor does
 * not guarantee that a newly added activity, A, will execute before a
 * previously created activity, B, even if A < B.
 */
template<typename IterTy, typename ComparatorTy, typename NhFunc, typename OpFunc>
static inline void for_each_ordered_2p(IterTy b, IterTy e, ComparatorTy comp, NhFunc f1, OpFunc f2, const char* loopname) {
  typedef Runtime::StandardRange<IterTy> Range;
  typedef typename Range::value_type T;
  typedef Runtime::DeterministicImpl::OrderedOptions<T,NhFunc,OpFunc,ComparatorTy> OptionsTy;
  typedef Runtime::DeterministicImpl::Executor<OptionsTy> WorkTy;

  OptionsTy options(f1, f2, comp);
  WorkTy W(options, loopname);
  for_each_det_impl(makeStandardRange(b,e), W);
}
#endif

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

//#endif

#endif
