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
#include "Galois/Callbacks.h"

#include <boost/utility/enable_if.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/counting_iterator.hpp>

#include <deque>
#include <queue>

namespace GaloisRuntime {
namespace Deterministic {

static const int ChunkSize = 32;
//! Wrapper around WorkList::ChunkedFIFO to allow peek() and empty() and still have FIFO order
template<int chunksize,typename T>
struct FIFO {
  WorkList::ChunkedFIFO<chunksize,T,false> m_data;
  WorkList::ChunkedLIFO<16,T,false> m_buffer;
  size_t m_size;
  T* m_last;

  FIFO(): m_size(0), m_last(0) { }

  ~FIFO() {
    boost::optional<T> p;
    while ((p = m_buffer.pop()))
      ;
    while ((p = m_data.pop()))
      ;
  }


  boost::optional<T> pop() {
    boost::optional<T> p;
    if ((p = m_buffer.pop()) || (p = m_data.pop())) {
      --m_size;
      if (m_size == 0)
        m_last = 0;
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

  boost::optional<T> last() {
    if (m_last) {
      return boost::optional<T>(*m_last);
    }
    return boost::optional<T>();
  }

  void push(const T& item) {
    m_last = m_data.push(item);
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

template<typename T,typename FunctionTy,typename Enable=void>
struct StateManager { 
  void alloc(GaloisRuntime::UserContextAccess<T>&, FunctionTy& self) { }
  void dealloc(GaloisRuntime::UserContextAccess<T>&) { }
  void save(GaloisRuntime::UserContextAccess<T>&, void*&) { }
  void restore(GaloisRuntime::UserContextAccess<T>&, void*) { } 
};

template<typename T,typename FunctionTy>
struct StateManager<T,FunctionTy,typename boost::enable_if<has_local_state<FunctionTy> >::type> {
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

template<typename FunctionTy,typename Enable=void>
struct BreakManager {
  BreakManager(FunctionTy&) { }
  bool checkBreak() { return false; }
};

template<typename FunctionTy>
class BreakManager<FunctionTy,typename boost::enable_if<has_break_fn<FunctionTy> >::type> {
  GBarrier barrier[1];
  LL::CacheLineStorage<volatile long> done;
  typename FunctionTy::BreakFn breakFn;

public:
  BreakManager(FunctionTy& fn): breakFn(fn) { 
    int numActive = (int) Galois::getActiveThreads();
    for (unsigned i = 0; i < sizeof(barrier)/sizeof(*barrier); ++i)
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
  ~ContextPool() {
    boost::optional<SimpleRuntimeContext> p;
    while ((p = pool.pop()))
      ;
  }

  SimpleRuntimeContext* next() {
    SimpleRuntimeContext* retval = pool.push(SimpleRuntimeContext());
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

struct OrderedTag { };
struct UnorderedTag { };

template<typename _T,typename _Function1Ty,typename _Function2Ty,typename _CompareTy>
struct OrderedOptions {
  typedef _Function1Ty Function1Ty;
  typedef _Function2Ty Function2Ty;
  typedef _T T;
  typedef _CompareTy CompareTy;
  static const bool useOrdered = true;
  typedef OrderedTag Tag;

  Function1Ty& fn1;
  Function2Ty& fn2;
  CompareTy& comp;

  OrderedOptions(Function1Ty& fn1, Function2Ty& fn2, CompareTy& comp): fn1(fn1), fn2(fn2), comp(comp) { }
};

template<typename _T,typename _Function1Ty,typename _Function2Ty>
struct UnorderedOptions {
  typedef _Function1Ty Function1Ty;
  typedef _Function2Ty Function2Ty;
  typedef _T T;
  static const bool useOrdered = false;
  typedef UnorderedTag Tag;
  
  struct DummyCompareTy: public Galois::CompareCallback {
    bool operator()(void*, void*) const {
      return false;
    }
    virtual bool compare(void*,void*) { return false; }
  };

  typedef DummyCompareTy CompareTy;

  Function1Ty& fn1;
  Function2Ty& fn2;
  CompareTy comp;

  UnorderedOptions(Function1Ty& fn1, Function2Ty& fn2): fn1(fn1), fn2(fn2) { }
};

template<typename> class DMergeManagerBase;
template<typename,typename> class DMergeManager;

template<typename Options>
class DMergeLocal {
  template<typename> friend class DMergeManagerBase;
  template<typename,typename> friend class DMergeManager;

  typedef typename Options::T T;
  typedef typename Options::CompareTy CompareTy;

  struct Compare {
    CompareTy& comp;
    Compare(CompareTy& c): comp(c) { }
    bool operator()(const T& a, const T& b) const {
      // reverse sense to get least items out of std::priority_queue
      return comp((void*) &b, (void*) &a);
    }
  };

  typedef DItem<T> Item;
  typedef DNewItem<T> NewItem;
  typedef std::vector<NewItem,typename Galois::PerIterAllocTy::rebind<NewItem>::other> NewItemsTy;
  typedef std::deque<T,typename Galois::PerIterAllocTy::rebind<T>::other> Deque;
  typedef FIFO<ChunkSize*8,Item> ReserveTy;
  typedef WorkList::ChunkedLIFO<ChunkSize*8,T,false> ItemPool;

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
  Deque newReserve;
  ItemPool itemPool;
  unsigned long minId;
  unsigned long maxId;
  
  void initialWindow(size_t w) {
    window = delta = w;
  }

  //! Update min and max from sorted iterator
  template<typename BiIteratorTy>
  void updateMinMax(BiIteratorTy ii, BiIteratorTy ei) {
    minId = std::numeric_limits<unsigned long>::max();
    maxId = std::numeric_limits<unsigned long>::min();

    if (ii != ei) {
      if (ii + 1 == ei) {
        minId = maxId = ii->parent;
      } else {
        minId = ii->parent;
        maxId = (ei-1)->parent;
      }
    }
  }

  template<typename WL>
  void nextWindowDispatch(WL* wl, Options&, UnorderedTag) {
    window += delta;
    boost::optional<Item> p;
    while ((p = reserve.peek())) {
      if (p->id > window)
        break;
      wl->push(*p);
      reserve.pop();
    }
  }

  template<typename WL>
  void nextWindowDispatch(WL* wl, Options& options, OrderedTag) {
    boost::optional<Item> p;
    size_t count = std::max((size_t) 1, delta / Galois::getActiveThreads());
    size_t i = 0;

    while ((p = reserve.peek()) && !newReserve.empty()) {
      if (options.comp((void*) &*p, (void*) &newReserve.front())) {
        wl->push(*p);
        reserve.pop();
      } else {
        wl->push(Item(newReserve.front(), 0));
        popNewReserve(options.comp);
      }
      if (i++ == count)
        return;
    }
    while ((p = reserve.peek())) {
      wl->push(*p);
      reserve.pop();
      if (i++ == count)
        return;
    }
    while (!newReserve.empty()) {
      wl->push(Item(newReserve.front(), 0));
      popNewReserve(options.comp);
      if (i++ == count)
        return;
    }
  }

  void popNewReserve(CompareTy& comp) {
    std::pop_heap(newReserve.begin(), newReserve.end(), Compare(comp));
    newReserve.pop_back();
  }

public:
  DMergeLocal(): alloc(&heap), newItems(alloc), newReserve(alloc) { resetStats(); }

  ~DMergeLocal() {
    resetItemPool();
  }

  void resetItemPool() {
    boost::optional<T> p;
    while ((p = itemPool.pop()))
      ;
  }

  T* pushItemPool(const T& item) {
    return itemPool.push(item);
  }

  void incrementIterations() {
    ++iterations;
  }

  void incrementCommitted() {
    ++committed;
  }

  template<typename WL>
  void nextWindow(WL* wl, Options& options) {
    nextWindowDispatch(wl, options, typename Options::Tag());
  }

  void resetStats() {
    committed = 0;
    iterations = 0;
    aborted = 0;
  }

  void pushNewReserve(const T& item, CompareTy& comp) {
    newReserve.push_back(item);
    std::push_heap(newReserve.begin(), newReserve.end(), Compare(comp));
  }

  void resetWindow() {
    window = 0;
  }

  bool empty() {
    return reserve.empty();
  }
};

template<typename Options>
class DMergeManagerBase {
protected:
  //static const int MinDelta = ChunkSize;
  static const int MinDelta = ChunkSize * 40;

  typedef typename Options::T T;
  typedef DItem<T> Item;
  typedef DNewItem<T> NewItem;
  typedef WorkList::dChunkedFIFO<ChunkSize,NewItem> NewWork;
  typedef DMergeLocal<Options> MergeLocal;
  typedef typename MergeLocal::NewItemsTy NewItemsTy;
  typedef typename NewItemsTy::iterator NewItemsIterator;

  Galois::IterAllocBaseTy heap;
  Galois::PerIterAllocTy alloc;
  PerThreadStorage<MergeLocal> data;

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
    MergeLocal& mlocal = *this->data.getLocal();
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
    return *data.getLocal();
  }

  void calculateWindow(bool inner) {
    MergeLocal& mlocal = *data.getLocal();

    // Accumulate all threads' info
    size_t allcommitted = 0;
    size_t alliterations = 0;
    for (int i = 0; i < numActive; ++i) {
      MergeLocal& mlocal = *data.getRemote(i);
      allcommitted += mlocal.committed;
      alliterations += mlocal.iterations;
    }

    const float target = 0.95;
    float commitRatio = alliterations > 0 ? allcommitted / (float) alliterations : 0.0;

    if (commitRatio >= target)
      mlocal.delta += mlocal.delta;
    else if (allcommitted == 0) // special case when we don't execute anything
      mlocal.delta += mlocal.delta;
    else
      mlocal.delta = commitRatio / target * mlocal.delta;

    if (!inner) {
      mlocal.delta = std::max(mlocal.delta, (size_t) MinDelta);
    } else if (mlocal.delta < (size_t) MinDelta) {
      // Try to get some new work instead of increasing window
      mlocal.delta = 0;
    }

    // Useful debugging info
    if (false) {
      if (LL::getTID() == 0) {
        printf("%.3f (%zu/%zu) window: %zu delta: %zu\n", 
            commitRatio, allcommitted, alliterations, mlocal.window, mlocal.delta);
      }
    }
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

//! Default implementation
template<typename Options,typename Enable=void>
class DMergeManager: public DMergeManagerBase<Options> {
  typedef DMergeManagerBase<Options> Base;
  typedef typename Base::T T;
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
  typedef DualLevelIterator<MergeOuterIt> MergeIt;

  std::vector<NewItem,typename Galois::PerIterAllocTy::rebind<NewItem>::other> mergeBuf;
  std::vector<T,typename Galois::PerIterAllocTy::rebind<T>::other> distributeBuf;

  GBarrier barrier[4];

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
    MergeLocal& mlocal = *this->data.getLocal();
    //size_t blockSize = std::max((size_t) (0.9*minfo.delta), (size_t) 1);
    size_t blockSize = mlocal.delta;
    size_t numBlocks = dist / blockSize;
    
    size_t cur = 0;
    this->safe_advance(b, tid, cur, dist);
    while (b != e) {
      unsigned long id;
      if (cur < blockSize * numBlocks)
        //id = (cur & mask) * blockSize + (cur / numBlocks);
        id = (cur % numBlocks) * blockSize + (cur / numBlocks);
      else
        id = cur;
      distributeBuf[id] = *b;
      this->safe_advance(b, this->numActive, cur, dist);
    }
  }

  template<typename InputIteratorTy,typename WL>
  void distribute(InputIteratorTy b, InputIteratorTy e, size_t dist, WL* wl) {
    unsigned int tid = LL::getTID();
    MergeLocal& mlocal = *this->data.getLocal();
    mlocal.initialWindow(std::max(dist / 100, (size_t) Base::MinDelta));
    if (true) {
      // Renumber to avoid pathological cases
      if (tid == 0) {
        distributeBuf.resize(dist);
      }
      barrier[0].wait();
      redistribute(b, e, dist);
      barrier[1].wait();
      this->copyIn(distributeBuf.begin(), distributeBuf.end(), dist, wl);
    } else {
      this->copyIn(b, e, dist, wl);
    }
  }

  template<typename WL>
  void parallelSort(WL* wl) {
    MergeLocal& mlocal = *this->data.getLocal();

    mlocal.newItems.clear();
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
        size += this->data.getRemote(i)->newItems.size();

      mergeBuf.reserve(size);
      
      for (int i = 0; i < this->numActive; ++i)
        this->data.getRemote(i)->size = size;

      merge(0, this->numActive);
    }

    barrier[3].wait();

    MergeOuterIt bbegin(boost::make_counting_iterator(0), GetNewItem(&this->data));
    MergeOuterIt eend(boost::make_counting_iterator(this->numActive), GetNewItem(&this->data));
    MergeIt ii(bbegin, eend), ei(eend, eend);

    distribute(boost::make_transform_iterator(ii, typename Base::NewItem::GetFirst()),
        boost::make_transform_iterator(ei, typename Base::NewItem::GetFirst()),
        mlocal.size, wl);
  }

  template<typename WL>
  void serialSort(WL* wl) {
    this->new_.flush();

    barrier[2].wait();
    
    if (LL::getTID() == 0) {
      mergeBuf.clear();
      //mergeBuf.reserve(tld.newSize * this->numActive);
      boost::optional<NewItem> p;
      while ((p = this->new_.pop())) {
        mergeBuf.push_back(*p);
      }

      std::sort(mergeBuf.begin(), mergeBuf.end());

      printf("R %ld\n", mergeBuf.size());
    }

    barrier[3].wait();

    distribute(boost::make_transform_iterator(mergeBuf.begin(), typename NewItem::GetFirst()),
        boost::make_transform_iterator(mergeBuf.end(), typename NewItem::GetFirst()),
        mergeBuf.size(), wl);
  }

public:
  DMergeManager(Options& o): mergeBuf(this->alloc), distributeBuf(this->alloc) {
    for (unsigned i = 0; i < sizeof(barrier)/sizeof(*barrier); ++i)
      barrier[i].reinit(this->numActive);
  }

  template<typename InputIteratorTy, typename WL>
  void addInitialWork(InputIteratorTy b, InputIteratorTy e, WL* wl) {
    size_t dist = std::distance(b, e);
    distribute(b, e, dist, wl);
  }

  template<typename WL>
  void pushNew(const T& item, unsigned long parent, unsigned count, WL* wl) {
    this->new_.push(NewItem(item, parent, count));
  }

  template<typename WL>
  void distributeNewWork(WL* wl) {
    if (true)
      parallelSort(wl);
    else
      serialSort(wl);
  }
};

//! Specialization for unordered algorithms with id function
template<typename Options>
class DMergeManager<Options,typename boost::enable_if_c<has_id_fn<typename Options::Function1Ty>::value && !Options::useOrdered>::type>: public DMergeManagerBase<Options> {
  typedef DMergeManagerBase<Options> Base;
  typedef typename Base::T T;
  typedef typename Base::Item Item;
  typedef typename Base::NewItem NewItem;
  typedef typename Base::MergeLocal MergeLocal;
  typedef typename Base::NewItemsTy NewItemsTy;
  typedef typename Base::NewItemsIterator NewItemsIterator;
  typedef typename Options::Function1Ty::IdFn IdFn;

  std::vector<NewItem,typename Galois::PerIterAllocTy::rebind<NewItem>::other> mergeBuf;

  GBarrier barrier[4];
  IdFn idFunction;

  void broadcastMinMax(MergeLocal& mlocal, unsigned int tid) {
    for (int i = 0; i < this->numActive; ++i) {
      if (i == (int) tid) continue;
      MergeLocal& mother = *this->data.getRemote(i);
      mother.minId = mlocal.minId;
      mother.maxId = mlocal.maxId;
    }
  }

  template<typename InputIteratorTy,typename WL>
  void distribute(InputIteratorTy b, InputIteratorTy e, size_t dist, WL* wl) {
    unsigned int tid = LL::getTID();
    MergeLocal& mlocal = *this->data.getLocal();
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
    barrier[0].wait();
    mlocal.initialWindow(std::max((mlocal.maxId - mlocal.minId) / 100, (size_t) Base::MinDelta));
    this->copyIn(boost::make_transform_iterator(mergeBuf.begin(), typename Base::NewItem::GetFirst()),
        boost::make_transform_iterator(mergeBuf.end(), typename Base::NewItem::GetFirst()),
        dist, wl);
  }

public:
  DMergeManager(Options& o): mergeBuf(this->alloc) {
    for (unsigned i = 0; i < sizeof(barrier)/sizeof(*barrier); ++i)
      barrier[i].reinit(this->numActive);
  }

  template<typename InputIteratorTy, typename WL>
  void addInitialWork(InputIteratorTy b, InputIteratorTy e, WL* wl) {
    size_t dist = std::distance(b, e);
    distribute(b, e, dist, wl);
  }

  template<typename WL>
  void pushNew(const T& item, unsigned long parent, unsigned count, WL* wl) {
    this->new_.push(NewItem(item, idFunction(item), 1));
  }

  template<typename WL>
  void distributeNewWork(WL* wl) {
    unsigned int tid = LL::getTID();
    MergeLocal& mlocal = *this->data.getLocal();

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
        MergeLocal& mother = *this->data.getRemote(i);
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

//! Specialization for ordered algorithms
template<typename Options>
class DMergeManager<Options,typename boost::enable_if_c<Options::useOrdered>::type>: public DMergeManagerBase<Options> {
  typedef DMergeManagerBase<Options> Base;
  typedef typename Base::T T;
  typedef typename Base::Item Item;
  typedef typename Base::NewItem NewItem;
  typedef typename Base::MergeLocal MergeLocal;
  typedef typename Base::NewItemsTy NewItemsTy;
  typedef typename Base::NewItemsIterator NewItemsIterator;
  typedef typename Options::CompareTy CompareTy;

  struct CompareNewItems {
    CompareTy& comp;
    CompareNewItems(CompareTy& c): comp(c) { }
    bool operator()(const NewItem& a, const NewItem& b) const {
      return comp((void*) &a.item, (void*) &b.item);
    }
  };

  std::vector<NewItem,typename Galois::PerIterAllocTy::rebind<NewItem>::other> mergeBuf;

  GBarrier barrier[4];
  CompareTy& comp;

  template<typename InputIteratorTy,typename WL>
  void distribute(InputIteratorTy b, InputIteratorTy e, size_t dist, WL* wl) {
    unsigned int tid = LL::getTID();
    MergeLocal& mlocal = *this->data.getLocal();
    if (tid == 0) {
      mergeBuf.clear();
      mergeBuf.reserve(dist);
      for (; b != e; ++b) {
        mergeBuf.push_back(NewItem(*b, 0, 1));
      }
      std::sort(mergeBuf.begin(), mergeBuf.end(), CompareNewItems(comp));
    }
    barrier[0].wait();
    mlocal.initialWindow(std::max(dist / 100, (size_t) Base::MinDelta));
    this->copyIn(boost::make_transform_iterator(mergeBuf.begin(), typename Base::NewItem::GetFirst()),
        boost::make_transform_iterator(mergeBuf.end(), typename Base::NewItem::GetFirst()),
        dist, wl);
  }

public:
  DMergeManager(Options& o): mergeBuf(this->alloc), comp(o.comp) {
    for (unsigned i = 0; i < sizeof(barrier)/sizeof(*barrier); ++i)
      barrier[i].reinit(this->numActive);
  }

  template<typename InputIteratorTy, typename WL>
  void addInitialWork(InputIteratorTy b, InputIteratorTy e, WL* wl) {
    size_t dist = std::distance(b, e);
    distribute(b, e, dist, wl);
  }

  template<typename WL>
  void pushNew(const T& item, unsigned long parent, unsigned count, WL* wl) {
    MergeLocal& mlocal = *this->data.getLocal();
    boost::optional<Item> head = mlocal.reserve.peek();
    boost::optional<Item> last = mlocal.reserve.last();

    if (!head || comp((void*) &item, (void*) &head->item)) {
      wl->push(Item(item, 0));
    } else if (comp((void*) &item, (void*) &last->item)) {
      mlocal.pushNewReserve(item, comp); 
    } else {
      this->new_.push(NewItem(item, 0, 1));
    }
  }

  template<typename WL>
  void distributeNewWork(WL* wl) {
    MergeLocal& mlocal = *this->data.getLocal();

    mlocal.newItems.clear();
    boost::optional<NewItem> p;
    while ((p = this->new_.pop())) {
      mlocal.newItems.push_back(*p);
    }

    std::sort(mlocal.newItems.begin(), mlocal.newItems.end(), CompareNewItems(comp));
    NewItemsIterator ii = mlocal.newItems.begin(), ei = mlocal.newItems.end();

    size_t top = std::max((size_t) 1, mlocal.delta / this->numActive);
    for (size_t count = 0; ii != ei; ++ii, ++count) {
      if (count > top)
        mlocal.reserve.push(Item(ii->item, 0));
      else
        wl->push(Item(ii->item, 0));
    }
  }
};

template<typename Options>
class Executor {
  typedef typename Options::T value_type;
  typedef DItem<value_type> Item;
  typedef DNewItem<value_type> NewItem;
  typedef DMergeManager<Options> MergeManager;
  typedef DMergeLocal<Options> MergeLocal;
  typedef WorkList::dChunkedFIFO<ChunkSize,Item> WL;
  typedef WorkList::dChunkedFIFO<ChunkSize,Item> PendingWork;
  typedef WorkList::ChunkedFIFO<ChunkSize,Item,false> LocalPendingWork;
  static const bool useLocalState = has_local_state<typename Options::Function1Ty>::value;

  // Truly thread-local
  struct ThreadLocalData: private boost::noncopyable {
    LocalPendingWork localPending;
    GaloisRuntime::UserContextAccess<value_type> facing;
    LoopStatistics<ForEachTraits<typename Options::Function1Ty>::NeedsStats || ForEachTraits<typename Options::Function2Ty>::NeedsStats> stat;
    WL* wlcur;
    WL* wlnext;
    size_t rounds;
    size_t outerRounds;
    bool hasNewWork;
    ThreadLocalData(const char* loopname): stat(loopname), rounds(0), outerRounds(0) { }
  };

  GBarrier barrier[4];
  WL worklists[2];
  BreakManager<typename Options::Function1Ty> breakManager;
  MergeManager mergeManager;
  StateManager<value_type,typename Options::Function1Ty> stateManager;
  PendingWork pending;
  ContextPool contextPool;
  Options& options;
  const char* loopname;
  LL::CacheLineStorage<volatile long> innerDone;
  LL::CacheLineStorage<volatile long> outerDone;
  LL::CacheLineStorage<volatile long> hasNewWork;
  int numActive;

  bool pendingLoop(ThreadLocalData& tld);
  bool commitLoop(ThreadLocalData& tld);
  void go();

public:
  Executor(Options& o, const char* ln):
    breakManager(o.fn1), mergeManager(o), options(o), loopname(ln)
  { 
    numActive = (int) Galois::getActiveThreads();
    for (unsigned i = 0; i < sizeof(barrier)/sizeof(*barrier); ++i)
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

template<typename Options>
void Executor<Options>::go() {
  ThreadLocalData tld(loopname);
  MergeLocal& mlocal = mergeManager.get();
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
      innerDone.data = true;

      barrier[1].wait();

      setPending(COMMITTING);
      bool nextCommit = commitLoop(tld);
      outerDone.data = true;
      if (nextPending || nextCommit)
        innerDone.data = false;

      barrier[2].wait();

      contextPool.commitAll();
      mlocal.resetItemPool();

      if (innerDone.data) {
        break;
      }

      mergeManager.calculateWindow(true);

      barrier[0].wait();

      mlocal.nextWindow(tld.wlnext, options);
      mlocal.resetStats();
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

    if (outerDone.data) {
      if (!ForEachTraits<typename Options::Function1Ty>::NeedsPush && !ForEachTraits<typename Options::Function2Ty>::NeedsPush)
        break;
      if (!hasNewWork.data)
        break;
      mergeManager.distributeNewWork(tld.wlnext);
      mlocal.resetWindow();
      tld.hasNewWork = false;
      hasNewWork.data = false;
    }

    mlocal.nextWindow(tld.wlnext, options);
    mlocal.resetStats();
  }

  setPending(NON_DET);

  if (ForEachTraits<typename Options::Function1Ty>::NeedsStats || ForEachTraits<typename Options::Function2Ty>::NeedsStats) {
    if (LL::getTID() == 0) {
      reportStat(loopname, "RoundsExecuted", tld.rounds);
      reportStat(loopname, "OuterRoundsExecuted", tld.outerRounds);
    }
  }
}

template<typename Options>
bool Executor<Options>::pendingLoop(ThreadLocalData& tld)
{
  SimpleRuntimeContext* cnx = contextPool.next();
  MergeLocal& mlocal = mergeManager.get();
  bool retval = false;
  boost::optional<Item> p;
  while ((p = tld.wlcur->pop())) {
    mlocal.incrementIterations();
    bool commit = true;
    cnx->set_id(p->id);
    if (Options::useOrdered) {
      cnx->set_comp(&options.comp);
      cnx->set_comp_data(mlocal.pushItemPool(p->item));
    }
    cnx->start_iteration();
    tld.stat.inc_iterations();
    setThreadContext(cnx);
    stateManager.alloc(tld.facing, options.fn1);
    int result = 0;
#if GALOIS_USE_EXCEPTION_HANDLER
    try {
      options.fn1(p->item, tld.facing.data());
    } catch (ConflictFlag flag) {
      clearConflictLock();
      result = flag;
    }
#else
    if ((result = setjmp(hackjmp)) == 0) {
      options.fn1(p->item, tld.facing.data());
    }
#endif
    switch (result) {
      case 0: 
      case REACHED_FAILSAFE: break;
      case CONFLICT: commit = false; break;
      default: assert(0 && "Unknown conflict flag"); abort(); break;
    }

    if (ForEachTraits<typename Options::Function1Ty>::NeedsPIA && !useLocalState)
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
      stateManager.dealloc(tld.facing); 
      tld.wlnext->push(*p);
      tld.stat.inc_conflicts();
      retval = true;
    }

    cnx = contextPool.next();
  }

  return retval;
}

template<typename Options>
bool Executor<Options>::commitLoop(ThreadLocalData& tld) 
{
  bool retval = false;
  MergeLocal& mlocal = mergeManager.get();
  boost::optional<Item> p;

  while ((p = (useLocalState) ? tld.localPending.pop() : pending.pop())) {
    bool commit = true;
    if (!p->cnx->is_ready())
      commit = false;

    if (commit) {
      setThreadContext(p->cnx);
      stateManager.restore(tld.facing, p->localState);
      int result = 0;
#if GALOIS_USE_EXCEPTION_HANDLER
      try {
        options.fn2(p->item, tld.facing.data());
      } catch (ConflictFlag flag) {
        clearConflictLock();
        result = flag;
      }
#else
      if ((result = setjmp(hackjmp)) == 0) {
        options.fn2(p->item, tld.facing.data());
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
      mlocal.incrementCommitted();
      if (ForEachTraits<typename Options::Function2Ty>::NeedsPush) {
        unsigned long parent = p->id;
        typedef typename UserContextAccess<value_type>::pushBufferTy::iterator iterator;
        unsigned count = 0;
        for (iterator ii = tld.facing.getPushBuffer().begin(), 
            ei = tld.facing.getPushBuffer().end(); ii != ei; ++ii) {
          mergeManager.pushNew(*ii, parent, ++count, tld.wlnext);
          tld.hasNewWork = true;
          if (count == 0) {
            assert(0 && "Counter overflow");
            abort();
          }
        }
      }
      assert(ForEachTraits<typename Options::Function2Ty>::NeedsPush
          || tld.facing.getPushBuffer().begin() == tld.facing.getPushBuffer().end());
    } else {
      p->cnx = NULL;
      //if (useLocalState) p->localState = NULL;
      tld.wlnext->push(*p);
      tld.stat.inc_conflicts();
      retval = true;
    }

    if (ForEachTraits<typename Options::Function2Ty>::NeedsPIA && !useLocalState)
      tld.facing.resetAlloc();

    tld.facing.resetPushBuffer();
  }

  if (ForEachTraits<typename Options::Function2Ty>::NeedsPIA && useLocalState)
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

//! Deterministic execution with prefix 
template<typename IterTy, typename Function1Ty, typename Function2Ty>
static inline void for_each_det(IterTy b, IterTy e, Function1Ty f1, Function2Ty f2, const char* loopname = 0) {
  typedef typename std::iterator_traits<IterTy>::value_type T;
  typedef GaloisRuntime::Deterministic::UnorderedOptions<T,Function1Ty,Function2Ty> Options;
  typedef GaloisRuntime::Deterministic::Executor<Options> WorkTy;

  Options options(f1, f2);
  WorkTy W(options, loopname);
  GaloisRuntime::Initializer<IterTy, WorkTy> init(b, e, W);
  for_each_det_impl(init, W);
}

template<typename T, typename Function1Ty, typename Function2Ty>
static inline void for_each_det(T e, Function1Ty f1, Function2Ty f2, const char* loopname = 0) {
  T wl[1] = { e };
  Galois::for_each_det(&wl[0], &wl[1], f1, f2, loopname);
}

//! Deterministic execution
template<typename IterTy, typename FunctionTy>
static inline void for_each_det(IterTy b, IterTy e, FunctionTy f, const char* loopname = 0) {
  Galois::for_each_det(b, e, f, f, loopname);
}

template<typename T, typename FunctionTy>
static inline void for_each_det(T e, FunctionTy f, const char* loopname = 0) {
  T wl[1] = { e };
  Galois::for_each_det(&wl[0], &wl[1], f, f, loopname);
}

}
#endif

#endif
