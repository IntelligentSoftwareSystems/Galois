// Scalable Local worklists -*- C++ -*-
// This contains final worklists.

#include <queue>
#include <stack>
#include <limits>

#include "Galois/Runtime/SimpleLock.h"
#include "Galois/Runtime/PerCPU.h"
#include "Galois/Runtime/QueuingLock.h"

#include <boost/utility.hpp>

//#include <iostream>

namespace GaloisRuntime {
namespace WorkList {

// Worklists may not be copied.
// Worklists should be default instantiatable
// All classes (should) conform to:
template<typename T>
class AbstractWorkList {
public:
  //! T is the value type of the WL
  typedef T value_type;
  //! Optional for top-level queues
  typedef AbstractWorkList<T> ConcurrentTy;
  //! Optional for top-level queues
  typedef AbstractWorkList<typename T::X> SingleThreadTy;
  
  //! push a value onto the queue
  void push(value_type);
  //! push an aborted value onto the queue
  void aborted(value_type);
  //! pop a value from the queue.
  std::pair<bool, value_type> pop();
  //! pop a value from the queue trying not as hard to take locks
  std::pair<bool, value_type> try_pop();
  //! return if the queue *may* be empty
  bool empty();
  //! return the number of items in the list
  //! this is not called size because it may not be constant
  unsigned count();
  
  //! called in sequential mode to seed the worklist
  template<typename iter>
  void fillInitial(iter begin, iter end);


};

template<bool concurrent>
class PaddedLock;

template<>
class PaddedLock<true> {
  cache_line_storage<SimpleLock<int, true> > Lock;

public:
  void lock() { Lock.data.lock(); }
  bool try_lock() { return Lock.data.try_lock(); }
  void unlock() { Lock.data.unlock(); }
};
template<>
class PaddedLock<false> {
public:
  void lock() {}
  bool try_lock() { return true; }
  void unlock() {}
};

template<typename T, class Compare = std::less<T>, bool concurrent = true>
class PriQueue : private boost::noncopyable, private PaddedLock<concurrent> {

  std::priority_queue<T, std::vector<T>, Compare> wl;

  using PaddedLock<concurrent>::lock;
  using PaddedLock<concurrent>::try_lock;
  using PaddedLock<concurrent>::unlock;

public:
  typedef PriQueue<T, Compare, true>  ConcurrentTy;
  typedef PriQueue<T, Compare, false> SingleThreadTy;

  friend class PriQueue<T, Compare, true>;
  friend class PriQueue<T, Compare, false>;

  typedef T value_type;

  void push(value_type val) {
    lock();
    wl.push(val);
    unlock();
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

  std::pair<bool, value_type> try_pop() {
    if (try_lock()) {
      if (!wl.empty()) {
	value_type retval = wl.top();
	wl.pop();
	unlock();
	return std::make_pair(true, retval);
      }
      unlock();
    }
    return std::make_pair(false, value_type());
  }
   
  bool empty() {
    lock();
    bool retval = wl.empty();
    unlock();
    return retval;
  }

  unsigned count() {
    lock();
    unsigned ret = wl.size();
    unlock();
    return ret;
  }

  void aborted(value_type val) {
    push(val);
  }

  //Not Thread Safe
  template<typename Iter>
  void fill_initial(Iter ii, Iter ee) {
    while (ii != ee) {
      wl.push(*ii++);
    }
  }
};

template<typename T, bool concurrent = true>
class LIFO : private boost::noncopyable, private PaddedLock<concurrent> {
  std::vector<T> wl;

  using PaddedLock<concurrent>::lock;
  using PaddedLock<concurrent>::try_lock;
  using PaddedLock<concurrent>::unlock;

public:
  typedef LIFO<T, true>  ConcurrentTy;
  typedef LIFO<T, false> SingleThreadTy;

  friend class LIFO<T, true>;
  friend class LIFO<T, false>;

  typedef T value_type;

  void push(value_type val) {
    lock();
    wl.push_back(val);
    unlock();
  }

  std::pair<bool, value_type> pop() {
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

  std::pair<bool, value_type> try_pop() {
    if (try_lock()) {
      if (!wl.empty()) {
	value_type retval = wl.back();
	wl.pop_back();
	unlock();
	return std::make_pair(true, retval);
      }
      unlock();
    }
    return std::make_pair(false, value_type());
  }

  bool empty() {
    lock();
    bool retval = wl.empty();
    unlock();
    return retval;
  }

  bool count() {
    lock();
    unsigned retval = wl.size();
    unlock();
    return retval;
  }

  void aborted(value_type val) {
    push(val);
  }

  //Not Thread Safe
  template<typename Iter>
  void fill_initial(Iter ii, Iter ee) {
    while (ii != ee) {
      wl.push(*ii++);
    }
  }
};

template<typename T, bool concurrent = true>
class FIFO : private boost::noncopyable, private PaddedLock<concurrent> {
  std::deque<T> wl;

  using PaddedLock<concurrent>::lock;
  using PaddedLock<concurrent>::try_lock;
  using PaddedLock<concurrent>::unlock;

public:
  typedef FIFO<T, true>  ConcurrentTy;
  typedef FIFO<T, false> SingleThreadTy;

  friend class FIFO<T, true>;
  friend class FIFO<T, false>;

  typedef T value_type;

  void push(value_type val) {
    lock();
    wl.push_back(val);
    unlock();
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

  std::pair<bool, value_type> try_pop() {
    if (try_lock()) {
      if (!wl.empty()) {
	value_type retval = wl.front();
	wl.pop_front();
	unlock();
	return std::make_pair(true, retval);
      }
      unlock();
    }
    return std::make_pair(false, value_type());
  }
    
  bool empty() {
    lock();
    bool retval = wl.empty();
    unlock();
    return retval;
  }

  unsigned count() {
    lock();
    unsigned retval = wl.size();
    unlock();
    return retval;
  }

  void aborted(value_type val) {
    push(val);
  }

  //Not Thread Safe
  template<typename Iter>
  void fill_initial(Iter ii, Iter ee) {
    while (ii != ee) {
      wl.push(*ii++);
    }
  }
};

template<typename T, int ChunkSize = 64, bool pushToLocal = true, typename BackingTy = LIFO<T> >
class ChunkedFIFO : private boost::noncopyable {
public:
   typedef T value_type;
private:
  typedef typename BackingTy::SingleThreadTy Chunk;

  typename FIFO<Chunk*>::ConcurrentTy Items;

  struct ProcRec {
    Chunk* next;
    int nextSize;
    Chunk* curr;
    ProcRec() : next(0), nextSize(0), curr(0) {}
    static void merge( ProcRec& lhs, ProcRec& rhs) {
      assert(!lhs.next || lhs.next->empty());
      assert(!lhs.curr || lhs.curr->empty());
      assert(!rhs.next || rhs.next->empty());
      assert(!rhs.curr || rhs.curr->empty());
    }
    bool empty() const {
      if (curr && !curr->empty()) return false;
      if (next && !next->empty()) return false;
      return true;
    }
  };

  PerCPU<ProcRec> data;

  void push_next(ProcRec& n, value_type val) {
    if (!n.next) {
      n.next = new Chunk;
      n.nextSize = 0;
    }
    if (n.nextSize == ChunkSize) {
      Items.push(n.next);
      n.next = new Chunk;
      n.nextSize = 0;
    }
    n.next->push(val);
    n.nextSize++;
  }

  void push_local(ProcRec& n, value_type val) {
    if (!n.curr)
      fill_curr(n);

    if (n.curr)
      n.curr->push(val);
    else
      push_next(n, val);
  }

  void fill_curr(ProcRec& n) {
    std::pair<bool, Chunk*> r = Items.pop();
    if (r.first) { // Got one
      n.curr = r.second;
    } else { //try taking over next
      n.curr = n.next;
      n.next = 0;
    }
  }

public:
  
  //typedef STLAdaptor<MQ, true>  ConcurrentTy;
  //typedef STLAdaptor<MQ, false> SingleThreadTy;

  //typedef typename MQ::value_type value_type;

  ChunkedFIFO() :data(ProcRec::merge) {
    // assert(data.getCount() > 1);
  }

  void push(value_type val) {
    ProcRec& n = data.get();
    if (pushToLocal)
      push_local(n, val);
    else
      push_next(n, val);
  }

  std::pair<bool, value_type> pop() {
    ProcRec& n = data.get();
    if (!n.curr) //Curr is empty, graph a new chunk
      fill_curr(n);

    //n.curr may still be null
    if (n.curr) {
      if (n.curr->empty()) {
	delete n.curr;
	n.curr = 0;
	return pop();
      } else {
	return n.curr->pop();
      }
    } else {
      return std::make_pair(false, value_type());
    }
  }
    
  bool empty() {
    ProcRec& n = data.get();
    if (!n.empty()) return false;
    if (!Items.empty()) return false;
    return true;
  }

  void aborted(value_type val) {
    ProcRec& n = data.get();
    push_next(n, val);
  }

  //Not Thread Safe
  template<typename Iter>
  void fill_initial(Iter ii, Iter ee) {
    ProcRec& n = data.get();
    for( ; ii != ee; ++ii) {
      push_next(n, *ii);
    }
    Items.push(n.next);
    n.next = 0;
  }

};


template<class T, class Indexer, typename ContainerTy = FIFO<T> >
class OrderedByIntegerMetric : private boost::noncopyable {

  ContainerTy* data;
  unsigned int size;
  Indexer I;
  PerCPU<unsigned int> cursor;

  static void merge(unsigned int& x, unsigned int& y) {
    x = 0;
    y = 0;
  }

  // void print() {
  //   static int iter = 0;
  //   ++iter;
  //   if (iter % 1024 == 0) {
  //     unsigned c[32];
  //     for (int i = 0; i < 32; ++i)
  // 	c[i] = 0;
  //     for (unsigned int i = 0; i < size; ++i)
  // 	c[i % 32] += data[i].count();
  //     for (int i = 0; i < 31; ++i)
  // 	std::cout << c[i] << ",";
  //     std::cout << c[31] << "\n";
  //   }
  // }

 public:

  typedef T value_type;
  
  OrderedByIntegerMetric(unsigned int range, const Indexer& x = Indexer())
    :size(range+1), I(x), cursor(&merge)
  {
    data = new ContainerTy[size];
    for (int i = 0; i < cursor.size(); ++i)
      cursor.get(i) = 0;
  }
  
  ~OrderedByIntegerMetric() {
    delete[] data;
  }

  void push(value_type val) __attribute__((noinline)) {
    unsigned int index = I(val, size);
    assert(index < size);
    data[index].push(val);
    unsigned int& cur = cursor.get();
    if (cur > index)
      cur = index;
  }

  std::pair<bool, value_type> pop()  __attribute__((noinline)) {
    // print();
    unsigned int& cur = cursor.get();
    std::pair<bool, value_type> ret;
    //Find a successful pop
    assert(cur < size);
    ret = data[cur].try_pop();
    if (ret.first)
      return ret;

    //cursor failed, scan from front
    //assuming queues tend to be full, this should let us pick up good
    //items sooner
    for (cur = 0; cur < size; ++cur) {
      ret = data[cur].try_pop();
      if (ret.first)
	return ret;
    }
    cur = 0;
    ret.first = false;
    return ret;
  }

  std::pair<bool, value_type> try_pop()  __attribute__((noinline)) {
    return pop();
  }

  bool empty() const {
    for (unsigned int i = 0; i < size; ++i)
      if (!data[i].empty())
	return false;
    return true;
  }

  unsigned count() {
    unsigned c = 0;
    for (unsigned int i = 0; i < size; ++i)
      c += data[i].size();
    return c;
  }

  void aborted(value_type val) {
    push(val);
  }

  //Not Thread Safe
  template<typename Iter>
  void fill_initial(Iter ii, Iter ee) {
    while (ii != ee) {
      push(*ii++);
    }
  }
};

template<typename ParentTy, unsigned int size, class Indexer>
class CacheByIntegerMetric : private boost::noncopyable {
  
  typedef typename ParentTy::value_type T;

  ParentTy& data;

  struct __cacheTy{
    bool valid;
    T data;
    __cacheTy() :valid(0) {}
  };
  struct cacheTy {
    __cacheTy cacheInst[size];
  };
  typedef __cacheTy (&cacheRef)[size];
  typedef const __cacheTy (&constCacheRef)[size];

  PerCPU<cacheTy> cache;
  Indexer I;

  static void merge(cacheTy& x, cacheTy& y) {
  }

 public:

  typedef T value_type;
  
  CacheByIntegerMetric(ParentTy& P, const Indexer& x = Indexer())
    :data(P), cache(merge), I(x,size)
  { }
  
  void push(value_type val) {
    cacheRef c = cache.get().cacheInst;
    unsigned int valIndex = I(val,size);

    for (unsigned int i = 0; i < size; ++i) {
      if (c[i].valid) {
	if (valIndex < I(c[i].data,size)) {
	  //swap
	  value_type tmp = c[i].data;
	  c[i].data = val;
	  val = tmp;
	  valIndex = I(val,size);
	}
      } else { //slot open
	c[i].valid = true;
	c[i].data = val;
	return;
      }
    }
    //val is either an old cached entry or the pushed one
    data.push(val);
  }

  std::pair<bool, value_type> pop() {
    cacheRef c = cache.get().cacheInst;

    for (unsigned int i = 0; i < size; ++i)
      if (c[i].valid) {
	value_type v = c[i].data;
	c[i].valid = false;
	return std::make_pair(true, v);
      }

    //cache was empty
    return data.pop();
  }

  bool empty() const {
    constCacheRef c = cache.get().cacheInst;

    for (unsigned int i = 0; i < size; ++i)
      if (c[i].valid)
	return false;
    return data.empty();
  }

  void aborted(value_type val) {
    push(val);
  }

  //Not Thread Safe
  template<typename Iter>
  void fill_initial(Iter ii, Iter ee) {
    data.fill_initial(ii, ee);
  }
};

template<class T, typename ContainerTy = FIFO<T> >
class StealingLocalWL : private boost::noncopyable {

  PerCPU_ring<ContainerTy> data;

  static void merge(ContainerTy& x, ContainerTy& y) {
    assert(x.empty());
    assert(y.empty());
  }

 public:

  typedef T value_type;
  
  StealingLocalWL() :data(&merge) {}

  void push(value_type val) __attribute__((noinline)) {
    data.get().push(val);
  }

  std::pair<bool, value_type> pop()  __attribute__((noinline)) {
    std::pair<bool, value_type> ret = data.get().pop();
    if (ret.first)
      return ret;
    return data.getNext().pop();
  }

  std::pair<bool, value_type> try_pop()  __attribute__((noinline)) {
    std::pair<bool, value_type> ret = data.get().try_pop();
    if (ret.first)
      return ret;
    return data.getNext().try_pop();
  }

  bool empty() {
    return data.get().empty();
  }
  void aborted(value_type val) {
    push(val);
  }

  //Not Thread Safe
  template<typename Iter>
  void fill_initial(Iter ii, Iter ee) {
    while (ii != ee) {
      push(*ii++);
    }
  }
};


template<typename T, typename GlobalQueueTy, typename LocalQueueTy>
class LocalQueues {

  PerCPU<typename LocalQueueTy::SingleThreadTy> local;
  GlobalQueueTy global;
  bool starved;

public:
  LocalQueues() :local(0), starved(false) {}

  typedef T value_type;
  
  void push(value_type val) {
    if (starved) {
      global.push(val);
      starved = 0;
    } else {
      local.get().push(val);
    }
  }

  void aborted(value_type val) {
    //Fixme: should be configurable
    global.push(val);
  }

  std::pair<bool, value_type> pop() {
    std::pair<bool, value_type> ret = local.get().pop();
    if (ret.first)
      return ret;
    ret = global.pop();
    if (!ret.first)
      starved = true;
    return ret;
  }

  std::pair<bool, value_type> try_pop() {
    std::pair<bool, value_type> ret = local.get().try_pop();
    if (ret.first)
      return ret;
    ret = global.try_pop();
    if (!ret.first)
      starved = true;
    return ret;
  }

  bool empty() {
    if (!local.get().empty()) return false;
    return global.empty();
  }

  template<typename iter>
  void fillInitial(iter begin, iter end) {
    while (begin != end)
      global.push(*begin++);
  }
};

template<class T, class Indexer, typename ContainerTy = FIFO<T> >
class ApproxOrderByIntegerMetric : private boost::noncopyable {

  ContainerTy data[2048];
  
  Indexer I;
  PerCPU<int> cursor;

  int num() {
    return 2048;
  }

 public:

  typedef T value_type;
  
  ApproxOrderByIntegerMetric(const Indexer& x = Indexer())
    :I(x), cursor(0)
  {  }
  
  void push(value_type val) __attribute__((noinline)) {   
    unsigned int index = I(val, std::numeric_limits<unsigned int>::max());
    index %= num();
    assert(index < num());
    data[index].push(val);
  }

  std::pair<bool, value_type> pop()  __attribute__((noinline)) {
    // print();
    int& cur = cursor.get();
    std::pair<bool, value_type> ret = data[cur].pop();
    if (ret.first)
      return ret;

    //must move cursor
    for (int i = 0; i < num(); ++i) {
      cur = (cur + 1) % num();
      ret = data[cur].try_pop();
      if (ret.first)
	return ret;
    }
    return std::pair<bool, value_type>(false, value_type());
  }

  std::pair<bool, value_type> try_pop()  __attribute__((noinline)) {
    return pop();
  }

  bool empty() {
    for (int i = 0; i < num(); ++i)
      if (!data[i].empty())
	return false;
    return true;
  }

  unsigned count() {
    unsigned c = 0;
    for (unsigned int i = 0; i < num(); ++i)
      c += data[i].count();
    return c;
  }

  void aborted(value_type val) {
    push(val);
  }

  //Not Thread Safe
  //Not ideal
  template<typename Iter>
  void fill_initial(Iter ii, Iter ee) {
    while (ii != ee) {
      push(*ii++);
    }
  }
};


//End namespace
}
}
