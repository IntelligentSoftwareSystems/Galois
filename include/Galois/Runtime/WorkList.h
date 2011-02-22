// Scalable Local worklists -*- C++ -*-
// This contains final worklists.

#include <queue>
#include <stack>
#include <limits>

#include "Galois/Runtime/SimpleLock.h"
#include "Galois/Runtime/PerCPU.h"
//#include "Galois/Runtime/QueuingLock.h"

#include <boost/utility.hpp>

#include <iostream>

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
  bool push(value_type);
  //! push an aborted value onto the queue
  bool aborted(value_type);
  //! pop a value from the queue.
  std::pair<bool, value_type> pop();
  //! pop a value from the queue trying not as hard to take locks
  std::pair<bool, value_type> try_pop();
  //! return if the queue *may* be empty
  bool empty();
  
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

template<typename T, int chunksize = 64, bool concurrent = true>
class FixedSizeRing :private boost::noncopyable {
  PaddedLock<concurrent> lock;
  unsigned start;
  unsigned end;
  T data[chunksize];

  bool _i_empty() {
    return start == end;
  }

  bool _i_full() {
    return (end + 1) % chunksize == start;
  }

public:
  typedef FixedSizeRing<T, chunksize, true>  ConcurrentTy;
  typedef FixedSizeRing<T, chunksize, false> SingleThreadTy;

  typedef T value_type;

  FixedSizeRing() :start(0), end(0) {}

  bool empty() {
    lock.lock();
    bool retval = _i_empty();
    lock.unlock();
    return retval;
  }

  bool full() {
    lock.lock();
    bool retval = _i_full();
    lock.unlock();
    return retval;
  }

  bool push_front(value_type val) {
    lock.lock();
    if (_i_full()) {
      lock.unlock();
      return false;
    }
    start += chunksize - 1;
    start %= chunksize;
    data[start] = val;
    lock.unlock();
    return true;
  }

  bool push_back(value_type val) {
    lock.lock();
    if (_i_full()) {
      lock.unlock();
      return false;
    }
    data[end] = val;
    end += 1;
    end %= chunksize;
    lock.unlock();
    return true;
  }

  std::pair<bool, value_type> pop_front() {
    lock.lock();
    if (_i_empty()) {
      lock.unlock();
      return std::make_pair(false, value_type());
    }
    value_type retval = data[start];
    ++start;
    start %= chunksize;
    lock.unlock();
    return std::make_pair(true, retval);
  }

  std::pair<bool, value_type> pop_back() {
    lock.lock();
    if (_i_empty()) {
      lock.unlock();
      return std::make_pair(false, value_type());
    }
    end += chunksize - 1;
    end %= chunksize;
    value_type retval = data[end];
    lock.unlock();
    return std::make_pair(true, retval);
  }
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

  typedef T value_type;

  bool push(value_type val) {
    lock();
    wl.push(val);
    unlock();
    return true;
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

  bool aborted(value_type val) {
    return push(val);
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

  typedef T value_type;

  bool push(value_type val) {
    lock();
    wl.push_back(val);
    unlock();
    return true;
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

  bool aborted(value_type val) {
    return push(val);
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
class sFIFO : private boost::noncopyable, private PaddedLock<concurrent>  {
  std::deque<T> wl;

  using PaddedLock<concurrent>::lock;
  using PaddedLock<concurrent>::try_lock;
  using PaddedLock<concurrent>::unlock;

public:
  typedef sFIFO<T, true>  ConcurrentTy;
  typedef sFIFO<T, false> SingleThreadTy;

  typedef T value_type;

  bool push(value_type val) {
    lock();
    wl.push_back(val);
    unlock();
    return true;
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

  bool aborted(value_type val) {
    return push(val);
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
class FIFO : private boost::noncopyable {

  struct Chunk {
    unsigned char start;
    unsigned char end;
    SimpleLock<unsigned char, concurrent> lock;
    Chunk* next;
    T data[256];
    Chunk() :start(0), end(0), next(0) {}
  };

  //tail shall always be not null
  Chunk* tail;
  //head shall always be not null
  Chunk* head;

  SimpleLock<long, concurrent> tailLock;
  SimpleLock<long, concurrent> headLock;

public:
  typedef FIFO<T, true>  ConcurrentTy;
  typedef FIFO<T, false> SingleThreadTy;

  typedef T value_type;

  void dump(char c) {
    // std::cerr << c << ' ' 
    // 	      << head << '{' << (int)head->start << ',' << (int)head->end << '}'
    // 	      << ' '
    // 	      << tail << '{' << (int)tail->start << ',' << (int)tail->end << '}'
    // 	      << '\n';
  }


  FIFO() {
    tail = head = new Chunk();
    dump('C');
  }

  bool push(value_type val) {
    dump('P');
    tailLock.lock();
    assert(tail);
    tail->lock.lock();
    //do all checks
    assert (!tail->next);
    if (tail->end == 255) {
      Chunk* nc = new Chunk();
      nc->lock.lock();
      tail->next = nc;
      Chunk* oldtail = tail;
      tail = nc;
      oldtail->lock.unlock();
    }
    tail->data[tail->end] = val;
    ++tail->end;
    tail->lock.unlock();
    tailLock.unlock();
    return true;
  }

  std::pair<bool, value_type> pop() {
    dump('G');
    headLock.lock();
    assert(head);
    head->lock.lock();
    if (head->start == head->end) {
      //Chunk is empty
      if (head->next) {
    	//more chunks exist
	Chunk* old = head;
	head->next->lock.lock();
	head = head->next;
	old->lock.unlock();
	delete old;
	head->lock.unlock();
	headLock.unlock();
	//try again
	return pop();
      } else {
    	// No more chunks
	head->lock.unlock();
	headLock.unlock();
	return std::make_pair(false, value_type());
      }
    } else {
      value_type retval = head->data[head->start];
      ++head->start;
      head->lock.unlock();
      headLock.unlock();
      return std::make_pair(true, retval);
    }
  }

  std::pair<bool, value_type> try_pop() {
    return pop();
  }
    
  bool empty() {
    dump('E');
    headLock.lock();
    assert(head);
    head->lock.lock();
    if (head->start == head->end) {
      //Chunk is empty
      if (head->next) {
	//more chunks exist
	head->next->lock.lock();
	Chunk* old = head;
	head = head->next;
	old->lock.unlock();
	delete old;
	head->lock.unlock();
	headLock.unlock();
	//try again
	return empty();
      } else {
	// No more chunks
	head->lock.unlock();
	headLock.unlock();
	return true;
      }
    } else {
      head->lock.unlock();
      headLock.unlock();
      return false;
    }
  }

  bool aborted(value_type val) {
    dump('A');
    return push(val);
  }

  //Not Thread Safe
  template<typename Iter>
  void fill_initial(Iter ii, Iter ee) {
    while (ii != ee) {
      push(*ii++);
    }
  }
};

template<typename T, int chunksize=64, bool concurrent=true>
class ChunkedFIFO : private boost::noncopyable {
  typedef FixedSizeRing<T, chunksize, false> Chunk;
  struct p {
    Chunk* cur;
    Chunk* next;
    p() : cur(0), next(0) {}
    ~p() {
      delete cur;
      delete next;
    }
  };

  PerCPU<p> data;
  FIFO<Chunk*, concurrent> Items;

  static void merge(p& lhs, p& rhs) {
    assert(!lhs.cur || lhs.cur->empty());
    assert(!rhs.cur || rhs.cur->empty());
    assert(!lhs.next || lhs.next->empty());
    assert(!rhs.next || rhs.next->empty());
  }

public:

  typedef T value_type;
  
  ChunkedFIFO() :data(merge) { }

  bool push(value_type val) {
    p& n = data.get();
    if (n.next && n.next->full()) {
      Items.push(n.next);
      n.next = 0;
    }
    if (!n.next)
      n.next = new Chunk;
    bool retval = n.next->push_back(val);
    assert(retval);
    return retval;
  }

  std::pair<bool, value_type> pop() {
    p& n = data.get();
    if (n.cur && n.cur->empty()) {
      delete n.cur;
      n.cur = 0;
    }
    if (!n.cur) {
      std::pair<bool, Chunk*> r = Items.pop();
      if (r.first) {
	//Shared queue had data
	n.cur = r.second;
      } else {
	//Shared queue was empty, check next
	n.cur = n.next;
	n.next = 0;
	if (!n.cur)
	  return std::make_pair(false, value_type());
      }
    }

    return n.cur->pop_front();
  }
  
  std::pair<bool, value_type> try_pop() {
    return pop();
  }
  
  bool empty() {
    p& n = data.get();
    if (n.cur && !n.cur->empty()) return false;
    if (n.next && !n.next->empty()) return false;
    if (!Items.empty()) return false;
    return true;
  }

  bool aborted(value_type val) {
    return push(val);
  }

  //Not Thread Safe
  template<typename Iter>
  void fill_initial(Iter ii, Iter ee) {
    p& n = data.get();
    for( ; ii != ee; ++ii) {
      push(*ii);
    }
    Items.push(n.next);
    n.next = 0;
  }

};

template<class T, class Indexer, typename ContainerTy = FIFO<T>, bool concurrent = true >
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
  typedef OrderedByIntegerMetric<T,Indexer,ContainerTy,false> SingleThreadTy;
  typedef OrderedByIntegerMetric<T,Indexer,ContainerTy,true> ConcurrentTy;

  OrderedByIntegerMetric(unsigned int range = 32*1024, const Indexer& x = Indexer())
    :size(range+1), I(x), cursor(&merge)
  {
    data = new ContainerTy[size];
    for (int i = 0; i < cursor.size(); ++i)
      cursor.get(i) = 0;
  }
  
  ~OrderedByIntegerMetric() {
    delete[] data;
  }

  bool push(value_type val) __attribute__((noinline)) {
    unsigned int index = I(val, size);
    assert(index < size);
    data[index].push(val);
    unsigned int& cur = concurrent ? cursor.get() : cursor.get(0);
    if (cur > index)
      cur = index;
  }

  std::pair<bool, value_type> pop()  __attribute__((noinline)) {
    // print();
    unsigned int& cur = concurrent ? cursor.get() : cursor.get(0);
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

  bool aborted(value_type val) {
    return push(val);
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
  
  bool push(value_type val) {
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
	return true;
      }
    }
    //val is either an old cached entry or the pushed one
    return data.push(val);
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

  bool aborted(value_type val) {
    return push(val);
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

  bool push(value_type val) __attribute__((noinline)) {
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
  bool aborted(value_type val) {
    return push(val);
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
  
  bool push(value_type val) {
    if (starved) {
      global.push(val);
      starved = 0;
    } else {
      local.get().push(val);
    }
  }

  bool aborted(value_type val) {
    //Fixme: should be configurable
    return global.push(val);
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
  PerCPU<unsigned int> cursor;

  int num() {
    return 2048;
  }

 public:

  typedef T value_type;
  
  ApproxOrderByIntegerMetric(const Indexer& x = Indexer())
    :I(x), cursor(0)
  {
    for (int i = 0; i < cursor.size(); ++i)
      cursor.get(i) = 0;
  }
  
  bool push(value_type val) __attribute__((noinline)) {   
    unsigned int index = I(val, std::numeric_limits<unsigned int>::max());
    index %= num();
    assert(index < num());
    data[index].push(val);
  }

  std::pair<bool, value_type> pop()  __attribute__((noinline)) {
    // print();
    unsigned int& cur = cursor.get();
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
    for (unsigned int i = 0; i < num(); ++i)
      if (!data[i].empty())
	return false;
    return true;
  }

  bool aborted(value_type val) {
    return push(val);
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
