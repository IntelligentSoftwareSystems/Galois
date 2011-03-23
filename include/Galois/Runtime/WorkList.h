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

//#define OPTNOINLINE __attribute__((noinline)) 
#define OPTNOINLINE

namespace GaloisRuntime {
namespace WorkList {

// Worklists may not be copied.
// Worklists should be default instantiatable
// All classes (should) conform to:
template<typename T, bool concurrent>
class AbstractWorkList {
public:
  //! T is the value type of the WL
  typedef T value_type;

  //! change the concurrency flag
  template<bool newconcurrent>
  struct rethread {
    typedef AbstractWorkList<T, newconcurrent> WL;
  };

  //! push a value onto the queue
  bool push(value_type val);
  //! push an aborted value onto the queue
  bool aborted(value_type val);
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
  template<bool newconcurrent>
  struct rethread {
    typedef FixedSizeRing<T, chunksize, newconcurrent> WL;
  };

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
  template<bool newconcurrent>
  struct rethread {
    typedef PriQueue<T, Compare, newconcurrent> WL;
  };

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
  template<bool newconcurrent>
  struct rethread {
    typedef LIFO<T, newconcurrent> WL;
  };

  typedef T value_type;

  bool push(value_type val) OPTNOINLINE {
    lock();
    wl.push_back(val);
    unlock();
    return true;
  }

  std::pair<bool, value_type> pop() OPTNOINLINE {
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

  std::pair<bool, value_type> try_pop() OPTNOINLINE {
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

  bool empty() OPTNOINLINE {
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
  template<bool newconcurrent>
  struct rethread {
    typedef sFIFO<T, newconcurrent> WL;
  };

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
  template<bool newconcurrent>
  struct rethread {
    typedef FIFO<T, newconcurrent> WL;
  };

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

struct NextPtr {
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
  sFIFO<Chunk*, concurrent> Items;

  void pushChunk(Chunk* C) {
    Items.push(C);
  }

  Chunk* popChunk() {
    std::pair<bool, Chunk*> r = Items.pop();
    if (r.first)
      return r.second;
    else
      return 0;
  }

public:
  template<bool newconcurrent>
  struct rethread {
    typedef ChunkedFIFO<T, chunksize, newconcurrent> WL;
  };

  typedef T value_type;
  
  ChunkedFIFO() :Items() {}

  bool push(value_type val) {
    p& n = data.get();
    if (n.next && n.next->full()) {
      pushChunk(n.next);
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
      Chunk* r = popChunk();
      if (r) {
	//Shared queue had data
	n.cur = r;
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
    pushChunk(n.next);
    n.next = 0;
  }

};

template<class T, class Indexer, typename ContainerTy = FIFO<T>, bool concurrent = true >
class OrderedByIntegerMetric : private boost::noncopyable {

  typedef typename ContainerTy::template rethread<concurrent>::WL CTy;

  CTy* data;
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
  template<bool newconcurrent>
  struct rethread {
    typedef  OrderedByIntegerMetric<T,Indexer,ContainerTy,newconcurrent> WL;
  };

  typedef T value_type;

  OrderedByIntegerMetric(unsigned int range = 32*1024, const Indexer& x = Indexer())
    :size(range+1), I(x), cursor(&merge)
  {
    data = new CTy[size];
    for (int i = 0; i < cursor.size(); ++i)
      cursor.get(i) = 0;
  }
  
  ~OrderedByIntegerMetric() {
    delete[] data;
  }

  bool push(value_type val) {
    unsigned int index = I(val);
    index = std::min(index, size - 1);
    assert(index < size);
    data[index].push(val);
    unsigned int& cur = concurrent ? cursor.get() : cursor.get(0);
    if (cur > index)
      cur = index;
  }

  std::pair<bool, value_type> pop() {
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

  std::pair<bool, value_type> try_pop() {
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

template<class T, typename ContainerTy = FIFO<T> >
class StealingLocalWL : private boost::noncopyable {

  PerCPU<ContainerTy> data;

  static void merge(ContainerTy& x, ContainerTy& y) {
    assert(x.empty());
    assert(y.empty());
  }

 public:
  template<bool newconcurrent>
  struct rethread {
    typedef StealingLocalWL<T, ContainerTy> WL;
  };

  typedef T value_type;
  
  StealingLocalWL() :data(&merge) {}

  bool push(value_type val) {
    data.get().push(val);
  }

  std::pair<bool, value_type> pop() {
    std::pair<bool, value_type> ret = data.get().pop();
    if (ret.first)
      return ret;
    return data.getNext().pop();
  }

  std::pair<bool, value_type> try_pop() {
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

  PerCPU<typename LocalQueueTy::template rethread<false>::WL> local;
  GlobalQueueTy global;

public:
  template<bool newconcurrent>
  struct rethread {
    typedef LocalQueues<T, GlobalQueueTy, LocalQueueTy> WL;
  };

  typedef T value_type;

  LocalQueues() {}

  bool push(value_type val) {
    local.get().push(val);
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
    return ret;
  }

  std::pair<bool, value_type> try_pop() {
    return pop();
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

template<class T, class Indexer, typename ContainerTy = FIFO<T>, bool concurrent=true >
class ApproxOrderByIntegerMetric : private boost::noncopyable {

  typename ContainerTy::template rethread<concurrent>::WL data[2048];
  
  Indexer I;
  PerCPU<unsigned int> cursor;

  int num() {
    return 2048;
  }

 public:

  typedef T value_type;
  template<bool newconcurrent>
  struct rethread {
    typedef ApproxOrderByIntegerMetric<T, Indexer, ContainerTy, newconcurrent> WL;
  };
  
  ApproxOrderByIntegerMetric(const Indexer& x = Indexer())
    :I(x)
  {
    for (int i = 0; i < cursor.size(); ++i)
      cursor.get(i) = 0;
  }
  
  bool push(value_type val) OPTNOINLINE {   
    unsigned int index = I(val);
    index %= num();
    assert(index < num());
    data[index].push(val);
  }

  std::pair<bool, value_type> pop() OPTNOINLINE {
    // print();
    unsigned int& cur = concurrent ? cursor.get() : cursor.get(0);
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

  std::pair<bool, value_type> try_pop() OPTNOINLINE {
    return pop();
  }

  bool empty() OPTNOINLINE {
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

template<class T, class Indexer, typename ContainerTy = FIFO<T>, bool concurrent=true >
class LogOrderByIntegerMetric : private boost::noncopyable {

  typename ContainerTy::template rethread<concurrent>::WL data[sizeof(unsigned int)*8 + 1];
  
  Indexer I;
  PerCPU<unsigned int> cursor;

  int num() {
    return sizeof(unsigned int)*8 + 1;
  }

  int getBin(unsigned int i) {
    if (i == 0) return 0;
    return sizeof(unsigned int)*8 - __builtin_clz(i);
  }

 public:

  typedef T value_type;
  template<bool newconcurrent>
  struct rethread {
    typedef LogOrderByIntegerMetric<T, Indexer, ContainerTy, newconcurrent> WL;
  };
  
  LogOrderByIntegerMetric(const Indexer& x = Indexer())
    :I(x), cursor(0)
  {
    for (int i = 0; i < cursor.size(); ++i)
      cursor.get(i) = 0;
  }
  
  bool push(value_type val) {   
    unsigned int index = I(val);
    index = getBin(index);
    data[index].push(val);
  }

  std::pair<bool, value_type> pop() {
    // print();
    unsigned int& cur = concurrent ? cursor.get() : cursor.get(0);
    std::pair<bool, value_type> ret = data[cur].pop();
    if (ret.first)
      return ret;

    //must move cursor
    for (cur = 0; cur < num(); ++cur) {
      ret = data[cur].pop();
      if (ret.first)
	return ret;
    }
    cur = 0;
    return std::pair<bool, value_type>(false, value_type());
  }

  std::pair<bool, value_type> try_pop() {
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

template<typename T, typename Indexer, typename LocalTy, typename GlobalTy>
class LocalFilter {
  GlobalTy globalQ;

  struct p {
    typename LocalTy::template rethread<false>::WL Q;
    unsigned int current;
  };
  PerCPU<p> localQs;
  Indexer I;

public:
  typedef T value_type;

  LocalFilter(const Indexer& x = Indexer()) : I(x) {
    for (int i = 0; i < localQs.size(); ++i)
      localQs.get(i).current = 0;
  }

    //! change the concurrency flag
  template<bool newconcurrent>
  struct rethread {
    typedef LocalFilter WL;
  };

  //! push a value onto the queue
  bool push(value_type val) OPTNOINLINE {
    unsigned int index = I(val);
    p& me = localQs.get();
    if (index <= me.current)
      return me.Q.push(val);
    else
      return globalQ.push(val);
  }

  //! push an aborted value onto the queue
  bool aborted(value_type val) {
    push(val);
  }

  //! pop a value from the queue.
  std::pair<bool, value_type> pop() OPTNOINLINE {
    std::pair<bool, value_type> r = localQs.get().Q.pop();
    if (r.first)
      return r;
    
    r = globalQ.pop();
    if (r.first)
      localQs.get().current = I(r.second);
    return r;
  }

  //! pop a value from the queue trying not as hard to take locks
  std::pair<bool, value_type> try_pop() OPTNOINLINE {
    return pop();
  }

  //! return if the queue *may* be empty
  bool empty() OPTNOINLINE {
    if (!localQs.get().Q.empty()) return false;
    return globalQ.empty();
  }
  
  //! called in sequential mode to seed the worklist
  template<typename iter>
  void fillInitial(iter begin, iter end) {
    globalQ.fillInitial(begin,end);
  }
};

//Queue per writer, reader cycles
template<typename T>
class MP_SC_FIFO {
  PerCPU<FIFO<T> > data;
  int cursor;
  
public:
  typedef T value_type;

  MP_SC_FIFO() :data(0), cursor(0) {}

  template<bool newconcurrent>
  struct rethread {
    typedef MP_SC_FIFO<T> WL;
  };

  bool push(value_type val) {
    return data.get().push(val);
  }

  bool aborted(value_type val) {
    return data.get().aborted(val);
  }

  std::pair<bool, value_type> pop() {
    //    ++cursor;
    //    cursor %= data.size();
    std::pair<bool, value_type> ret = data.get(cursor).pop();
    if (ret.first)
      return ret;
    for (int i = 0; i < data.size(); ++i) {
      ++cursor;
      cursor %= data.size();
      ret = data.get(cursor).pop();
      if (ret.first)
	return ret;
    }
    //failure
    return std::make_pair(false, value_type());
  }

  std::pair<bool, value_type> try_pop() {
    return pop();
  }

  bool empty() {
    for (int i = 0; i < data.size(); ++i)
      if (!data.get(i).empty())
	return false;
    return true;
  }

  
  //! called in sequential mode to seed the worklist
  template<typename iter>
  void fillInitial(iter begin, iter end) {
    while (begin != end)
      push(*begin++);
  }

};

//End namespace
}
}
