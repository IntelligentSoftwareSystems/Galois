// Scalable Local worklists -*- C++ -*-
// This contains final (leaf) worklists.  Some support stealing, some don't
// All classes conform to:
// void push(T)
// bool,T pop()
// bool empty()
// void fillInitial(begin, end)

#include <queue>
#include <stack>

#include "Galois/Runtime/SimpleLock.h"

#include <boost/utility.hpp>

namespace GaloisRuntime {
namespace WorkList {

template<typename MQ, bool concurrent = true>
class STLAdaptor : private boost::noncopyable, private SimpleLock<long, concurrent> {

  MQ wl;

  using SimpleLock<long, concurrent>::lock;
  using SimpleLock<long, concurrent>::unlock;

public:
  typedef STLAdaptor<MQ, true>  ConcurrentTy;
  typedef STLAdaptor<MQ, false> SingleThreadTy;

  typedef typename MQ::value_type value_type;

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
    
  bool empty() {
    lock();
    bool retval = wl.empty();
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

template<typename Q>
class stdqueueFix : public Q {
public:
  typename Q::value_type top() {
    return Q::front();
  }
};

template<typename T, bool concurrent = true>
class LIFO : public STLAdaptor<std::stack<T>, concurrent > {};

template<typename T, bool concurrent = true>
class FIFO : public STLAdaptor<stdqueueFix<std::queue<T> >, concurrent > {};

template<typename T, class Compare = std::less<T>, bool concurrent = true >
class PriQueue : public STLAdaptor<std::priority_queue<T, std::vector<T>, Compare>, concurrent > {};


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
    assert(data.getCount() > 1);
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

 public:

  typedef T value_type;
  
  OrderedByIntegerMetric(unsigned int range, const Indexer& x = Indexer())
    :size(range+1), I(x), cursor(&merge)
  {
    data = new ContainerTy[size];
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
    unsigned int& cur = cursor.get();
    //Find a successful pop
    if (cur == size) //handle out of range
      cur = 0;

    std::pair<bool, value_type> ret;
    do {
      ret = data[cur].pop();
      if (ret.first)
	return ret;
      ++cur;
    } while (cur < size);
    return ret;
  }

  bool empty() const {
    for (unsigned int i = 0; i < size; ++i)
      if (!data[i].empty())
	return false;
    return true;
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


}
}
