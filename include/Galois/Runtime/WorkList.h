// Scalable Local worklists -*- C++ -*-
// This contains final worklists.

#ifndef __WORKLIST_H_
#define __WORKLIST_H_

#include <queue>
#include <stack>
#include <limits>
#include <map>
#include <boost/utility.hpp>

#include "Galois/Runtime/PaddedLock.h"
#include "Galois/Runtime/PerCPU.h"
//#include "Galois/Runtime/QueuingLock.h"

#include <boost/utility.hpp>

#include "mm/mem.h"
#include "WorkListHelpers.h"

#define OPTNOINLINE __attribute__((noinline)) 
//#define OPTNOINLINE

#ifndef WLCOMPILECHECK
#define WLCOMPILECHECK(name) //
#endif

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
  //! return if the queue is empty
  //! *not thread safe*
  bool empty() const;
  
  //! called in sequential mode to seed the worklist
  template<typename iter>
  void fill_initial(iter begin, iter end);


};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


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
   
  bool empty() const {
    return wl.empty();
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
WLCOMPILECHECK(PriQueue);

template<typename T, bool concurrent = true>
class LIFO : private boost::noncopyable, private PaddedLock<concurrent> {
  std::deque<T> wl;

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

  bool empty() const OPTNOINLINE {
    return wl.empty();
  }

  bool aborted(value_type val) {
    return push(val);
  }

  //Not Thread Safe
  template<typename Iter>
  void fill_initial(Iter ii, Iter ee) {
    while (ii != ee) {
      wl.push_back(*ii++);
    }
  }
};
WLCOMPILECHECK(LIFO);

template<typename T, bool concurrent = true>
class FIFO : private boost::noncopyable, private PaddedLock<concurrent>  {
  std::deque<T> wl;

  using PaddedLock<concurrent>::lock;
  using PaddedLock<concurrent>::try_lock;
  using PaddedLock<concurrent>::unlock;

public:
  template<bool newconcurrent>
  struct rethread {
    typedef FIFO<T, newconcurrent> WL;
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

  bool empty() const {
    return wl.empty();
  }

  bool aborted(value_type val) {
    return push(val);
  }

  //Not Thread Safe
  template<typename Iter>
  void fill_initial(Iter ii, Iter ee) {
    while (ii != ee) {
      wl.push_back(*ii++);
    }
  }
};
WLCOMPILECHECK(FIFO);

template<typename T, int chunksize=64, bool concurrent=true>
class ChunkedFIFO : private boost::noncopyable {
  class Chunk : public FixedSizeRing<T, chunksize, false>, public ConExtListNode<Chunk> { };

  MM::FixedSizeAllocator heap;

  struct p {
    Chunk* cur;
    Chunk* next;
  };

  PerCPU<p> data;
  ConExtLinkedQueue<Chunk, concurrent> queue;

  Chunk* mkChunk() {
    return new (heap.allocate(sizeof(Chunk))) Chunk();
  }

  void delChunk(Chunk* C) {
    C->~Chunk();
    heap.deallocate(C);
  }

public:
  template<bool newconcurrent>
  struct rethread {
    typedef ChunkedFIFO<T, chunksize, newconcurrent> WL;
  };

  typedef T value_type;
  
  ChunkedFIFO() : heap(sizeof(Chunk)) {
    for (int i = 0; i < data.size(); ++i) {
      p& r = data.get(i);
      r.next = 0;
      r.cur = 0;
    }
  }

  ~ChunkedFIFO() {
    for (int i = 0; i < data.size(); ++i) {
      p& r = data.get(i);
      if (r.next)
	delChunk(r.next);
      if (r.cur)
	delChunk(r.cur);
      r.next = r.cur = 0;
    }
    while (Chunk* C = queue.pop())
      delChunk(C);
  }

  bool push(value_type val) {
    p& n = data.get();
    if (n.next && n.next->push_back(val))
      return true;
    if (n.next)
      queue.push(n.next);
    n.next = mkChunk();
    bool worked = n.next->push_back(val);
    assert(worked);
    return true;
  }

  std::pair<bool, value_type> pop() {
    p& n = data.get();
    std::pair<bool, value_type> retval;
    if (n.cur && (retval = n.cur->pop_front()).first)
      return retval;
    if(n.cur)
      delChunk(n.cur);
    n.cur = queue.pop();
    if (!n.cur) {
      n.cur = n.next;
      n.next = 0;
    }
    if (n.cur)
      return n.cur->pop_front();
    return std::make_pair(false, value_type());
  }
  
  bool empty() const {
    for (int i = 0; i < data.size(); ++i) {
      const p& n = data.get(i);
      if (n.cur && !n.cur->empty()) return false;
      if (n.next && !n.next->empty()) return false;
    }
    return queue.empty();
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
    if (n.next) {
      queue.push(n.next);
      n.next = 0;
    }
  }
};
WLCOMPILECHECK(ChunkedFIFO);

template<typename T, int chunksize=64, bool concurrent=true>
class ChunkedLIFO : private boost::noncopyable {
  class Chunk : public FixedSizeRing<T, chunksize, false>, public ConExtListNode<Chunk> {};

  MM::FixedSizeAllocator heap;

  struct p {
    Chunk* cur;
  };

  PerCPU<p> data;
  ConExtLinkedStack<Chunk, concurrent> stack;

  Chunk* mkChunk() {
    return new (heap.allocate(sizeof(Chunk))) Chunk();
  }

  void delChunk(Chunk* C) {
    C->~Chunk();
    heap.deallocate(C);
  }

public:
  template<bool newconcurrent>
  struct rethread {
    typedef ChunkedLIFO<T, chunksize, newconcurrent> WL;
  };

  typedef T value_type;
  
  ChunkedLIFO() : heap(sizeof(Chunk)) {
    for (int i = 0; i < data.size(); ++i)
      data.get(i).cur = 0;
  }

  ~ChunkedLIFO() {
    for (int i = 0; i < data.size(); ++i) {
      p& r = data.get(i);
      if (r.cur)
	delChunk(r.cur);
      r.cur = 0;
    }
    while (Chunk* C = stack.pop())
      delChunk(C);
  }

  bool push(value_type val) {
    p& n = data.get();
    if (n.cur && n.cur->push_back(val))
      return true;
    if (n.cur)
      stack.push(n.cur);
    n.cur = mkChunk();
    bool worked = n.cur->push_back(val);
    assert(worked);
    return true;
  }

  std::pair<bool, value_type> pop() {
    p& n = data.get();
    std::pair<bool, value_type> retval;
    if (n.cur && (retval = n.cur->pop_front()).first)
      return retval;
    if (n.cur)
      delChunk(n.cur);
    n.cur = stack.pop();
    if (n.cur)
      return n.cur->pop_front();
    return std::make_pair(false, value_type());
  }
  
  bool empty() {
    for (int i = 0; i < data.size(); ++i) {
      p& n = data.get(i);
      if (n.cur && !n.cur->empty()) return false;
    }
    return stack.empty();
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
    if (n.cur) {
      stack.push(n.cur);
      n.cur = 0;
    }
  }
};
WLCOMPILECHECK(ChunkedLIFO);

template<class T, class Indexer = DummyIndexer<T>, typename ContainerTy = FIFO<T>, bool concurrent = true >
class OrderedByIntegerMetric : private boost::noncopyable {

  typedef typename ContainerTy::template rethread<concurrent>::WL CTy;

  struct perItem {
    CTy* current;
    std::map<int, CTy*> local;
  };

  std::map<int, CTy*> master;
  SimpleLock<int, concurrent> masterLock;

  Indexer I;

  PerCPU<perItem> current;

 public:
  template<bool newconcurrent>
  struct rethread {
    typedef  OrderedByIntegerMetric<T,Indexer,ContainerTy,newconcurrent> WL;
  };

  typedef T value_type;

  OrderedByIntegerMetric(const Indexer& x = Indexer())
    :I(x)
  {
    for (int i = 0; i < current.size(); ++i)
      current.get(i).current = 0;
  }

  ~OrderedByIntegerMetric() {
    for (typename std::map<int, CTy*>::iterator ii = master.begin(), ee = master.end(); ii != ee; ++ii)
      delete ii->second;
  }
  
  bool push(value_type val) {
    unsigned int index = I(val);
    perItem& pI = current.get();
    CTy*& lC = pI.local[index];
    if (!lC) {
      masterLock.lock();
      CTy*& gC = master[index];
      if (!gC)
	gC = new CTy();
      lC = gC;
      masterLock.unlock();
    }
    lC->push(val);
  }

  std::pair<bool, value_type> pop() {
    //Find a successful pop
    perItem& pI = current.get();
    CTy*& C = pI.current;
    std::pair<bool, value_type> retval;
    if (C && (retval = C->pop()).first)
      return retval;
    //Failed, find minimum bin
    masterLock.lock();
    for (typename std::map<int, CTy*>::iterator ii = master.begin(), ee = master.end(); ii != ee; ++ii) {
      C = ii->second;
      if ((retval = C->pop()).first) {
	masterLock.unlock();
	return retval;
      }
    }
    masterLock.unlock();
    retval.first = false;
    return retval;
  }

  bool empty() const {
    for (typename std::map<int, CTy*>::const_iterator ii = master.begin(), ee = master.end(); ii != ee; ++ii)
      if (!ii->second->empty())
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
WLCOMPILECHECK(OrderedByIntegerMetric);

template<class T, typename ContainerTy = FIFO<T> >
class StealingLocalWL : private boost::noncopyable {

  PerCPU<ContainerTy> data;

  // static void merge(ContainerTy& x, ContainerTy& y) {
  //   assert(x.empty());
  //   assert(y.empty());
  // }

 public:
  template<bool newconcurrent>
  struct rethread {
    typedef StealingLocalWL<T, ContainerTy> WL;
  };

  typedef T value_type;
  
  StealingLocalWL() {}

  bool push(value_type val) {
    return data.get().push(val);
  }

  std::pair<bool, value_type> pop() {
    std::pair<bool, value_type> ret = data.get().pop();
    if (ret.first)
      return ret;
    return data.getNext().pop();
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
WLCOMPILECHECK(StealingLocalWL);

template<typename T, typename GlobalQueueTy = FIFO<T>, typename LocalQueueTy = FIFO<T> >
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
    return local.get().push(val);
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

  bool empty() {
    if (!local.get().empty()) return false;
    return global.empty();
  }

  template<typename iter>
  void fill_initial(iter begin, iter end) {
    while (begin != end)
      global.push(*begin++);
  }
};
WLCOMPILECHECK(LocalQueues);

template<class T, class Indexer = DummyIndexer<T>, typename ContainerTy = FIFO<T>, bool concurrent=true >
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
    return data[index].push(val);
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
      ret = data[cur].pop();
      if (ret.first)
	return ret;
    }
    return std::pair<bool, value_type>(false, value_type());
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
WLCOMPILECHECK(ApproxOrderByIntegerMetric);

template<class T, class Indexer = DummyIndexer<T>, typename ContainerTy = FIFO<T>, bool concurrent=true >
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
    :I(x)
  {
    for (int i = 0; i < cursor.size(); ++i)
      cursor.get(i) = 0;
  }
  
  bool push(value_type val) {   
    unsigned int index = I(val);
    index = getBin(index);
    return data[index].push(val);
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
WLCOMPILECHECK(LogOrderByIntegerMetric);

template<typename T, typename Indexer = DummyIndexer<T>, typename LocalTy = FIFO<T>, typename GlobalTy = FIFO<T> >
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
    return push(val);
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

  //! return if the queue *may* be empty
  bool empty() OPTNOINLINE {
    if (!localQs.get().Q.empty()) return false;
    return globalQ.empty();
  }
  
  //! called in sequential mode to seed the worklist
  template<typename iter>
  void fill_initial(iter begin, iter end) {
    globalQ.fill_initial(begin,end);
  }
};
WLCOMPILECHECK(LocalFilter);

//Queue per writer, reader cycles
template<typename T>
class MP_SC_FIFO {
  PerCPU<FIFO<T> > data;
  int cursor;
  
public:
  typedef T value_type;

  MP_SC_FIFO() :cursor(0) {}

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

  bool empty() {
    for (int i = 0; i < data.size(); ++i)
      if (!data.get(i).empty())
	return false;
    return true;
  }

  
  //! called in sequential mode to seed the worklist
  template<typename iter>
  void fill_initial(iter begin, iter end) {
    while (begin != end)
      push(*begin++);
  }

};
WLCOMPILECHECK(MP_SC_FIFO);

#if 0
//Bag per writer, reader steals entire bag
template<typename T, int chunksize = 64>
class MP_SC_Bag {
  class Chunk : public FixedSizeRing<T, chunksize, false>, public ConExtListNode<Chunk>::ListNode {};

  MM::FixedSizeAllocator heap;

  PerCPU<PtrLock<Chunk*, true> > write_stack;

  ConExtLinkedStack<Chunk, true> read_stack;
  Chunk* current;

public:
  typedef T value_type;

  template<bool newconcurrent>
  struct rethread {
    typedef MP_SC_Bag<T, chunksize> WL;
  };

  MP_SC_Bag() :heap(sizeof(Chunk)), current(0) {}

  bool push(value_type val) {
    PtrLock<Chunk*, true>& L = write_stack.get();
    L.lock();
    Chunk* OldL = L.getValue();
    if (OldL && OldL->push_back(val)) {
      L.unlock();
      return true;
    }
    Chunk* nc = new (heap.allocate(sizeof(Chunk))) Chunk();
    bool retval = nc->push_back(val);
    assert(retval);
    L.unlock_and_set(nc);
    if (OldL)
      read_stack.push(OldL);
    return true;
  }

  bool aborted(value_type val) {
    return push(val);
  }

  std::pair<bool, value_type> pop() {
    //Read stack
    if (!current)
      current = read_stack.pop();
    if (current)
      std::pair<bool, value_type> ret = current->pop_front();
    if (ret.first)
      return ret;
    //try taking from our write queue
    read_stack.steal(write_stack.get());
    ret = read_stack.pop();
    if (ret.first)
      return ret;
    //try stealing from everywhere
    for (int i = 0; i < write_stack.size(); ++i) {
      read_stack.steal(write_stack.get(i));
    }
    return read_stack.pop();
  }

  bool empty() {
    if (!read_stack.empty()) return false;
    for (int i = 0; i < write_stack.size(); ++i)
      if (!write_stack.get(i).empty())
	return false;
    return true;
  }

  
  //! called in sequential mode to seed the worklist
  template<typename iter>
  void fill_initial(iter begin, iter end) {
    while (begin != end)
      push(*begin++);
  }

};
WLCOMPILECHECK(MP_SC_Bag);

#endif
//End namespace
}
}

#endif
