// Scalable Local worklists -*- C++ -*-
// This contains final (leaf) worklists.  Some support stealing, some don't
// All classes conform to:
// Galois::WorkList (for virtual push)
// T pop(bool&)
// bool empty()
// If they support stealing:
// T steal(bool&)
// bool canSteal

#include <queue>
#include <stack>

#include "Support/ThreadSafe/simple_lock.h"
#include "Support/PackedInt.h"

#include <boost/utility.hpp>
#include <boost/type_traits/is_member_function_pointer.hpp>

namespace GaloisRuntime {
namespace WorkList {

template<typename MQ, bool concurrent = true>
class STLAdaptor : private boost::noncopyable, private threadsafe::simpleLock<int, concurrent> {

  MQ wl;

  using threadsafe::simpleLock<int, concurrent>::lock;
  using threadsafe::simpleLock<int, concurrent>::unlock;

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
class ChunkedFIFO {
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
      assert(!lhs.next || rhs.next->empty());
      assert(!lhs.curr || rhs.curr->empty());
      assert(!rhs.next || rhs.next->empty());
      assert(!rhs.curr || rhs.curr->empty());
    }
  };

  CPUSpaced<ProcRec> data;

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

  ChunkedFIFO() :data(ProcRec::merge) {}

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
    if ( n.curr && !n.curr->empty()) return false;
    if (n.next && !n.next->empty()) return false;
    return Items.empty();
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


template<class T, class Indexer>
class OrderedByIntegerMetric {

  typedef FIFO<T> ContainerTy;

  ContainerTy* data;
  CPUSpaced<unsigned int> cursor;
  unsigned int size;
  Indexer I;

  static void merge(unsigned int& x, unsigned int& y) {
    x = 0;
    y = 0;
  }

 public:

  typedef T value_type;
  
  OrderedByIntegerMetric(unsigned int range, const Indexer& x = Indexer())
    :cursor(&merge), size(range+1), I(x)
  {
    data = new ContainerTy[size];
  }
  
  ~OrderedByIntegerMetric() {
    delete[] data;
  }

  void push(value_type val) {
    unsigned int index = I(val) % size; //Don't overflow if range is wrong
    data[index].push(val);
    unsigned int& cur = cursor.get();
    if (cur > index)
      cur = index;
  }

  std::pair<bool, value_type> pop() {
    unsigned int& cur = cursor.get();
    //Find a successful pop
    if (cur == size) //handle out of range
      cur = 0;

    std::pair<bool, value_type> ret;
    ret = data[cur].pop();
    while (cur < size && !ret.first) {
      ++cur;
      ret = data[cur].pop();
    }
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

}
}
