/** Scalable local worklists -*- C++ -*-
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
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#ifndef GALOIS_RUNTIME_WORKLIST_H
#define GALOIS_RUNTIME_WORKLIST_H

#include "Galois/Runtime/PerThreadStorage.h"
#include "Galois/Runtime/Barrier.h"
#include "Galois/Runtime/ThreadPool.h"
#include "Galois/WorkList/WorkListHelpers.h"
#include "Galois/Runtime/ll/PaddedLock.h"
#include "Galois/Runtime/ll/gio.h"
#include "Galois/Runtime/mm/Mem.h"

#include "Galois/gdeque.h"
#include "Galois/FixedSizeRing.h"
#include "Galois/util/GAlgs.h"

#include <limits>
#include <iterator>
#include <map>
#include <vector>
#include <deque>
#include <algorithm>
#include <iterator>
#include <utility>

#include <boost/utility.hpp>
#include <boost/optional.hpp>
#include <boost/ref.hpp>


#include "Lifo.h"
#include "Fifo.h"
#include "GFifo.h"
#include "LocalQueues.h"
#include "Obim.h"

namespace Galois {
namespace WorkList {

// Worklists may not be copied.
// Worklists should be default instantiatable
// All classes (should) conform to:
template<typename T, bool concurrent>
class AbstractWorkList {
  AbstractWorkList(const AbstractWorkList&);
  const AbstractWorkList& operator=(const AbstractWorkList&);

public:
  AbstractWorkList() { }

  //! T is the value type of the WL
  typedef T value_type;

  //! change the concurrency flag
  template<bool newconcurrent>
  using rethread = AbstractWorkList<T, newconcurrent>;

  //! change the type the worklist holds
  template<typename Tnew>
  using retype = AbstractWorkList<Tnew, concurrent>;

  //! push a value onto the queue
  void push(const value_type& val) { abort(); }

  //! push a range onto the queue
  template<typename Iter>
  void push(Iter b, Iter e) { abort(); }

  //! push initial range onto the queue
  //! called with the same b and e on each thread
  template<typename RangeTy>
  void push_initial(RangeTy) { abort(); }

  //Optional, but this is the likely interface for stealing
  //! steal from a similar worklist
  boost::optional<value_type> steal(AbstractWorkList& victim, bool half, bool pop);

  //! pop a value from the queue.
  boost::optional<value_type> pop() { abort(); }
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


//This overly complex specialization avoids a pointer indirection for non-distributed WL when accessing PerLevel
template<bool d, typename TQ>
struct squeues;

template<typename TQ>
struct squeues<true,TQ> {
  Runtime::PerPackageStorage<TQ> queues;
  TQ& get(int i) { return *queues.getRemote(i); }
  TQ& get() { return *queues.getLocal(); }
  int myEffectiveID() { return Runtime::LL::getTID(); }
  int size() { return Runtime::galoisActiveThreads; }
};

template<typename TQ>
struct squeues<false,TQ> {
  TQ queue;
  TQ& get(int i) { return queue; }
  TQ& get() { return queue; }
  int myEffectiveID() { return 0; }
  int size() { return 0; }
};

template<typename T, template<typename, bool> class QT, bool distributed = false, bool isStack = false, int chunksize=64, bool concurrent=true>
class ChunkedMaster : private boost::noncopyable {
  class Chunk : public Galois::FixedSizeRing<T, chunksize>, public QT<Chunk, concurrent>::ListNode {};

  Runtime::MM::FixedSizeAllocator heap;

  struct p {
    Chunk* cur;
    Chunk* next;
  };

  typedef QT<Chunk, concurrent> LevelItem;

  Runtime::PerThreadStorage<p> data;
  squeues<distributed, LevelItem> Q;

  Chunk* mkChunk() {
    return new (heap.allocate(sizeof(Chunk))) Chunk();
  }
  
  void delChunk(Chunk* C) {
    C->~Chunk();
    heap.deallocate(C);
  }

  void pushChunk(Chunk* C)  {
    LevelItem& I = Q.get();
    I.push(C);
  }

  Chunk* popChunkByID(unsigned int i)  {
    LevelItem& I = Q.get(i);
    return I.pop();
  }

  Chunk* popChunk()  {
    int id = Q.myEffectiveID();
    Chunk* r = popChunkByID(id);
    if (r)
      return r;

    for (int i = id + 1; i < (int) Q.size(); ++i) {
      r = popChunkByID(i);
      if (r) 
	return r;
    }

    for (int i = 0; i < id; ++i) {
      r = popChunkByID(i);
      if (r)
	return r;
    }

    return 0;
  }

  T* pushi(const T& val, p* n)  {
    T* retval = 0;

    if (n->next && (retval = n->next->push_back(val)))
      return retval;
    if (n->next)
      pushChunk(n->next);
    n->next = mkChunk();
    retval = n->next->push_back(val);
    assert(retval);
    return retval;
  }

public:
  typedef T value_type;

  template<bool newconcurrent>
  using rethread = ChunkedMaster<T, QT, distributed, isStack, chunksize, newconcurrent>;
  template<typename Tnew>
  using retype = ChunkedMaster<Tnew, QT, distributed, isStack, chunksize, concurrent>;

  ChunkedMaster() : heap(sizeof(Chunk)) { }

  void flush() {
    p& n = *data.getLocal();
    if (n.next)
      pushChunk(n.next);
    n.next = 0;
  }
  
  //! Most worklists have void return value for push. This push returns address
  //! of placed item to facilitate some internal runtime uses. The address is
  //! generally not safe to use in the presence of concurrent pops.
  value_type* push(const value_type& val)  {
    p* n = data.getLocal();
    return pushi(val, n);
  }

  template<typename Iter>
  void push(Iter b, Iter e) {
    p* n = data.getLocal();
    while (b != e)
      pushi(*b++, n);
  }

  template<typename RangeTy>
  void push_initial(RangeTy range) {
    push(range.local_begin(), range.local_end());
  }

  boost::optional<value_type> pop()  {
    p& n = *data.getLocal();
    boost::optional<value_type> retval;
    if (isStack) {
      if (n.next && (retval = n.next->extract_back()))
	return retval;
      if (n.next)
	delChunk(n.next);
      n.next = popChunk();
      if (n.next)
	return n.next->extract_back();
      return boost::optional<value_type>();
    } else {
      if (n.cur && (retval = n.cur->extract_front()))
	return retval;
      if (n.cur)
	delChunk(n.cur);
      n.cur = popChunk();
      if (!n.cur) {
	n.cur = n.next;
	n.next = 0;
      }
      if (n.cur)
	return n.cur->extract_front();
      return boost::optional<value_type>();
    }
  }
};

template<int chunksize=64, typename T = int, bool concurrent=true>
class ChunkedFIFO : public ChunkedMaster<T, ConExtLinkedQueue, false, false, chunksize, concurrent> {};
GALOIS_WLCOMPILECHECK(ChunkedFIFO)

template<int chunksize=64, typename T = int, bool concurrent=true>
class ChunkedLIFO : public ChunkedMaster<T, ConExtLinkedStack, false, true, chunksize, concurrent> {};
GALOIS_WLCOMPILECHECK(ChunkedLIFO)

template<int chunksize=64, typename T = int, bool concurrent=true>
class dChunkedFIFO : public ChunkedMaster<T, ConExtLinkedQueue, true, false, chunksize, concurrent> {};
GALOIS_WLCOMPILECHECK(dChunkedFIFO)

template<int chunksize=64, typename T = int, bool concurrent=true>
class dChunkedLIFO : public ChunkedMaster<T, ConExtLinkedStack, true, true, chunksize, concurrent> {};
GALOIS_WLCOMPILECHECK(dChunkedLIFO)

template<typename OwnerFn=DummyIndexer<int>, typename WLTy=ChunkedLIFO<256>, typename T = int>
class OwnerComputesWL : private boost::noncopyable {
  typedef typename WLTy::template retype<T> lWLTy;

  typedef lWLTy cWL;
  typedef lWLTy pWL;

  OwnerFn Fn;
  Runtime::PerPackageStorage<cWL> items;
  Runtime::PerPackageStorage<pWL> pushBuffer;

public:
  template<bool newconcurrent>
  using rethread = OwnerComputesWL<OwnerFn,typename WLTy::template rethread<newconcurrent>, T>;
  template<typename Tnew>
  using retype = OwnerComputesWL<OwnerFn,typename WLTy::template retype<Tnew>,Tnew>;

  typedef T value_type;

  void push(const value_type& val)  {
    unsigned int index = Fn(val);
    unsigned int tid = Runtime::LL::getTID();
    unsigned int mindex = Runtime::LL::getPackageForThread(index);
    //std::cerr << "[" << index << "," << index % active << "]\n";
    if (mindex == Runtime::LL::getPackageForSelf(tid))
      items.getLocal()->push(val);
    else
      pushBuffer.getRemote(mindex)->push(val);
  }

  template<typename ItTy>
  void push(ItTy b, ItTy e) {
    while (b != e)
      push(*b++);
  }

  template<typename RangeTy>
  void push_initial(RangeTy range) {
    push(range.local_begin(), range.local_end());
    for (unsigned int x = 0; x < pushBuffer.size(); ++x)
      pushBuffer.getRemote(x)->flush();
  }

  boost::optional<value_type> pop() {
    cWL& wl = *items.getLocal();
    boost::optional<value_type> retval = wl.pop();
    if (retval)
      return retval;
    pWL& p = *pushBuffer.getLocal();
    while ((retval = p.pop()))
      wl.push(*retval);
    return wl.pop();
  }
};
GALOIS_WLCOMPILECHECK(OwnerComputesWL)

template<class ContainerTy=dChunkedFIFO<>, class T=int, bool concurrent = true>
class BulkSynchronous : private boost::noncopyable {

  typedef typename ContainerTy::template rethread<concurrent> CTy;

  struct TLD {
    unsigned round;
    TLD(): round(0) { }
  };

  CTy wls[2];
  Galois::Runtime::PerThreadStorage<TLD> tlds;
  Galois::Runtime::GBarrier barrier1;
  Galois::Runtime::GBarrier barrier2;
  Galois::Runtime::LL::CacheLineStorage<volatile long> some;
  volatile bool empty;

 public:
  typedef T value_type;

  template<bool newconcurrent>
  using rethread = BulkSynchronous<ContainerTy,T,newconcurrent>;
  template<typename Tnew>
  using retype = BulkSynchronous<typename ContainerTy::template retype<Tnew>,Tnew,concurrent>;

  BulkSynchronous(): empty(false) {
    unsigned num = Runtime::galoisActiveThreads;
    barrier1.reinit(num);
    barrier2.reinit(num);
  }

  void push(const value_type& val) {
    wls[(tlds.getLocal()->round + 1) & 1].push(val);
  }

  template<typename ItTy>
  void push(ItTy b, ItTy e) {
    while (b != e)
      push(*b++);
  }

  template<typename RangeTy>
  void push_initial(RangeTy range) {
    push(range.local_begin(), range.local_end());
    tlds.getLocal()->round = 1;
    some.data = true;
  }

  boost::optional<value_type> pop() {
    TLD& tld = *tlds.getLocal();
    boost::optional<value_type> r;
    
    while (true) {
      if (empty)
        return r; // empty

      r = wls[tld.round].pop();
      if (r)
        return r;

      barrier1.wait();
      if (Galois::Runtime::LL::getTID() == 0) {
        if (!some.data)
          empty = true;
        some.data = false; 
      }
      tld.round = (tld.round + 1) & 1;
      barrier2.wait();

      r = wls[tld.round].pop();
      if (r) {
        some.data = true;
        return r;
      }
    }
  }
};
GALOIS_WLCOMPILECHECK(BulkSynchronous)

//End namespace

} // end namespace WorkList
} // end namespace Galois

#endif

