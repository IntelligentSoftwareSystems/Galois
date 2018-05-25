/**
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of XYZ License (a copy is located in
 * LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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
 */

#ifndef GALOIS_RUNTIME_WORKLISTALT_H
#define GALOIS_RUNTIME_WORKLISTALT_H

#include "galois/worklists/Simple.h"
#include "galois/substrate/CompilerSpecific.h"

namespace galois {
namespace worklists {

template<typename QueueTy>
galois::optional<typename QueueTy::value_type>
stealHalfInPackage(substrate::PerThreadStorage<QueueTy>& queues) {
  unsigned id = substrate::ThreadPool::getTID();
  unsigned pkg = substrate::ThreadPool::getPackage();
  unsigned num = galois::getActiveThreads();
  QueueTy* me = queues.getLocal();
  galois::optional<typename QueueTy::value_type> retval;

  //steal from this package
  //Having 2 loops avoids a modulo, though this is a slow path anyway
  auto& tp = substrate::getThreadPool();
  for (unsigned i = id + 1; i < num; ++i)
    if (tp.getPackage(i) == pkg)
      if ((retval = me->steal(*queues.getRemote(i), true, true)))
	return retval;
  for (unsigned i = 0; i < id; ++i)
    if (tp.getPackage(i) == pkg)
      if ((retval = me->steal(*queues.getRemote(i), true, true)))
	return retval;
  return retval;
}

template<typename QueueTy>
galois::optional<typename QueueTy::value_type>
stealRemote(substrate::PerThreadStorage<QueueTy>& queues) {
  unsigned id = substrate::ThreadPool::getTID();
  //  unsigned pkg = runtime::LL::getPackageForThread(id);
  unsigned num = galois::getActiveThreads();
  QueueTy* me = queues.getLocal();
  galois::optional<typename QueueTy::value_type> retval;

  //steal from this package
  //Having 2 loops avoids a modulo, though this is a slow path anyway
  for (unsigned i = id + 1; i < num; ++i)
    if ((retval = me->steal(*queues.getRemote(i), true, true)))
      return retval;
  for (unsigned i = 0; i < id; ++i)
    if ((retval = me->steal(*queues.getRemote(i), true, true)))
      return retval;
  return retval;
}

template<typename QueueTy>
class PerThreadQueues : private boost::noncopyable {
public:
  typedef typename QueueTy::value_type value_type;

private:
  substrate::PerThreadStorage<QueueTy> local;

  galois::optional<value_type> doSteal() {
    galois::optional<value_type> retval = stealHalfInPackage(local);
    if (retval)
      return retval;
    return stealRemote(local);
  }

  template<typename Iter>
  void fill_work_l2(Iter& b, Iter& e) {
    unsigned int a = galois::getActiveThreads();
    unsigned int id = substrate::ThreadPool::getTID();
    unsigned dist = std::distance(b, e);
    unsigned num = (dist + a - 1) / a; //round up
    unsigned int A = std::min(num * id, dist);
    unsigned int B = std::min(num * (id + 1), dist);
    e = b;
    std::advance(b, A);
    std::advance(e, B);
  }

  // runtime::LL::SimpleLock<true> L;
  // std::vector<unsigned> sum;

  template<typename Iter>
  void fill_work_l1(Iter b, Iter e) {
    Iter b2 = b;
    Iter e2 = e;
    fill_work_l2(b2, e2);
    unsigned int a = galois::getActiveThreads();
    //    unsigned int id = runtime::LL::getTID();
    std::vector<std::vector<value_type> > ranges;
    ranges.resize(a);
    while (b2 != e2) {
      unsigned i = getID(*b2);
      ranges[i].push_back(*b2);
      ++b2;
      if (ranges[i].size() > 128) {
	local.getRemote(i)->push(ranges[i].begin(), ranges[i].end());
	ranges[i].clear();
      }
    }
    // L.lock();
    // if (sum.empty())
    //   sum.resize(a + 1);
    // sum[a]++;
    // std::cerr << id << ":";
    // for (unsigned int x = 0; x < a; ++x) {
    //   std::cerr << " " << ranges[x].size();
    //   sum[x] += ranges[x].size();
    // }
    // std::cerr << "\n";
    // if (sum[a] == a) {
    //   std::cerr << "total:";
    //   for (unsigned int x = 0; x < a; ++x)
    // 	std::cerr << " " << sum[x];
    //   std::cerr << "\n";
    // }
    // L.unlock();
    for (unsigned int x = 0; x < a; ++x)
      if (!ranges[x].empty())
	local.getRemote(x)->push(ranges[x].begin(), ranges[x].end());
  }

public:
  template<typename Tnew>
  using retype = PerThreadQueues<typename QueueTy::template retype<Tnew>::type>;

  template<bool newConcurrent>
  using rethread = PerThreadQueues<typename QueueTy::template rethread<newConcurrent>::type>;

  void push(const value_type& val) {
    local.getLocal()->push(val);
  }

  template<typename Iter>
  void push(Iter b, Iter e) {
    local.getLocal()->push(b,e);
  }

  template<typename RangeTy>
  void push_initial(RangeTy range) {
    fill_work_l1(range.begin(), range.end());
  }

  galois::optional<value_type> pop() {
    galois::optional<value_type> retval = local.getLocal()->pop();
    if (retval)
      return retval;
    return doSteal();// stealHalfInPackage(local);
  }
};
//GALOIS_WLCOMPILECHECK(LocalQueues);

template<typename WLTy = GFIFO<int>, typename T = int>
class LocalWorklist : private boost::noncopyable {
  typedef typename WLTy::template rethread<false> lWLTy;
  substrate::PerThreadStorage<lWLTy> local;

public:
  template<bool newconcurrent>
  using rethread = LocalWorklist<typename WLTy::template rethread<newconcurrent>, T>;

  template<typename Tnew>
  using retype = LocalWorklist<typename WLTy::template retype<Tnew>, Tnew>;

  typedef T value_type;

  void push(const value_type& val) {
    local.getLocal()->push(val);
  }

  template<typename Iter>
  void push(Iter b, Iter e) {
    local.getLocal()->push(b, e);
  }

  template<typename RangeTy>
  void push_initial(RangeTy range) {
    local.getLocal()->push(range.local_begin(), range.local_end());
  }

  galois::optional<value_type> pop() {
    return local.getLocal()->pop();
  }
};
GALOIS_WLCOMPILECHECK(LocalWorklist)

template<typename T, typename OwnerFn, template<typename, bool> class QT, bool distributed = false, bool isStack = false, int chunksize=64, bool concurrent=true>
class OwnerComputeChunkedMaster : private boost::noncopyable {
  class Chunk : public galois::FixedSizeRing<T, chunksize>, public QT<Chunk, concurrent>::ListNode {};

  runtime::FixedSizeHeap heap;
  OwnerFn Fn;

  struct p {
    Chunk* cur;
    Chunk* next;
  };

  typedef QT<Chunk, concurrent> LevelItem;

  substrate::PerThreadStorage<p> data;
  internal::squeue<distributed, substrate::PerPackageStorage, LevelItem> Q;

  Chunk* mkChunk() {
    return new (heap.allocate(sizeof(Chunk))) Chunk();
  }

  void delChunk(Chunk* C) {
    C->~Chunk();
    heap.deallocate(C);
  }

  void pushChunk(Chunk* C)  {
    unsigned int tid = substrate::ThreadPool::getTID();
    unsigned int index = isStack ? Fn(C->back()) : Fn(C->front());
    if (tid == index) {
      LevelItem& I = Q.get();
      I.push(C);
    } else {
      unsigned int mindex = substrate::getThreadPool().getPackage(index);
      LevelItem& I = Q.get(mindex);
      I.push(C);
    }
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
  using rethread = OwnerComputeChunkedMaster<T, OwnerFn,QT, distributed, isStack, chunksize, newconcurrent>;

  template<typename Tnew>
  using retype = OwnerComputeChunkedMaster<Tnew, OwnerFn, QT, distributed, isStack, chunksize, concurrent>;

  OwnerComputeChunkedMaster() : heap(sizeof(Chunk)) { }

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

  galois::optional<value_type> pop()  {
    p& n = *data.getLocal();
    galois::optional<value_type> retval;
    if (isStack) {
      if (n.next && (retval = n.next->extract_back()))
	return retval;
      if (n.next)
	delChunk(n.next);
      n.next = popChunk();
      if (n.next)
	return n.next->extract_back();
      return galois::optional<value_type>();
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
      return galois::optional<value_type>();
    }
  }
};

template<typename OwnerFn=DummyIndexer<int> , int chunksize=64, typename T = int, bool concurrent=true>
class OwnerComputeChunkedLIFO : public OwnerComputeChunkedMaster<T,OwnerFn,ConExtLinkedQueue, true, true, chunksize, concurrent> {};
GALOIS_WLCOMPILECHECK(OwnerComputeChunkedLIFO)


}//End namespace
} // end namespace galois

#endif
