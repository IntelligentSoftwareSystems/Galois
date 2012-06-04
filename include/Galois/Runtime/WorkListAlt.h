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
#ifndef GALOIS_RUNTIME_WORKLISTALT_H
#define GALOIS_RUNTIME_WORKLISTALT_H

#include "ll/CompilerSpecific.h"

namespace GaloisRuntime {
namespace WorkList {

struct ChunkHeader {
  ChunkHeader* next;
  ChunkHeader* prev;
};

class AtomicChunkDequeue {

  LL::PtrLock<ChunkHeader, true> head;
  ChunkHeader* volatile tail;

public:

  void push_front(ChunkHeader* obj) {
    head.lock();
    obj->prev = 0;
    obj->next = head.getValue();
    if (obj->next)
      obj->next->prev = obj;
    if (!tail)
      tail = obj;
    head.unlock_and_set(obj);
  }

  void push_back(ChunkHeader* obj) {
    head.lock();
    obj->next = 0;
    obj->prev = tail;
    if (obj->prev)
      obj->prev->next = obj;
    tail = obj;
    if (head.getValue())
      head.unlock();
    else
      head.unlock_and_set(obj);
  }

  ChunkHeader* pop_back() {
    //lockfree empty fast path
    if (!tail) return 0;
    head.lock();
    ChunkHeader* retval = tail;
    if (retval) {
      if (retval->prev)
	retval->prev->next = 0;
      tail = retval->prev;
      if (head.getValue() == retval)
	head.unlock_and_clear();
      else
	head.unlock();
      //clean up obj
      retval->prev = retval->next = 0;
      return retval;
    } else {
      head.unlock();
      return 0;
    }
  }

  ChunkHeader* pop_front() {
    //lockfree empty fast path
    if (!tail) return 0; //tail is just as useful as head
    head.lock();
    ChunkHeader* retval = head.getValue();
    if (retval) {
      if (retval->next)
	retval->next->prev = 0;
      if (tail == retval)
	tail = 0;
      head.unlock_and_set(retval->next);
      //clean up obj
      retval->prev = retval->next = 0;
      return retval;
    } else {
      head.unlock();
      return 0;
    }
  }
};

class AtomicChunkLIFO {

  LL::PtrLock<ChunkHeader, true> head;

  void prepend(ChunkHeader* C) {
    //Find tail of stolen stuff
    ChunkHeader* tail = C;
    while (tail->next) { tail = tail->next; }
    head.lock();
    tail->next = head.getValue();
    head.unlock_and_set(C);
  }

public:

  bool empty() const {
    return !head.getValue();
  }

  void push(ChunkHeader* obj) {
    ChunkHeader* oldhead = 0;
    do {
      oldhead = head.getValue();
      obj->next = oldhead;
    } while (!head.CAS(oldhead, obj));
  }

  ChunkHeader* pop() {
    //lock free Fast empty path
    if (empty()) return 0;

    //Disable CAS
    head.lock();
    ChunkHeader* retval = head.getValue();
    ChunkHeader* setval = 0;
    if (retval) {
      setval = retval->next;
      retval->next = 0;
    }
    head.unlock_and_set(setval);
    return retval;
  }

  ChunkHeader* stealAllAndPop(AtomicChunkLIFO& victim) {
    //Don't do work on empty victims (lockfree check)
    if (victim.empty()) return 0;
    //Steal everything
    victim.head.lock();
    ChunkHeader* C = victim.head.getValue();
    victim.head.unlock_and_clear();
    if (!C) return 0; //Didn't get anything
    ChunkHeader* retval = C;
    C = C->next;
    retval->next = 0;
    if (!C) return retval; //Only got one thing
    prepend(C);
    return retval;
  }

  ChunkHeader* stealHalfAndPop(AtomicChunkLIFO& victim) {
    //Don't do work on empty victims (lockfree check)
    if (victim.empty()) return 0;
    //Steal half
    victim.head.lock();
    ChunkHeader* C = victim.head.getValue();
    ChunkHeader* ntail = C;
    bool count = false;
    while (C) { 
      C = C->next;
      if (count)
	ntail = ntail->next;
      count = !count;
    }
    if (ntail) {
      C = ntail->next;
      ntail->next = 0;
    }
    victim.head.unlock();
    if (!C) return 0; //Didn't get anything
    ChunkHeader* retval = C;
    C = C->next;
    retval->next = 0;
    if (!C) return retval; //Only got one thing
    prepend(C);
    return retval;
  }

};


class StealingQueues : private boost::noncopyable {
  PerThreadStorage<std::pair<AtomicChunkLIFO, unsigned> > local;
  //PerThreadStorage<bool> LocalStarving;
  
  volatile unsigned Starving;
  AtomicChunkLIFO global;

  GALOIS_ATTRIBUTE_NOINLINE
  ChunkHeader* doSteal() {
    
    std::pair<AtomicChunkLIFO, unsigned>& me = *local.getLocal();
    unsigned id = LL::getTID();
    unsigned pkg = LL::getPackageForThread(id);
    unsigned num = ThreadPool::getActiveThreads();

    ChunkHeader* c = 0;

    //First steal from this package
    for (unsigned i = 1; i < num; ++i) {
      unsigned eid = (id + i) % num;
      if (LL::getPackageForThreadInternal(eid) == pkg) {
	ChunkHeader* c = me.first.stealHalfAndPop(local.getRemote(eid)->first);
	if (c)
	  return c;
	  //	  goto stex;
      }
    }
#if 0
    //Then try the global queue
    c = global.pop();
    if (c) 
      goto stex;

    //Leaders signal starvation
    if (LL::isLeaderForPackage(id)) {
      bool& sig = *LocalStarving.getLocal();
      if (!sig) {
	std::cerr << id << " Starving: " << Starving << "\n";
	__sync_fetch_and_add(&Starving, 1);
	sig = true;
      }
    }

  stex:
    if (LL::isLeaderForPackage(id) && c) {
      bool& sig = *LocalStarving.getLocal();
      if (sig) {
	std::cerr << id << " Not starving: " << Starving - 1 << "\n";
	__sync_fetch_and_sub(&Starving, 1);
	sig = false;
      }
    }
    return c;
#endif
    //Leaders can cross package
    if (LL::isLeaderForPackage(id)) {
      unsigned eid = (id + me.second) % num;
      ++me.second;
      if (id != eid && LL::isLeaderForPackageInternal(eid)) {
	ChunkHeader* c = me.first.stealAllAndPop(local.getRemote(eid)->first);
	if (c)
	  return c;
      }
    }
    return 0;
  }

public:
  StealingQueues() {} // :Starving(0) {}

  void push(ChunkHeader* c) {
    local.getLocal()->first.push(c);
  }
  ChunkHeader* pop() {
    // ChunkHeader* c = 0;
    // if (Starving)
    //   c = global.stealHalfAndPop(*local.getLocal());
    // if (!c)
    //   c = local.getLocal()->pop();
    // if (c)
    //   return c;
    if (ChunkHeader* c = local.getLocal()->first.pop())
      return c;
    return doSteal();
  }
};

template< bool separateBuffering = true, int chunksize = 64,
	  typename gWL = StealingQueues, typename T = int>
class ChunkedAdaptor : private boost::noncopyable {

  class Chunk :public ChunkHeader, public FixedSizeRing<T, chunksize> {};

  MM::FixedSizeAllocator heap;

  PerThreadStorage<std::pair<Chunk*, Chunk*> > data;

  gWL worklist;

  Chunk* mkChunk() {
    return new (heap.allocate(sizeof(Chunk))) Chunk();
  }
  
  void delChunk(Chunk* C) {
    C->~Chunk();
    heap.deallocate(C);
  }

  void swapInPush(std::pair<Chunk*, Chunk*>& d) {
    if (separateBuffering)
      std::swap(d.first, d.second);
  }

  Chunk*& getPushChunk(std::pair<Chunk*, Chunk*>& d) {
    if (separateBuffering)
      return d.second;
    else
      return d.first;
  }

  Chunk*& getPopChunk(std::pair<Chunk*, Chunk*>& d) {
    return d.first;
  }

  bool doPush(Chunk* c, const T& val) {
    return c->push_back(val);
  }

  boost::optional<T> doPop(Chunk* c) {
    if (separateBuffering)
      return c->pop_front();
    else
      return c->pop_back();
  }

  void push_internal(std::pair<Chunk*, Chunk*>& tld, Chunk*& n, const T& val) {
    //Simple case, space in current chunk
    if (n && doPush(n, val))
      return;
    //full chunk, push
    if (n)
      worklist.push(static_cast<ChunkHeader*>(n));
    //get empty chunk;
    n = mkChunk();
    //There better be some room in the new chunk
    doPush(n, val);
  }

public:
  template<typename Tnew>
  struct retype {
    typedef ChunkedAdaptor<separateBuffering, chunksize, gWL, Tnew> WL;
  };

  typedef T value_type;

  ChunkedAdaptor() : heap(sizeof(Chunk)) {}

  void push(value_type val) {
    std::pair<Chunk*, Chunk*>& tld = *data.getLocal();
    Chunk*& n = getPushChunk(tld);
    push_internal(tld, n, val);
  }

  template<typename Iter>
  void push(Iter b, Iter e) {
    std::pair<Chunk*, Chunk*>& tld = *data.getLocal();
    Chunk*& n = getPushChunk(tld);
    while (b != e)
      push_internal(tld, n, *b++);
  }

  template<typename Iter>
  void push_initial(Iter b, Iter e)  {
    fill_work(*this, b, e);
  }

  boost::optional<value_type> pop() {
    std::pair<Chunk*, Chunk*>& tld = *data.getLocal();
    Chunk*& n = getPopChunk(tld);
    boost::optional<value_type> retval;
    //simple case, things in current chunk
    if (n && (retval = doPop(n)))
      return retval;
    //empty chunk, trash it
    if (n)
      delChunk(n);
    //get a new chunk
    n = static_cast<Chunk*>(worklist.pop());
    if (n && (retval = doPop(n)))
      return retval;
    //try stealing the push buffer if we can
    swapInPush(tld);
    if (n)
      retval = doPop(n);
    return retval;
  }
};
//WLCOMPILECHECK(ChunkedAdaptor);

template<typename QueueTy>
boost::optional<typename QueueTy::value_type>
stealHalfInPackage(PerThreadStorage<QueueTy>& queues) {
  unsigned id = LL::getTID();
  unsigned pkg = LL::getPackageForThread(id);
  unsigned num = ThreadPool::getActiveThreads();
  QueueTy* me = queues.getLocal();
  boost::optional<typename QueueTy::value_type> retval;
  
  //steal from this package
  //Having 2 loops avoids a modulo, though this is a slow path anyway
  for (unsigned i = id + 1; i < num; ++i)
    if (LL::getPackageForThreadInternal(i) == pkg)
      if ((retval = me->steal(*queues.getRemote(i), true, true)))
	return retval;
  for (unsigned i = 0; i < id; ++i)
    if (LL::getPackageForThreadInternal(i) == pkg)
      if ((retval = me->steal(*queues.getRemote(i), true, true)))
	return retval;
  return retval;
}

template<typename QueueTy>
boost::optional<typename QueueTy::value_type>
stealRemote(PerThreadStorage<QueueTy>& queues) {
  unsigned id = LL::getTID();
  unsigned pkg = LL::getPackageForThread(id);
  unsigned num = ThreadPool::getActiveThreads();
  QueueTy* me = queues.getLocal();
  boost::optional<typename QueueTy::value_type> retval;
  
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
  PerThreadStorage<QueueTy> local;

  boost::optional<value_type> doSteal() {
    boost::optional<value_type> retval = stealHalfInPackage(local);
    if (retval)
      return retval;
    return stealRemote(local);
  }

  template<typename Iter>
  void fill_work_l2(Iter& b, Iter& e) {
    unsigned int a = ThreadPool::getActiveThreads();
    unsigned int id = LL::getTID();
    unsigned dist = std::distance(b, e);
    unsigned num = (dist + a - 1) / a; //round up
    unsigned int A = std::min(num * id, dist);
    unsigned int B = std::min(num * (id + 1), dist);
    e = b;
    std::advance(b, A);
    std::advance(e, B);
  }

  // LL::SimpleLock<true> L;
  // std::vector<unsigned> sum;

  template<typename Iter>
  void fill_work_l1(Iter b, Iter e) {
    Iter b2 = b;
    Iter e2 = e;
    fill_work_l2(b2, e2);
    unsigned int a = ThreadPool::getActiveThreads();
    unsigned int id = LL::getTID();
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
  struct retype {
    typedef PerThreadQueues<typename QueueTy::template retype<Tnew>::WL> WL;
  };

  template<bool newConcurrent>
  struct rethread {
    typedef PerThreadQueues<typename QueueTy::template rethread<newConcurrent>::WL> WL;
  };

  void push(const value_type& val) {
    local.getLocal()->push(val);
  }

  template<typename Iter>
  void push(Iter b, Iter e) {
    local.getLocal()->push(b,e);
  }

  template<typename Iter>
  void push_initial(Iter b, Iter e) {
    //    fill_work(*this, b, e);
    fill_work_l1(b,e);
  }

  boost::optional<value_type> pop() {
    boost::optional<value_type> retval = local.getLocal()->pop();
    if (retval)
      return retval;
    return doSteal();// stealHalfInPackage(local);
  }
};
//WLCOMPILECHECK(LocalQueues);


} }//End namespace

#endif
