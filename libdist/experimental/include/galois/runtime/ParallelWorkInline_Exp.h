/** Simple Worklists that do not adhere to the general worklist contract -*- C++ -*-
 * @file
 * This is the only file to include for basic Galois functionality.
 *
 * @section License
 *
 * This file is part of Galois.  Galois is a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, version 2.1 of the
 * License.
 *
 * Galois is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Galois.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @section Copyright
 *
 * Copyright (C) 2015, The University of Texas at Austin. All rights
 * reserved.
 *
 * @author Amber Hassaan <ahassaan@ices.utexas.edu>
 */

#ifndef GALOIS_RUNTIME_PARALLELWORKINLINE_EXP_H
#define GALOIS_RUNTIME_PARALLELWORKINLINE_EXP_H

#include <cstdio>

#include "galois/runtime/ForEachTraits.h"
// #define _DO_OUTER_PREFETCH 1

#include <xmmintrin.h>

namespace galois {
namespace runtime {
namespace exp {

template<typename T, bool isLIFO, unsigned ChunkSize>
struct FixedSizeRingAdaptor: public galois::FixedSizeRing<T,ChunkSize> {
  typedef typename FixedSizeRingAdaptor::reference reference;

  reference cur() { return isLIFO ? this->front() : this->back();  }

  template<typename U>
  void push(U&& val) {
    this->push_front(std::forward<U>(val));
  }

  void pop()  {
    if (isLIFO) this->pop_front();
    else this->pop_back();
  }

#ifdef _DO_OUTER_PREFETCH
  typename FixedSizeRingAdaptor::const_reference  lookAhead (unsigned off) const {
    assert (!this->empty ());
    if (isLIFO) {
      return this->getAt (off);
    } else {
      return this->getAt (this->size () - off);
    }
  }
#endif


};

struct WID {
  unsigned tid;
  unsigned pid;
  WID(unsigned t): tid(t) {
    pid = substrate::getThreadPool().getLeader(tid);
  }
  WID() {
    tid = substrate::ThreadPool::getTID();
    pid = substrate::ThreadPool::getLeader();
  }
};

template<typename T,template<typename,bool> class OuterTy, bool isLIFO,int ChunkSize>
class dChunkedMaster : private boost::noncopyable {
  class Chunk : public FixedSizeRingAdaptor<T,isLIFO,ChunkSize>, public OuterTy<Chunk,true>::ListNode {};

  FixedSizeAllocator<Chunk> alloc;

  struct p {
    Chunk* next;
  };

  typedef OuterTy<Chunk, true> LevelItem;

  substrate::PerThreadStorage<p> data;
  substrate::PerPackageStorage<LevelItem> Q;

  Chunk* mkChunk() {
    Chunk* ptr = alloc.allocate(1);
    alloc.construct(ptr);
    return ptr;
  }

  void delChunk(Chunk* ptr) {
    alloc.destroy(ptr);
    alloc.deallocate(ptr, 1);
  }

  void pushChunk(const WID& id, Chunk* C)  {
    LevelItem& I = *Q.getLocal(id.pid);
    I.push(C);
  }

  Chunk* popChunkByID(unsigned int i)  {
    LevelItem* I = Q.getRemote(i);
    if (I)
      return I->pop();
    return 0;
  }

  Chunk* popChunk(const WID& id)  {
    Chunk* r = popChunkByID(id.pid);
    if (r)
      return r;

    for (unsigned int i = id.pid + 1; i < Q.size(); ++i) {
      r = popChunkByID(i);
      if (r)
	return r;
    }

    for (unsigned int i = 0; i < id.pid; ++i) {
      r = popChunkByID(i);
      if (r)
	return r;
    }

    return 0;
  }

  void pushSP(const WID& id, p& n, const T& val);
  bool emptySP(const WID& id, p& n);
  void popSP(const WID& id, p& n);

public:
  typedef T value_type;

  dChunkedMaster() {
    for (unsigned int i = 0; i < data.size(); ++i) {
      p& r = *data.getRemote(i);
      r.next = 0;
    }
  }

  // exp
  inline void push (const value_type& val) {
    WID id;
    push (id, val);
  }

  void push(const WID& id, const value_type& val)  {
    p& n = *data.getLocal(id.tid);
    if (n.next && !n.next->full()) {
      n.next->push(val);
      return;
    }
    pushSP(id, n, val);
  }

  unsigned currentChunkSize(const WID& id) {
    p& n = *data.getLocal(id.tid);
    if (n.next) {
      return n.next->size();
    }
    return 0;
  }

  template<typename Iter>
  void push(const WID& id, Iter b, Iter e) {
    while (b != e)
      push(id, *b++);
  }

  template<typename Iter>
  void push_initial(const WID& id, Iter b, Iter e) {
    push(id, b, e);
  }

  value_type& cur(const WID& id) {
    p& n = *data.getLocal(id.tid);
    return n.next->cur();
  }

#ifdef _DO_OUTER_PREFETCH
  const value_type& lookAhead (const WID& id, unsigned off) const {
    p& n = *data.getLocal (id.tid);
    return n.next->lookAhead (off);
  }
#endif // _DO_OUTER_PREFETCH

  bool empty(const WID& id) {
    p& n = *data.getRemote(id.tid);
    if (n.next && !n.next->empty())
      return false;
    return emptySP(id, n);
  }

  bool sempty() {
    WID id;
    for (unsigned i = 0; i < data.size(); ++i) {
      id.tid = i;
      id.pid = substrate::getThreadPool().getLeader(i);
      if (!empty(id))
        return false;
    }
    return true;
  }

  void pop(const WID& id)  {
    p& n = *data.getLocal(id.tid);
    if (n.next && !n.next->empty()) {
      n.next->pop();
      return;
    }
    popSP(id, n);
  }
};

template<typename T,template<typename,bool> class OuterTy, bool isLIFO,int ChunkSize>
void dChunkedMaster<T,OuterTy,isLIFO,ChunkSize>::popSP(const WID& id, p& n) {
  while (true) {
    if (n.next && !n.next->empty()) {
      n.next->pop();
      return;
    }
    if (n.next)
      delChunk(n.next);
    n.next = popChunk(id);
    if (!n.next)
      return;
  }
}

template<typename T,template<typename,bool> class OuterTy, bool isLIFO,int ChunkSize>
bool dChunkedMaster<T,OuterTy,isLIFO,ChunkSize>::emptySP(const WID& id, p& n) {
  while (true) {
    if (n.next && !n.next->empty())
      return false;
    if (n.next)
      delChunk(n.next);
    n.next = popChunk(id);
    if (!n.next)
      return true;
  }
}

template<typename T,template<typename,bool> class OuterTy, bool isLIFO,int ChunkSize>
void dChunkedMaster<T,OuterTy,isLIFO,ChunkSize>::pushSP(const WID& id, p& n, const T& val) {
  if (n.next)
    pushChunk(id, n.next);
  n.next = mkChunk();
  n.next->push(val);
}

template<typename T,int ChunkSize>
// class Worklist: public dChunkedMaster<T, worklists::ConExtLinkedStack, true, ChunkSize> { };
class Worklist: public dChunkedMaster<T, worklists::ConExtLinkedQueue, true, ChunkSize> { };

template<class T, class FunctionTy, typename PreFunc=FunctionTy>
class BSInlineExecutor {
  typedef T value_type;
  static const unsigned CHUNK_SIZE = FunctionTy::CHUNK_SIZE;
  typedef Worklist<value_type,CHUNK_SIZE> WLTy;

  struct ThreadLocalData {
    galois::runtime::UserContextAccess<value_type> facing;
    SimpleRuntimeContext ctx;
    LoopStatistics<runtime::DEPRECATED::ForEachTraits<FunctionTy>::NeedsStats> stat;
    ThreadLocalData(const char* ln): stat(ln) { }
  };

  WLTy wls[2];
  FunctionTy function;
  PreFunc preFunc;
  const char* loopname;
  galois::substrate::Barrier& barrier;
  substrate::CacheLineStorage<volatile long> done;

  bool empty(WLTy* wl) {
    return wl->sempty();
  }

  GALOIS_ATTRIBUTE_NOINLINE
  void abortIteration(ThreadLocalData& tld, const WID& wid, WLTy* cur, WLTy* next) {
    tld.ctx.cancelIteration();
    tld.stat.inc_conflicts();
    if (runtime::DEPRECATED::ForEachTraits<FunctionTy>::NeedsPush) {
      tld.facing.resetPushBuffer();
    }
    value_type& val = cur->cur(wid);
    next->push(wid, val);
    cur->pop(wid);
  }

  void processWithAborts(ThreadLocalData& tld, const WID& wid, WLTy* cur, WLTy* next) {
    int result = 0;
#ifdef GALOIS_USE_LONGJMP
    if ((result = setjmp(hackjmp)) == 0) {
#else
    try {
#endif
      process(tld, wid, cur, next);
#ifdef GALOIS_USE_LONGJMP
    } else { clearConflictLock(); }
#else
    } catch (const ConflictFlag& flag) { clearConflictLock(); result = flag; }
#endif
    //FIXME: clearReleasable();
    switch (result) {
    case 0: break;
    case galois::runtime::CONFLICT:
      abortIteration(tld, wid, cur, next);
      break;
    case galois::runtime::BREAK:
    default:
      abort();
    }
  }

  void process(ThreadLocalData& tld, const WID& wid, WLTy* cur, WLTy* next) {
    int cs = std::max(cur->currentChunkSize(wid), 1U);
    for (int i = 0; i < cs; ++i) {


// #ifdef _DO_OUTER_PREFETCH
      // const unsigned l1_pftch_dist = 1;
      // if ((i + l1_pftch_dist) < cs) {
        // const value_type& next = cur->lookAhead (wid, l1_pftch_dist);
        // preFunc (next, _MM_HINT_T0);
      // }
// #endif
      value_type& val = cur->cur(wid);
// #ifdef _DO_OUTER_PREFETCH
      // preFunc (val, _MM_HINT_T0);
// #endif

      tld.stat.inc_iterations();
      function(val, *next);

      // function(val, tld.facing);
//
      // if (ForEachTraits<FunctionTy>::NeedsPush) {
        // next->push(wid,
            // tld.facing.getPushBuffer().begin(),
            // tld.facing.getPushBuffer().end());
        // tld.facing.resetPushBuffer();
      // }

      cur->pop(wid);

// #ifdef _DO_OUTER_PREFETCH
      // const unsigned l2_pftch_dist = 1;
      // if ((i + l2_pftch_dist) < cs) {
        // const value_type& next = cur->lookAhead (wid, l2_pftch_dist);
        // preFunc (next, _MM_HINT_T1);
      // }
// #endif


// #ifdef _DO_OUTER_PREFETCH
      // const unsigned l2_pftch_dist = 2;
      // if ((i + l2_pftch_dist) < cs) {
        // const value_type& next = cur->lookAhead (wid, l2_pftch_dist);
        // preFunc (next, _MM_HINT_T1);
      // }
// #endif

      if (runtime::DEPRECATED::ForEachTraits<FunctionTy>::NeedsAborts)
        tld.ctx.commitIteration();


    }
  }

  void go() {
    ThreadLocalData tld(loopname);
    setThreadContext(&tld.ctx);
    unsigned tid = substrate::ThreadPool::getTID();
    WID wid;

    WLTy* cur = &wls[0];
    WLTy* next = &wls[1];

    while (true) {
      while (!cur->empty(wid)) {
        if (runtime::DEPRECATED::ForEachTraits<FunctionTy>::NeedsAborts) {
          processWithAborts(tld, wid, cur, next);
        } else {
          process(tld, wid, cur, next);
        }
        if (runtime::DEPRECATED::ForEachTraits<FunctionTy>::NeedsPIA)
          tld.facing.resetAlloc();
      }

      std::swap(next, cur);

      barrier.wait();

      if (tid == 0) {
        if (empty(cur))
          done.get() = true;
      }

      barrier.wait();

      if (done.get())
        break;
    }

    setThreadContext(0);
  }

public:
  BSInlineExecutor(const FunctionTy& f, const char* ln): function(f), preFunc (f), loopname(ln), barrier(runtime::getBarrier(activeThreads)) {
    if (runtime::DEPRECATED::ForEachTraits<FunctionTy>::NeedsBreak) {
      assert(0 && "not supported by this executor");
      abort();
    }
  }

  BSInlineExecutor (const FunctionTy& f, const PreFunc& preFunc, const char* ln)
    :
      function(f),
      preFunc (preFunc),
      loopname(ln),
      barrier(runtime::getBarrier (activeThreads)),
      done (false)
  {
    if (runtime::DEPRECATED::ForEachTraits<FunctionTy>::NeedsBreak) {
      assert(0 && "not supported by this executor");
      abort();
    }
  }

  template<typename RangeTy>
  void AddInitialWork(RangeTy range) {
    wls[0].push_initial(WID(), range.local_begin(), range.local_end());
  }

  void initThread() {}

  void operator()() {
    go();
  }
};


} // end namespace Exp

template <typename R, typename OpFunc, typename PreFunc>
void for_each_bs (const R& range, const OpFunc& opFunc, const PreFunc& preFunc, const char* loopname=nullptr) {
  typedef typename R::value_type T;

  typedef Exp::BSInlineExecutor<T, OpFunc, PreFunc> Executor;

  Executor e (opFunc, preFunc, loopname);
  substrate::Barrier& barrier = runtime::getBarrier (activeThreads);

  substrate::getThreadPool ().run (activeThreads,
    std::bind (&Executor::template AddInitialWork<R>, std::ref (e), range),
    std::ref (barrier),
    std::ref (e));
}


} // end runtime

} //galois

#undef _DO_OUTER_PREFETCH

#endif
