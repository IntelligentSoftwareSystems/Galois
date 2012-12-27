/** Generic interface for bulk synchronous algorithms -*- C++ -*-
 * @file
 * This is the only file to include for basic Galois functionality.
 *
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
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#ifndef GALOIS_RUNTIME_BULKSYNCHRONOUSWORK_H
#define GALOIS_RUNTIME_BULKSYNCHRONOUSWORK_H

#include "Galois/Runtime/ParallelWork.h"

#include <boost/mpl/range_c.hpp>
#include <boost/fusion/adapted/mpl.hpp>
#include <boost/fusion/container/vector.hpp>
#include <boost/fusion/container/generation/make_vector.hpp>
#include <boost/fusion/view/transform_view.hpp>
#include <boost/fusion/sequence/intrinsic/value_at.hpp>
#include <boost/fusion/view/zip_view.hpp>
#include <boost/fusion/sequence/intrinsic/at.hpp>
#include <boost/fusion/sequence/intrinsic/size.hpp>
#include <boost/fusion/algorithm/iteration/for_each.hpp>
#include <boost/fusion/iterator/deref.hpp>
#include <boost/fusion/iterator/advance.hpp>

#include <cstdio>

namespace GaloisRuntime {
namespace BulkSynchronousWork {

//! Alternative implementation of FixedSizeRing which is non-concurrent and
//! supports pushX/empty/popX rather than pushX/popX with boost::optional

template<typename T, unsigned ChunkSize>
struct FixedSizeBase: private boost::noncopyable {
  char datac[sizeof(T[ChunkSize])] __attribute__ ((aligned (__alignof__(T))));

  T* data() { return reinterpret_cast<T*>(&datac[0]); }
  T* at(unsigned i) { return &data()[i]; }
  void create(int i, const T& val) { new (at(i)) T(val); }
  void destroy(unsigned i) { (at(i))->~T(); }
  inline unsigned chunksize() const { return ChunkSize; }
  typedef T value_type;
};

template<typename T, bool isLIFO, int ChunkSize>
class FixedSizeRing: public FixedSizeBase<T,ChunkSize> {
  unsigned start;
  unsigned count;

public:
  FixedSizeRing(): start(0), count(0) { }

  unsigned size() const { return count; }
  bool empty() const { return count == 0; }
  bool full() const { return count == this->chunksize(); }

  void push(const T& val) {
    start = (start + this->chunksize() - 1) % this->chunksize();
    ++count;
    this->create(start, val);
  }

  void pop() {
    unsigned end = (start + count - 1) % this->chunksize();
    this->destroy(end);
    --count;
  }

  T& cur() { 
    unsigned end = (start + count - 1) % this->chunksize();
    return *this->at(end); 
  }
};


template<typename T, int ChunkSize>
class FixedSizeRing<T,true,ChunkSize>: public FixedSizeBase<T,ChunkSize>  {
  unsigned end;

public:
  FixedSizeRing(): end(0) { }

  unsigned size() const { return end; }
  bool empty() const { return end == 0; }
  bool full() const { return end >= this->chunksize(); }
  void pop() { this->destroy(--end); }
  T& cur() { return *this->at(end-1); }
  void push(const T& val) { this->create(end, val); ++end; }
};

struct WID {
  unsigned tid;
  unsigned pid;
  explicit WID(unsigned t): tid(t) {
    pid = LL::getLeaderForThread(tid);
  }
};

template<typename T,template<typename,bool> class OuterTy, bool isLIFO,int ChunkSize>
class dChunkedMaster : private boost::noncopyable {
  class Chunk : public FixedSizeRing<T,isLIFO,ChunkSize>, public OuterTy<Chunk,true>::ListNode {};

  MM::FixedSizeAllocator heap;

  struct p {
    Chunk* next;
  };

  typedef OuterTy<Chunk, true> LevelItem;

  PerThreadStorage<p> data;
  PerPackageStorage<LevelItem> Q;

  Chunk* mkChunk() {
    return new (heap.allocate(sizeof(Chunk))) Chunk();
  }
  
  void delChunk(Chunk* C) {
    C->~Chunk();
    heap.deallocate(C);
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

  dChunkedMaster() : heap(sizeof(Chunk)) {
    for (unsigned int i = 0; i < data.size(); ++i) {
      p& r = *data.getRemote(i);
      r.next = 0;
    }
  }

  unsigned currentChunkSize(const WID& id) {
    p& n = *data.getLocal(id.tid);
    if (n.next) {
      return n.next->size();
    }
    return 0;
  }

  value_type& cur(const WID& id) {
    p& n = *data.getLocal(id.tid);
    return n.next->cur();
  }

  void push(const WID& id, const value_type& val)  {
    p& n = *data.getLocal(id.tid);
    if (n.next && !n.next->full()) {
      n.next->push(val);
      return;
    }
    pushSP(id, n, val);
  }

  template<typename Iter>
  void push(const WID& id, Iter b, Iter e) {
    while (b != e)
      push(id, *b++);
  }

  bool empty(const WID& id) {
    p& n = *data.getLocal(id.tid);
    if (n.next && !n.next->empty())
      return false;
    return emptySP(id, n);
  }

  //! Serial empty
  bool sempty() {
    for (unsigned i = 0; i < data.size(); ++i) {
      WID id(i);
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
class dChunkedLIFO: public dChunkedMaster<T, WorkList::ConExtLinkedStack, true, ChunkSize> { };

template<typename T,int ChunkSize>
class dChunkedFIFO: public dChunkedMaster<T, WorkList::ConExtLinkedQueue, false, ChunkSize> { };

//! Encapsulation of initial work to pass to executor
template<typename RangeTy, typename InitFnTy>
struct InitialWork {
  typedef RangeTy range_type;

  RangeTy range;
  InitFnTy fn;

  InitialWork(const RangeTy& r, const InitFnTy& f): range(r), fn(f) { }
};

// Apply type function to vector of types
template<typename TyVector, template<typename> class TyFn>
class map_type_fn {
  struct apply {
    template<typename Sig>
    struct result;

    template<typename U>
    struct result<apply(U)>: boost::remove_reference<typename TyFn<U>::type> { };

    template<typename U>
    typename TyFn<U>::type operator()(U) const;
  };

  typedef boost::fusion::transform_view<TyVector, apply> T1;
public:
  typedef typename boost::fusion::result_of::as_vector<T1>::type type;
};

template<typename ItemsTy, typename FnsTy, typename InitialWorkTy>
class Executor {
  template<typename ItemTy>
  struct typeof_worklist { typedef dChunkedLIFO<ItemTy,256> type; };

  template<typename ItemTy>
  struct typeof_usercontext { typedef GaloisRuntime::UserContextAccess<ItemTy> type; };

  typedef typename map_type_fn<ItemsTy, typeof_worklist>::type WLS;
  typedef typename boost::fusion::result_of::value_at<WLS, boost::mpl::int_<0> >::type FirstWL;
  typedef typename map_type_fn<ItemsTy, typeof_usercontext>::type UserContexts;

  struct ThreadLocalData {
    UserContexts facing;
    long iterations;
    BulkSynchronousWork::WID wid;

    explicit ThreadLocalData(unsigned tid): iterations(0), wid(tid) { }
  };

  FnsTy fns;
  WLS wls;
  FirstWL first;
  GBarrier barrier;
  InitialWorkTy init;
  const char* loopname;
  LL::CacheLineStorage<volatile long> done;
  LL::SimpleLock<true> lock;

  struct ExecuteFn {
    Executor* self;
    ThreadLocalData& tld;
    FirstWL* cur;
    FirstWL* next;

    ExecuteFn(Executor* s, ThreadLocalData& t, FirstWL* c, FirstWL* n):
      self(s), tld(t), cur(c), next(n) { }
    
    template<typename InWL, typename OutWL, typename FacingTy, typename FnTy>
    void process(InWL& in, OutWL& out, FacingTy& facing, FnTy& fn) const {
      typedef typename InWL::value_type value_type;
      while (!in.empty(tld.wid)) {
        int cs = std::max(in.currentChunkSize(tld.wid), 1U);
        for (int i = 0; i < cs; ++i) {
          value_type& val = in.cur(tld.wid);
          fn(val, facing.data());
          tld.iterations += 1;
          out.push(tld.wid,
              facing.getPushBuffer().begin(),
              facing.getPushBuffer().end());
          facing.resetPushBuffer();
          in.pop(tld.wid);
        }
      }
    }

    template<typename PairTy>
    void operator()(const PairTy& pair) const {
      // Bunch of template meta-programming to select appropriate worklists.
      // Effect of the below, simplified:
      //  WL in = cur, out = next;
      //  Facing f = facing[0]
      //  if (cur != 0) { in = wls[cur] }
      //  if (cur + 1 != size) { out = wls[cur+1]; f = facing[cur+1] }
      //  process(in, out, f, fn)
      typedef boost::mpl::int_<0> Zero;
      typedef boost::mpl::int_<1> One;
      typedef typename boost::fusion::result_of::value_at<PairTy,Zero>::type CurrentIndex;
      typedef typename boost::mpl::plus<CurrentIndex,One>::type NextIndex;
      typedef typename boost::fusion::result_of::size<WLS>::type Size;
      typedef typename boost::mpl::equal_to<NextIndex,Size>::type IsLast;
      typedef typename boost::mpl::if_<IsLast,Zero,NextIndex>::type SafeNextIndex;
      typedef typename boost::fusion::result_of::value_at<WLS,CurrentIndex>::type CurrentWL;
      typedef typename boost::fusion::result_of::value_at<WLS,SafeNextIndex>::type NextWL;
      typedef boost::fusion::vector<FirstWL*,CurrentWL*> InWL;
      typedef boost::fusion::vector<FirstWL*,NextWL*> OutWL;
      typedef typename boost::mpl::if_<boost::mpl::equal_to<CurrentIndex,Zero>,Zero,One>::type InIndex;
      typedef typename boost::mpl::if_<boost::mpl::equal_to<NextIndex,Size>,Zero,One>::type OutIndex;

      InWL in(cur, &boost::fusion::at<CurrentIndex>(self->wls)); 
      OutWL out(next, &boost::fusion::at<SafeNextIndex>(self->wls)); 

      process(*boost::fusion::at<InIndex>(in),
          *boost::fusion::at<OutIndex>(out),
          boost::fusion::at<SafeNextIndex>(tld.facing),
          boost::fusion::at<One>(pair));
      self->barrier.wait();
    }
  };

  template<typename FacingTy>
  void initialize(ThreadLocalData& tld, FirstWL& out, FacingTy& facing) {
    typedef typename InitialWorkTy::range_type::local_iterator local_iterator;
    for (local_iterator ii = init.range.local_begin(), ei = init.range.local_end(); ii != ei; ++ii) {
      init.fn(*ii, facing.data());
      out.push(tld.wid,
          facing.getPushBuffer().begin(),
          facing.getPushBuffer().end());
      facing.resetPushBuffer();
    }
  }

  void initialize(ThreadLocalData& tld, FirstWL& out) {
    initialize(tld, out, boost::fusion::at<boost::mpl::int_<0> >(tld.facing));
  }

public:
  explicit Executor(const FnsTy& f, const InitialWorkTy& i, const char* l): fns(f), init(i), loopname(l) { 
    barrier.reinit(GaloisRuntime::galoisActiveThreads);
  }

  void operator()() {
    int rounds = 0;
    ThreadLocalData tld(LL::getTID());

    typedef boost::mpl::range_c<int, 0, boost::fusion::result_of::size<FnsTy>::type::value> Range;
    Range range;

    typedef boost::fusion::vector<Range&,FnsTy&> Pair;

    FirstWL* cur = &first;
    FirstWL* next = &boost::fusion::at<boost::mpl::int_<0> >(wls);

    initialize(tld, *cur);

    while (true) {
      boost::fusion::for_each(
          boost::fusion::zip_view<Pair>(Pair(range, fns)),
          ExecuteFn(this, tld, cur, next));
      
      if (tld.wid.tid == 0) {
        rounds += 1;
        if (next->sempty())
          done.data = true;
      }
      barrier.wait();

      if (done.data)
        break;

      std::swap(next, cur);
    }

    if (tld.wid.tid == 0)
      reportStat(loopname, "Rounds", rounds);
    reportStat(loopname, "Iterations", tld.iterations);
  }
};

template<typename T>
struct CopyIn {
  void operator()(const T& item, Galois::UserContext<T>& ctx) const {
    ctx.push(item);
  }
};

template<typename ItemsTy, typename IterTy, typename FnsTy, typename InitFnTy>
static inline void do_all_bs_impl(IterTy b, IterTy e, FnsTy fns, InitFnTy initFn, const char* loopname) {
  typedef StandardRange<IterTy> R;
  typedef InitialWork<R,InitFnTy> InitialWork;
  typedef Executor<ItemsTy,FnsTy,InitialWork> Work;

  Work W(fns, InitialWork(makeStandardRange(b, e), initFn), loopname);

  assert(!inGaloisForEach);

  inGaloisForEach = true;
  RunCommand w[2] = {Config::ref(W),
		     Config::ref(getSystemBarrier())};
  getSystemThreadPool().run(&w[0], &w[2]);
  runAllLoopExitHandlers();
  inGaloisForEach = false;
}

} // end BulkSynchronousWork
} // end GaloisRuntime

namespace Galois {
template<typename ItemsTy, typename IterTy, typename FnsTy>
static inline void do_all_bs(IterTy b, IterTy e, FnsTy fns) {
  typedef typename std::iterator_traits<IterTy>::value_type value_type;
  typedef typename GaloisRuntime::BulkSynchronousWork::CopyIn<value_type> InitFn;
  GaloisRuntime::BulkSynchronousWork::do_all_bs_impl<ItemsTy>(b, e, fns, InitFn(), 0);
}

template<typename ItemsTy, typename IterTy, typename FnsTy, typename InitFnTy>
static inline void do_all_bs(IterTy b, IterTy e, FnsTy fns, InitFnTy initFn) {
  typedef typename std::iterator_traits<IterTy>::value_type value_type;
  typedef typename GaloisRuntime::BulkSynchronousWork::CopyIn<value_type> InitFn;
  GaloisRuntime::BulkSynchronousWork::do_all_bs_impl<ItemsTy>(b, e, fns, initFn, 0);
}
}

#endif
