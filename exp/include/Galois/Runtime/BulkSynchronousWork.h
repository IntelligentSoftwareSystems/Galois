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

#include <boost/fusion/adapted/mpl.hpp>
#include <boost/fusion/algorithm/iteration/fold.hpp>
#include <boost/fusion/algorithm/iteration/for_each.hpp>
#include <boost/fusion/algorithm/transformation/filter_if.hpp>
#include <boost/fusion/container/generation/make_vector.hpp>
#include <boost/fusion/container/vector.hpp>
#include <boost/fusion/iterator/advance.hpp>
#include <boost/fusion/iterator/deref.hpp>
#include <boost/fusion/sequence/intrinsic/at.hpp>
#include <boost/fusion/sequence/intrinsic/size.hpp>
#include <boost/fusion/sequence/intrinsic/value_at.hpp>
#include <boost/fusion/view/transform_view.hpp>
#include <boost/fusion/view/zip_view.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/mpl/for_each.hpp>
#include <boost/mpl/has_xxx.hpp>
#include <boost/mpl/range_c.hpp>
#include <boost/mpl/vector_c.hpp>

#include <cstdio>

namespace Galois {
namespace Runtime {
namespace BulkSynchronousWork {

template<typename T, bool isLIFO, unsigned ChunkSize>
struct RingAdaptor: public Galois::FixedSizeRing<T,ChunkSize> {
  typedef typename RingAdaptor::reference reference;

  reference cur() { return isLIFO ? this->front() : this->back();  }

  template<typename U>
  void push(U&& val) {
    this->push_front(std::forward<U>(val));
  }

  void pop()  {
    if (isLIFO) this->pop_front();
    else this->pop_back();
  }
};

struct WID {
  unsigned tid;
  unsigned pid;
  explicit WID(unsigned t): tid(t) {
    pid = LL::getLeaderForThread(tid);
  }
};

template<typename T,template<typename,bool> class OuterTy, bool isLIFO, int ChunkSize>
class dChunkedMaster : private boost::noncopyable {
  struct Chunk: public RingAdaptor<T,isLIFO,ChunkSize>, public OuterTy<Chunk, true>::ListNode {

    int mark;
    Chunk(): mark(0) { }

    template<typename FnTy>
    unsigned map(FnTy& fn) {
      for (typename Chunk::iterator ii = this->begin(), ei = this->end(); ii != ei; ++ii) {
        fn(*ii);
      }
      return this->size();
    }
  };

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


  template<typename FnTy>
  size_t map(const WID& id, FnTy fn, int mark) {
    p& n = *data.getLocal(id.tid);
    size_t iterations = 0;
    if (n.next) {
      iterations += n.next->map(fn);
    }
    LevelItem& I = *Q.getLocal(id.pid);
    for (typename LevelItem::iterator ii = I.begin(), ei = I.end(); ii != ei; ++ii) {
      int m;
      if ((m = ii->mark) != mark && __sync_bool_compare_and_swap(&ii->mark, m, mark)) {
        iterations += ii->map(fn);
      }
    }
    return iterations;
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

#define GALOIS_USE_BAG 1

#if GALOIS_USE_BAG
template<typename T,int ChunkSize>
class Worklist: public galois_insert_bag<T> { };
#else
template<typename T,int ChunkSize>
//class Worklist: public dChunkedMaster<T, WorkList::ConExtLinkedQueue, true, ChunkSize> { };
class Worklist: public dChunkedMaster<T, WorkList::ConExtLinkedStack, true, ChunkSize> { };

template<typename WorklistTy>
struct BindPush {
  typedef typename WorklistTy::value_type value_type;
  const WID& wid;
  WorklistTy& wl;
  BindPush(const WID& wid, WorklistTy& wl): wid(wid), wl(wl) { }
  void push(const value_type& x) {
    wl.push(wid, x);
  }
};
#endif

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

// Some type functions
template<typename ItemTy>
struct typeof_worklist { typedef Worklist<ItemTy, 256> type; };

template<typename ItemTy>
struct typeof_usercontext { typedef Galois::Runtime::UserContextAccess<ItemTy> type; };

template<typename FnTy>
struct needs_push { typedef boost::mpl::bool_<ForEachTraits<FnTy>::NeedsPush> type; };

template<typename VecTy>
struct project1: boost::fusion::result_of::value_at<VecTy, boost::mpl::int_<1> >  { };

template<typename ItemsTy, typename FnsTy, typename InitialWorkTy>
class Executor {
  typedef typename map_type_fn<ItemsTy, typeof_worklist>::type WLS;
  typedef typename map_type_fn<ItemsTy, typeof_usercontext>::type UserContexts;
  typedef typename boost::fusion::result_of::value_at<WLS, boost::mpl::int_<0> >::type FirstWL;

  struct ThreadLocalData {
    UserContexts facing;
    long iterations;
    int rounds;
    BulkSynchronousWork::WID wid;

    explicit ThreadLocalData(unsigned tid): iterations(0), rounds(0), wid(tid) { }
  };

  FnsTy fns;
  WLS wls;
  FirstWL first;
  GBarrier barrier;
  InitialWorkTy init;
  const char* loopname;
  LL::CacheLineStorage<volatile long> done;

  template<typename FnTy, typename FacingTy>
  struct FnWrap {
    FnTy& fn;
    FacingTy& facing;
    FnWrap(FnTy& f, FacingTy& n): fn(f), facing(n) { }

    template<typename T>
    void operator()(T&& x) const {
      fn(std::forward<T>(x), facing.data());
    }
  };

  struct ExecuteFn {
    Executor* self;
    ThreadLocalData& tld;
    FirstWL* cur;
    FirstWL* next;

    ExecuteFn(Executor* s, ThreadLocalData& t, FirstWL* c, FirstWL* n):
      self(s), tld(t), cur(c), next(n) { }
    
#if GALOIS_USE_BAG
    template<typename InWL, typename OutWL, typename FacingTy, typename FnTy>
    void process(InWL& in, OutWL& out, FacingTy& facing, FnTy& fn) const {
      typedef typename InWL::value_type value_type;

      for (typename InWL::local_iterator ii = in.local_begin(), ei = in.local_end(); ii != ei; ++ii) {
        fn(*ii, out); //facing.data());
        //tld.iterations += 1;
        //if (ForEachTraits<FnTy>::NeedsPush) {
        //  std::copy(facing.getPushBuffer().begin(), facing.getPushBuffer().end(),
        //      std::back_inserter(out));
        //  facing.resetPushBuffer();
        //}
      }
    }
#else
    template<typename InWL, typename OutWL, typename FacingTy, typename FnTy>
    void process(InWL& in, OutWL& out, FacingTy& facing, FnTy& fn) const {
      BindPush<OutWL> binder(tld.wid, out);
      typedef typename InWL::value_type value_type;
      while (!in.empty(tld.wid)) {
        int cs = std::max(in.currentChunkSize(tld.wid), 1U);
        for (int i = 0; i < cs; ++i) {
          value_type& val = in.cur(tld.wid);
          fn(val, binder); //facing.data());
          //tld.iterations += 1;
          //out.push(tld.wid,
          //    facing.getPushBuffer().begin(),
          //    facing.getPushBuffer().end());
          //facing.resetPushBuffer();
          in.pop(tld.wid);
        }
      }
    }
#endif
    template<typename FnIndex, typename InWL, typename OutWL, typename FacingTy, typename FnTy>
    void processInPlace(InWL& in, OutWL& out, FacingTy& facing, FnTy& fn) const {
      int mark = boost::fusion::result_of::size<FnsTy>::value * tld.rounds + FnIndex::value + 1;
      tld.iterations += in.map(tld.wid, FnWrap<FnTy, FacingTy>(fn, facing), mark);
    }

    template<typename TupleTy>
    void operator()(const TupleTy& tuple) const {
      // Bunch of template meta-programming to select appropriate worklists.
      // Effect of the below, simplified:
      //  WL in = cur, out = next; Facing f = facing[0]
      //  if (cur != 0) { in = wls[cur] }
      //  if (cur + 1 != size) { out = wls[cur+1]; f = facing[cur+1] }
      //  process(in, out, f, fn)
      using namespace boost;

      typedef mpl::int_<0> Zero;
      typedef mpl::int_<1> One;
      typedef mpl::int_<2> Two;

      typedef typename fusion::result_of::size<WLS>::type Size;
      typedef typename fusion::result_of::value_at<TupleTy, Zero>::type FnRefTy;
      typedef typename boost::remove_reference<FnRefTy>::type FnTy;
      typedef typename fusion::result_of::value_at<TupleTy, One>::type FnIndex;
      typedef typename fusion::result_of::value_at<TupleTy, Two>::type InIndex;

      typedef typename mpl::plus<InIndex, One>::type OutIndex;
      typedef typename mpl::equal_to<OutIndex, Size>::type IsLast;
      typedef typename mpl::if_<IsLast, Zero, OutIndex>::type SafeOutIndex;
      
      typedef typename fusion::result_of::value_at<WLS, InIndex>::type InWL;
      typedef typename fusion::result_of::value_at<WLS, SafeOutIndex>::type OutWL;
      typedef fusion::vector<FirstWL*, InWL*> InWLPtrs;
      typedef fusion::vector<FirstWL*, OutWL*> OutWLPtrs;
      
      typedef typename mpl::if_<mpl::equal_to<InIndex, Zero>, Zero, One>::type InWLIndex;
      typedef typename mpl::if_<mpl::equal_to<OutIndex, Size>, Zero, One>::type OutWLIndex;

      InWLPtrs inWlPtrs(cur, &fusion::at<InIndex>(self->wls)); 
      OutWLPtrs outWlPtrs(next, &fusion::at<SafeOutIndex>(self->wls)); 

#if GALOIS_USE_BAG
      process(*fusion::at<InWLIndex>(inWlPtrs),
          *fusion::at<OutWLIndex>(outWlPtrs),
          fusion::at<SafeOutIndex>(tld.facing),
          fusion::at<Zero>(tuple));
#else
      if (ForEachTraits<FnTy>::NeedsPush) {
        process(*fusion::at<InWLIndex>(inWlPtrs),
            *fusion::at<OutWLIndex>(outWlPtrs),
            fusion::at<SafeOutIndex>(tld.facing),
            fusion::at<Zero>(tuple));
      } else {
        processInPlace<FnIndex>(*fusion::at<InWLIndex>(inWlPtrs),
            *fusion::at<OutWLIndex>(outWlPtrs),
            fusion::at<SafeOutIndex>(tld.facing),
            fusion::at<Zero>(tuple));
      }
#endif
      self->barrier.wait();
    }
  };

  struct Clear {
    template<typename WL>
    void operator()(WL& wl) const {
      const_cast<typename boost::remove_const<WL>::type &>(wl).clear();
    }
  };

  struct value_printer {
    template<typename T> void operator()(T x) {
      std::cout << "C: " << x << "\n";
    }
  };

  void processFunctions(ThreadLocalData& tld, FirstWL* cur, FirstWL* next) {
    using namespace boost;

    // N element bool vector, true where FnTy::NeedsPush
    typedef typename map_type_fn<FnsTy, needs_push>::type NeedsPush;
    // N + 1 element int vector, [0] + [<prefix sum of NeedsPush>]
    typedef typename mpl::fold<
      NeedsPush,
      mpl::vector_c<int, 0>, 
      mpl::if_<
        mpl::_2,
        mpl::push_back<mpl::_1, mpl::plus<mpl::back<mpl::_1>, mpl::int_<1> > >,
        mpl::push_back<mpl::_1, mpl::back<mpl::_1> > >
      >::type PrefixSum;
    // Indexes of input worklists
    typedef typename mpl::pop_back<PrefixSum>::type InputIndices;

    typedef mpl::range_c<int, 0, fusion::result_of::size<FnsTy>::type::value> FnIndices;
    typedef fusion::vector<FnsTy&, FnIndices&, InputIndices&> Tuple;
 
    //std::cout << "begin\n";
    //mpl::for_each<InputIndices>(value_printer());
    //std::cout << "end\n";

    FnIndices fnIndices;
    InputIndices inputIndices;

    fusion::for_each(
        fusion::zip_view<Tuple>(Tuple(fns, fnIndices, inputIndices)),
        ExecuteFn(this, tld, cur, next));
  }

  template<typename FacingTy>
  void initialize(ThreadLocalData& tld, FirstWL& out, FacingTy& facing) {
    typedef typename InitialWorkTy::range_type::local_iterator local_iterator;
    for (local_iterator ii = init.range.local_begin(), ei = init.range.local_end(); ii != ei; ++ii) {
#if GALOIS_USE_BAG
      init.fn(*ii, out);
      //std::copy(facing.getPushBuffer().begin(), facing.getPushBuffer().end(),
      //    std::back_inserter(out));
#else
      BindPush<FirstWL> binder(tld.wid, out);
      init.fn(*ii, binder);
      //out.push(tld.wid,
      //    facing.getPushBuffer().begin(),
      //    facing.getPushBuffer().end());
      //facing.resetPushBuffer();
#endif
    }
  }

  void initialize(ThreadLocalData& tld, FirstWL& out) {
    initialize(tld, out, boost::fusion::at<boost::mpl::int_<0> >(tld.facing));
  }

public:
  explicit Executor(const FnsTy& f, const InitialWorkTy& i, const char* l): fns(f), init(i), loopname(l) { 
    barrier.reinit(Galois::Runtime::galoisActiveThreads);
  }

  void operator()() {
    ThreadLocalData tld(LL::getTID());

    FirstWL* cur = &first;
    FirstWL* next = &boost::fusion::at<boost::mpl::int_<0> >(wls);

    initialize(tld, *cur);

    while (true) {
      processFunctions(tld, cur, next);

      if (tld.wid.tid == 0) {
#if GALOIS_USE_BAG
        if (next->empty())
          done.data = true;
        cur->clear();
        boost::fusion::for_each(boost::fusion::pop_front(wls), Clear());
#else
        if (next->sempty())
          done.data = true;
#endif
      }
      tld.rounds += 1;
      barrier.wait();

      if (done.data)
        break;

      std::swap(next, cur);
    }

    if (tld.wid.tid == 0)
      reportStat(loopname, "Rounds", tld.rounds);
    reportStat(loopname, "Iterations", tld.iterations);
  }
};

template<class T, class FunctionTy, typename InitialWorkTy>
class Executor2 {
  typedef T value_type;
  typedef Worklist<value_type,256> WLTy;

  struct ThreadLocalData {
    Galois::Runtime::UserContextAccess<value_type> facing;
    SimpleRuntimeContext cnx;
    LoopStatistics<ForEachTraits<FunctionTy>::NeedsStats> stat;
    ThreadLocalData(const char* ln): stat(ln) { }
  };

  Galois::Runtime::GBarrier barrier1;
  Galois::Runtime::GBarrier barrier2;
  WLTy wls[2];
  FunctionTy function;
  InitialWorkTy init;
  const char* loopname;
  LL::CacheLineStorage<volatile long> done;
  unsigned numActive;

  bool empty(WLTy* wl) {
    return wl->sempty();
  }

  GALOIS_ATTRIBUTE_NOINLINE
  void abortIteration(ThreadLocalData& tld, const WID& wid, WLTy* cur, WLTy* next) {
    tld.cnx.cancel_iteration();
    tld.stat.inc_conflicts();
    if (ForEachTraits<FunctionTy>::NeedsPush) {
      tld.facing.resetPushBuffer();
    }
    value_type& val = cur->cur(wid);
    next->push(wid, val);
    cur->pop(wid);
  }

  void processWithAborts(ThreadLocalData& tld, const WID& wid, WLTy* cur, WLTy* next) {
    int result = 0;
#if GALOIS_USE_EXCEPTION_HANDLER
    try {
      process(tld, wid, cur, next);
    } catch (ConflictFlag const& flag) {
      clearConflictLock();
      result = flag;
    }
#else
    if ((result = setjmp(hackjmp)) == 0) {
      process(tld, wid, cur, next);
    }
#endif
    switch (result) {
    case 0: break;
    case Galois::Runtime::CONFLICT:
      abortIteration(tld, wid, cur, next);
      break;
    case Galois::Runtime::BREAK:
    default:
      abort();
    }
  }

  void process(ThreadLocalData& tld, const WID& wid, WLTy* cur, WLTy* next) {
    int cs = std::max(cur->currentChunkSize(wid), 1U);
    for (int i = 0; i < cs; ++i) {
      value_type& val = cur->cur(wid);
      tld.stat.inc_iterations();
      function(val, tld.facing.data());
      if (ForEachTraits<FunctionTy>::NeedsPush) {
        next->push(wid,
            tld.facing.getPushBuffer().begin(),
            tld.facing.getPushBuffer().end());
        tld.facing.resetPushBuffer();
      }
      if (ForEachTraits<FunctionTy>::NeedsAborts)
        tld.cnx.commit_iteration();
      cur->pop(wid);
    }
  }

  void initialize(ThreadLocalData& tld, WID& wid) {
    typedef typename InitialWorkTy::range_type::local_iterator local_iterator;
    for (local_iterator ii = init.range.local_begin(), ei = init.range.local_end(); ii != ei; ++ii) {
      init.fn(*ii, tld.facing.data());
      wls[0].push(wid, 
          tld.facing.getPushBuffer().begin(),
          tld.facing.getPushBuffer().end());
      tld.facing.resetPushBuffer();
    }
  }

  void go() {
    ThreadLocalData tld(loopname);
    setThreadContext(&tld.cnx);
    unsigned tid = LL::getTID();
    WID wid(tid);

    WLTy* cur = &wls[0];
    WLTy* next = &wls[1];

    initialize(tld, wid);

    while (true) {
      while (!cur->empty(wid)) {
        if (ForEachTraits<FunctionTy>::NeedsAborts) {
          processWithAborts(tld, wid, cur, next);
        } else {
          process(tld, wid, cur, next);
        }
        if (ForEachTraits<FunctionTy>::NeedsPIA)
          tld.facing.resetAlloc();
      }

      std::swap(next, cur);

      barrier1.wait();

      if (tid == 0) {
        if (empty(cur))
          done.data = true;
      }
      
      barrier2.wait();

      if (done.data)
        break;
    }

    setThreadContext(0);
  }

public:
  Executor2(const FunctionTy& f, const InitialWorkTy& i, const char* ln): function(f), init(i), loopname(ln) { 
    if (ForEachTraits<FunctionTy>::NeedsBreak) {
      assert(0 && "not supported by this executor");
      abort();
    }

    numActive = galoisActiveThreads;
    barrier1.reinit(numActive);
    barrier2.reinit(numActive);
  }

  void operator()() {
    go();
  }
};

#if 0
template<typename ItemsTy, typename IterTy, typename FnsTy, typename InitFnTy>
static inline void do_all_bs_impl(IterTy b, IterTy e, FnsTy fns, InitFnTy initFn, const char* loopname) {
  typedef StandardRange<IterTy> R;
  typedef InitialWork<R,InitFnTy> InitialWork;
  typedef typename boost::fusion::result_of::value_at<ItemsTy, boost::mpl::int_<0> >::type FirstItem;
  typedef typename boost::fusion::result_of::value_at<FnsTy, boost::mpl::int_<0> >::type FirstFn;
  typedef Executor2<FirstItem,FirstFn,InitialWork> Work;

  Work W(boost::fusion::at<boost::mpl::int_<0> >(fns), InitialWork(makeStandardRange(b, e), initFn), loopname);

  assert(!inGaloisForEach);

  inGaloisForEach = true;
  RunCommand w[2] = {Config::ref(W),
		     Config::ref(getSystemBarrier())};
  getSystemThreadPool().run(&w[0], &w[2]);
  runAllLoopExitHandlers();
  inGaloisForEach = false;
}
#endif

template<typename T>
struct CopyIn {
  template<typename Context>
  void operator()(const T& item, Context& ctx) const {
    ctx.push(item);
  }
};

#if 1
template<typename ItemsTy, typename IterTy, typename FnsTy, typename InitFnTy>
static inline void do_all_bs_impl(IterTy b, IterTy e, FnsTy fns, InitFnTy initFn, const char* loopname) {
  using namespace boost;

  //! Keep only items for functions that need push
  typedef typename map_type_fn<FnsTy, needs_push>::type NeedsPush;
  typedef typename fusion::result_of::filter_if<
    fusion::zip_view<fusion::vector<NeedsPush&, ItemsTy&> >,
    fusion::result_of::value_at<mpl::_1, mpl::int_<0> >
    >::type FilteredItemPairs;
  typedef typename map_type_fn<FilteredItemPairs, project1>::type FilteredItems;
  typedef typename fusion::result_of::as_vector<FilteredItems>::type FilteredItemsVector;

  typedef StandardRange<IterTy> R;
  typedef InitialWork<R, InitFnTy> InitialWork;
  typedef Executor<FilteredItemsVector, FnsTy, InitialWork> Work;

  Work W(fns, InitialWork(makeStandardRange(b, e), initFn), loopname);

  assert(!inGaloisForEach);

  inGaloisForEach = true;
  RunCommand w[2] = {Config::ref(W),
		     Config::ref(getSystemBarrier())};
  getSystemThreadPool().run(&w[0], &w[2]);
  runAllLoopExitHandlers();
  inGaloisForEach = false;
}
#endif
} // end BulkSynchronousWork
} // end Runtime
} // end Galois

namespace Galois {
template<typename ItemsTy, typename IterTy, typename FnsTy>
static inline void do_all_bs(IterTy b, IterTy e, FnsTy fns) {
  typedef typename std::iterator_traits<IterTy>::value_type value_type;
  typedef typename Galois::Runtime::BulkSynchronousWork::CopyIn<value_type> InitFn;
  Galois::Runtime::BulkSynchronousWork::do_all_bs_impl<ItemsTy>(b, e, fns, InitFn(), 0);
}

template<typename ItemsTy, typename IterTy, typename FnsTy, typename InitFnTy>
static inline void do_all_bs(IterTy b, IterTy e, FnsTy fns, InitFnTy initFn) {
  typedef typename std::iterator_traits<IterTy>::value_type value_type;
  typedef typename Galois::Runtime::BulkSynchronousWork::CopyIn<value_type> InitFn;
  Galois::Runtime::BulkSynchronousWork::do_all_bs_impl<ItemsTy>(b, e, fns, initFn, 0);
}
}

#endif
