/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
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

#ifndef GALOIS_RUNTIME_BULKSYNCHRONOUSWORK_H
#define GALOIS_RUNTIME_BULKSYNCHRONOUSWORK_H

#include <boost/version.hpp>

#if BOOST_VERSION < 104300
// Bug in boost fusion fixed in version 1.43
#define GALOIS_HAS_NO_BULKSYNCHRONOUS_EXECUTOR
#else
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

namespace galois {
namespace runtime {
namespace {

template <typename T, unsigned ChunkSize>
struct RingAdaptor : public galois::FixedSizeBag<T, ChunkSize> {
  typedef typename RingAdaptor::reference reference;

  int mark;

  RingAdaptor() : mark(0) {}

  template <typename U>
  void push(U&& val) {
    this->push_front(std::forward<U>(val));
  }

  template <typename FnTy>
  unsigned map(FnTy fn) {
    for (typename RingAdaptor::iterator ii = this->begin(), ei = this->end();
         ii != ei; ++ii) {
      fn(*ii);
    }
    return this->size();
  }
};

template <typename T, unsigned ChunkSize>
struct Bag : public galois::gdeque<T, ChunkSize, RingAdaptor<T, ChunkSize>> {
  typedef typename Bag::Block block_type;

  block_type* volatile cur;
  block_type* middle;
  unsigned int chunkCount;
  bool toFront;

  Bag() : cur(0), middle(0), chunkCount(0), toFront(false) {}

  void push(const T& item) {
    this->push_front(item);
#if 0
    if (toFront)
      this->push_front(item);
    else
      this->push_back(item);

    if (++chunkCount == ChunkSize) {
      toFront = !toFront;
      chunkCount = 0;
    }

    if (this->size() == 1) {
      middle = this->first;
    }
#endif
  }

  void reset() {
    this->clear();
    cur        = 0;
    middle     = 0;
    chunkCount = 0;
    toFront    = false;
  }

  void finish() { cur = this->first; }

  template <typename FnTy>
  size_t map(FnTy fn, int mark) {
    size_t iterations = 0;
    for (block_type* ii = this->first; ii; ii = cur = ii->next) {
      int m;
      if ((m = ii->mark) != mark &&
          __sync_bool_compare_and_swap(&ii->mark, m, mark))
        iterations += ii->map(fn);
    }
    return iterations;
  }

  template <typename FnTy>
  size_t map_steal(FnTy fn, int mark, int count) {
    size_t iterations = 0;
    int c             = 0;
    for (block_type* ii = cur; ii && c < count; ii = ii->next) {
      int m;
      if ((m = ii->mark) != mark &&
          __sync_bool_compare_and_swap(&ii->mark, m, mark)) {
        iterations += ii->map(fn);
        ++c;
      }
    }
    return iterations;
  }

  void divideWith(Bag& other) {}
};

struct WIDb {
  unsigned tid;
  unsigned pid;
  explicit WIDb(unsigned t) : tid(t) {
    pid = substrate::getThreadPool().getLeader(tid);
  }
};

template <typename T, unsigned ChunkSize>
struct BagMaster : boost::noncopyable {
  typedef Bag<T, ChunkSize> local_type;

  substrate::PerThreadStorage<local_type> bags;

  local_type& get(unsigned mytid) { return *bags.getLocal(mytid); }

  void reset() { bags.getLocal()->reset(); }

  template <typename FnTy>
  size_t map(const WIDb& id, FnTy fn, int mark) {
    size_t iterations = bags.getLocal()->map(fn, mark);

    return iterations + mapSlow(id, fn, mark);
  }

  template <typename FnTy>
  GALOIS_ATTRIBUTE_NOINLINE size_t mapSlow(const WIDb& id, FnTy fn, int mark) {
    size_t iterations = 0;
    auto& tp          = substrate::getThreadPool();
    while (true) {
      unsigned failures = 0;

      for (unsigned int i = 0; i < bags.size() - 1; ++i) {
        unsigned idx = id.tid + 1 + i;
        if (idx >= bags.size())
          idx -= bags.size();

        unsigned opid = tp.getSocket(idx);

        if (opid == id.pid) {
          size_t executed = bags.getRemote(idx)->map_steal(fn, mark, 1);
          if (executed == 0)
            ++failures;
          iterations += executed;
        } else {
          ++failures;
        }
      }

      if (failures == bags.size() - 1)
        break;
    }

    return iterations;
  }

  bool sempty() {
    for (unsigned i = 0; i < bags.size(); ++i) {
      if (!bags.getLocal(i)->empty())
        return false;
    }
    return true;
  }
};

template <typename T, int ChunkSize>
class Worklistb : public BagMaster<T, ChunkSize> {};

//! Encapsulation of initial work to pass to executor
template <typename RangeTy, typename InitFnTy>
struct InitialWork {
  typedef RangeTy range_type;

  RangeTy range;
  InitFnTy fn;

  InitialWork(const RangeTy& r, const InitFnTy& f) : range(r), fn(f) {}
};

// Apply type function to vector of types
template <typename TyVector, template <typename> class TyFn>
class map_type_fn {
  struct apply {
    template <typename Sig>
    struct result;

    template <typename U>
    struct result<apply(U)> : boost::remove_reference<typename TyFn<U>::type> {
    };

    template <typename U>
    typename TyFn<U>::type operator()(U) const;
  };

  typedef boost::fusion::transform_view<TyVector, apply> T1;

public:
  typedef typename boost::fusion::result_of::as_vector<T1>::type type;
};

// Some type functions
template <typename ItemTy>
struct typeof_worklist {
  typedef Worklistb<ItemTy, 256> type;
};

template <typename ItemTy>
struct typeof_usercontext {
  typedef galois::runtime::UserContextAccess<ItemTy> type;
};

template <typename FnTy>
struct needs_push {
  typedef boost::mpl::bool_<DEPRECATED::ForEachTraits<FnTy>::NeedsPush> type;
};

template <typename VecTy>
struct project1
    : boost::fusion::result_of::value_at<VecTy, boost::mpl::int_<1>> {};

template <typename FnTy, typename T>
struct Bind2nd {
  FnTy& fn;
  T& obj;
  Bind2nd(FnTy& f, T& o) : fn(f), obj(o) {}

  template <typename U>
  void operator()(U&& x) const {
    fn(std::forward<U>(x), obj);
  }
};

template <typename ItemsTy, typename FnsTy, typename InitialWorkTy>
class Executor {
  typedef typename map_type_fn<ItemsTy, typeof_worklist>::type WLS;
  typedef typename map_type_fn<ItemsTy, typeof_usercontext>::type UserContexts;
  typedef typename boost::fusion::result_of::value_at<
      WLS, boost::mpl::int_<0>>::type FirstWL;

  struct ThreadLocalData {
    UserContexts facing;
    size_t iterations;
    int rounds;
    WIDb wid;

    explicit ThreadLocalData(unsigned tid)
        : iterations(0), rounds(0), wid(tid) {}
  };

  FnsTy fns;
  WLS wls;
  FirstWL first;
  InitialWorkTy init;
  const char* loopname;
  substrate::Barrier& barrier;
  substrate::CacheLineStorage<volatile long> done;

  struct ExecuteFn {
    Executor* self;
    ThreadLocalData& tld;
    FirstWL* cur;
    FirstWL* next;

    ExecuteFn(Executor* s, ThreadLocalData& t, FirstWL* c, FirstWL* n)
        : self(s), tld(t), cur(c), next(n) {}

    template <typename InWL, typename WL>
    void rebalance(InWL& in, WL& self) const {
      // XXX
      if (tld.wid.tid + 10 >= activeThreads)
        return;

      WL& other = in.get(tld.wid.tid + 10);
      if (self.size() / 2 > other.size()) {
        self.divideWith(other);
      }
    }

    template <bool NeedsRebalancing, typename InWL, typename OutWL,
              typename FacingTy, typename FnTy>
    void process(InWL& in, OutWL& out, FacingTy& facing, FnTy& fn,
                 int mark) const {
      typedef typename OutWL::local_type local_out_type;
      typedef typename InWL::local_type local_in_type;

      local_in_type& localIn   = in.get(tld.wid.tid);
      local_out_type& localOut = out.get(tld.wid.tid);

      localIn.finish();

      self->barrier.wait();
      if (false && NeedsRebalancing) {
        rebalance(in, localIn);
        self->barrier.wait();
      }

      tld.iterations +=
          in.map(tld.wid, Bind2nd<FnTy, local_out_type>(fn, localOut), mark);
    }

    template <typename TupleTy>
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
      typedef
          typename fusion::result_of::value_at<WLS, SafeOutIndex>::type OutWL;
      typedef fusion::vector<FirstWL*, InWL*> InWLPtrs;
      typedef fusion::vector<FirstWL*, OutWL*> OutWLPtrs;

      typedef typename mpl::if_<mpl::equal_to<InIndex, Zero>, Zero, One>::type
          InWLIndex;
      typedef typename mpl::if_<mpl::equal_to<OutIndex, Size>, Zero, One>::type
          OutWLIndex;

      InWLPtrs inWlPtrs(cur, &fusion::at<InIndex>(self->wls));
      OutWLPtrs outWlPtrs(next, &fusion::at<SafeOutIndex>(self->wls));

      int mark = boost::fusion::result_of::size<FnsTy>::value * tld.rounds +
                 FnIndex::value + 1;

      const bool NeedsRebalancing = true; // XXX

      process<NeedsRebalancing>(
          *fusion::at<InWLIndex>(inWlPtrs), *fusion::at<OutWLIndex>(outWlPtrs),
          fusion::at<SafeOutIndex>(tld.facing), fusion::at<Zero>(tuple), mark);
    }
  };

  struct Clear {
    template <typename WL>
    void operator()(WL& wl) const {
      const_cast<typename boost::remove_const<WL>::type&>(wl).reset();
    }
  };

  // struct value_printer {
  //  template<typename T> void operator()(T x) {
  //    std::cout << "C: " << x << "\n";
  //  }
  //};

  void processFunctions(ThreadLocalData& tld, FirstWL* cur, FirstWL* next) {
    using namespace boost;

    // N element bool vector, true where FnTy::NeedsPush
    typedef typename map_type_fn<FnsTy, needs_push>::type NeedsPush;
    // N + 1 element int vector, [0] + [<prefix sum of NeedsPush>]
    typedef typename mpl::fold<
        NeedsPush, mpl::vector_c<int, 0>,
        mpl::if_<mpl::_2,
                 mpl::push_back<mpl::_1,
                                mpl::plus<mpl::back<mpl::_1>, mpl::int_<1>>>,
                 mpl::push_back<mpl::_1, mpl::back<mpl::_1>>>>::type PrefixSum;
    // Indexes of input worklists
    typedef typename mpl::pop_back<PrefixSum>::type InputIndices;

    typedef mpl::range_c<int, 0, fusion::result_of::size<FnsTy>::type::value>
        FnIndices;
    typedef fusion::vector<FnsTy&, FnIndices&, InputIndices&> Tuple;

    // std::cout << "begin\n";
    // mpl::for_each<InputIndices>(value_printer());
    // std::cout << "end\n";

    FnIndices fnIndices;
    InputIndices inputIndices;

    fusion::for_each(
        fusion::zip_view<Tuple>(Tuple(fns, fnIndices, inputIndices)),
        ExecuteFn(this, tld, cur, next));
  }

  template <typename FacingTy>
  void initialize(ThreadLocalData& tld, FirstWL& out, FacingTy& facing) {
    typedef typename InitialWorkTy::range_type::local_iterator local_iterator;
    typedef typename FirstWL::local_type local_type;
    local_type& localOut = out.get(tld.wid.tid);

    for (local_iterator ii = init.range.local_begin(),
                        ei = init.range.local_end();
         ii != ei; ++ii) {
      init.fn(*ii, localOut);
    }
    localOut.finish();
  }

  void initialize(ThreadLocalData& tld, FirstWL& out) {
    initialize(tld, out, boost::fusion::at<boost::mpl::int_<0>>(tld.facing));
  }

public:
  explicit Executor(const FnsTy& f, const InitialWorkTy& i, const char* l)
      : fns(f), init(i), loopname(l), barrier(getBarrier(activeThreads)) {}

  void operator()() {
    ThreadLocalData tld(substrate::ThreadPool::getTID());

    FirstWL* cur  = &first;
    FirstWL* next = &boost::fusion::at<boost::mpl::int_<0>>(wls);

    initialize(tld, *cur);

    while (true) {
      processFunctions(tld, cur, next);

      barrier.wait();

      if (tld.wid.tid == 0) {
        if (next->sempty())
          done.data = true;
      }
      tld.rounds += 1;
      barrier.wait();

      cur->reset();
      boost::fusion::for_each(boost::fusion::pop_front(wls), Clear());

      if (done.data)
        break;

      std::swap(next, cur);
    }

    if (tld.wid.tid == 0)
      reportStat(loopname, "Rounds", tld.rounds, 0);
    reportStat(loopname, "Iterations", tld.iterations, tld.wid.tid);
  }
};

template <typename T>
struct CopyIn {
  template <typename Context>
  void operator()(const T& item, Context& ctx) const {
    ctx.push(item);
  }
};

template <typename ItemsTy, typename RangeTy, typename FnsTy, typename InitFnTy>
static inline void do_all_bs_impl(const RangeTy& range, const FnsTy& fns,
                                  InitFnTy initFn, const char* loopname) {
  using namespace boost;

  //! Keep only items for functions that need push
  typedef typename map_type_fn<FnsTy, needs_push>::type NeedsPush;
  typedef typename fusion::result_of::filter_if<
      fusion::zip_view<fusion::vector<NeedsPush&, ItemsTy&>>,
      fusion::result_of::value_at<mpl::_1, mpl::int_<0>>>::type
      FilteredItemPairs;
  typedef typename map_type_fn<FilteredItemPairs, project1>::type FilteredItems;
  typedef typename fusion::result_of::as_vector<FilteredItems>::type
      FilteredItemsVector;

  typedef InitialWork<RangeTy, InitFnTy> InitialWork;
  typedef Executor<FilteredItemsVector, FnsTy, InitialWork> Work;

  Work W(fns, InitialWork(range, initFn), loopname);
  substrate::getThreadPool().run(activeThreads, std::ref(W));
}

} // namespace
} // namespace runtime
} // namespace galois

namespace galois {
template <typename ItemsTy, typename ConTy, typename FnsTy>
static inline void do_all_bs_local(ConTy& c, const FnsTy& fns) {
  typedef typename std::iterator_traits<typename ConTy::iterator>::value_type
      value_type;
  typedef typename galois::runtime::CopyIn<value_type> InitFn;
  galois::runtime::do_all_bs_impl<ItemsTy>(galois::runtime::makeLocalRange(c),
                                           fns, InitFn(), 0);
}

template <typename ItemsTy, typename ConTy, typename FnsTy, typename InitFnTy>
static inline void do_all_bs_local(ConTy& c, const FnsTy& fns,
                                   InitFnTy initFn) {
  galois::runtime::do_all_bs_impl<ItemsTy>(galois::runtime::makeLocalRange(c),
                                           fns, initFn, 0);
}

template <typename ItemsTy, typename IterTy, typename FnsTy>
static inline void do_all_bs(IterTy b, IterTy e, const FnsTy& fns) {
  typedef typename std::iterator_traits<IterTy>::value_type value_type;
  typedef typename galois::runtime::CopyIn<value_type> InitFn;
  galois::runtime::do_all_bs_impl<ItemsTy>(
      galois::runtime::makeStandardRange(b, e), fns, InitFn(), 0);
}

template <typename ItemsTy, typename IterTy, typename FnsTy, typename InitFnTy>
static inline void do_all_bs(IterTy b, IterTy e, const FnsTy& fns,
                             InitFnTy initFn) {
  galois::runtime::do_all_bs_impl<ItemsTy>(
      galois::runtime::makeStandardRange(b, e), fns, initFn, 0);
}
} // namespace galois

#endif

#endif
