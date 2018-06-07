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

#ifndef GALOIS_RUNTIME_KDG_ADD_REM_H
#define GALOIS_RUNTIME_KDG_ADD_REM_H

#include "galois/GaloisForwardDecl.h"
#include "galois/Reduction.h"
#include "galois/Atomic.h"
#include "galois/gdeque.h"
#include "galois/PriorityQueue.h"
#include "galois/Timer.h"
#include "galois/AltBag.h"
#include "galois/PerThreadContainer.h"

#include "galois/runtime/Context.h"
#include "galois/runtime/OrderedLockable.h"
#include "galois/runtime/Executor_DoAll.h"
#include "galois/runtime/Range.h"
#include "galois/runtime/Mem.h"
#include "galois/runtime/IKDGbase.h"

#include "galois/worklists/WorkList.h"

#include "galois/gIO.h"

#include "llvm/ADT/SmallVector.h"

#include <iostream>
#include <unordered_map>

namespace galois {
namespace runtime {

namespace cll = llvm::cl;

static cll::opt<bool> addRemWinArg("addRemWin", cll::desc("enable windowing in add-rem executor"), cll::init(false));

static const bool debug = false;

template <typename Ctxt, typename CtxtCmp>
class NhoodItem: public OrdLocBase<NhoodItem<Ctxt, CtxtCmp>, Ctxt, CtxtCmp> {
  using Base = OrdLocBase<NhoodItem, Ctxt, CtxtCmp>;

public:
  // using PQ =  galois::ThreadSafeOrderedSet<Ctxt*, CtxtCmp>;
  using PQ =  galois::ThreadSafeMinHeap<Ctxt*, CtxtCmp>;
  using Factory = OrdLocFactoryBase<NhoodItem, Ctxt, CtxtCmp>;

protected:
  PQ sharers;

public:
  NhoodItem (Lockable* l, const CtxtCmp& ctxtcmp):  Base (l), sharers (ctxtcmp) {}

  void add (const Ctxt* ctxt) {

    assert (!sharers.find (const_cast<Ctxt*> (ctxt)));
    sharers.push (const_cast<Ctxt*> (ctxt));
  }

  bool isHighestPriority (const Ctxt* ctxt) const {
    assert (ctxt);
    assert (!sharers.empty ());
    return (sharers.top () == ctxt);
  }

  Ctxt* getHighestPriority () const {
    if (sharers.empty ()) {
      return NULL;

    } else {
      return sharers.top ();
    }
  }

  void remove (const Ctxt* ctxt) {
    sharers.remove (const_cast<Ctxt*> (ctxt));
    // XXX: may fail in parallel execution
    assert (!sharers.find (const_cast<Ctxt*> (ctxt)));
  }

  void print () const {
    // TODO
  }
};


template <typename T, typename Cmp>
class KDGaddRemContext: public OrderedContextBase<T> {

public:
  typedef T value_type;
  typedef KDGaddRemContext MyType;
  typedef ContextComparator<MyType, Cmp> CtxtCmp;
  typedef NhoodItem<MyType, CtxtCmp> NItem;
  typedef PtrBasedNhoodMgr<NItem> NhoodMgr;
  typedef galois::GAtomic<bool> AtomicBool;
  // typedef galois::gdeque<NItem*, 4> NhoodList;
  // typedef llvm::SmallVector<NItem*, 8> NhoodList;
  typedef typename gstl::Vector<NItem*> NhoodList;
  // typedef std::vector<NItem*> NhoodList;

  // TODO: fix visibility below
public:
  // FIXME: nhood should be a set instead of list
  // AtomicBool onWL;
  NhoodMgr& nhmgr;
  NhoodList nhood;
  // GALOIS_ATTRIBUTE_ALIGN_CACHE_LINE AtomicBool onWL;
  AtomicBool onWL;

public:

  KDGaddRemContext (const T& active, NhoodMgr& nhmgr)
    :
      OrderedContextBase<T> (active), // to make acquire call virtual function sub_acquire
      nhmgr (nhmgr),
      onWL (false)
  {}

  GALOIS_ATTRIBUTE_PROF_NOINLINE
  virtual void subAcquire (Lockable* l, galois::MethodFlag) {
    NItem& nitem = nhmgr.getNhoodItem (l);

    assert (NItem::getOwner (l) == &nitem);

    if (std::find (nhood.begin (), nhood.end (), &nitem) == nhood.end ()) {
      nhood.push_back (&nitem);
      nitem.add (this);
    }

  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE bool isSrc () const {
    bool ret = true;

    for (auto n = nhood.begin () , endn = nhood.end (); n != endn; ++n) {
      if (!(*n)->isHighestPriority (this)) {
        ret = false;
        break;
      }
    }

    return ret;
  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE void removeFromNhood () {
    for (auto n = nhood.begin ()
        , endn = nhood.end (); n != endn; ++n) {

      (*n)->remove (this);
    }
  }

// for DEBUG
  std::string str () const {
    std::stringstream ss;
#if 0
    ss << "[" << this << ": " << active << "]";
#endif
    return ss.str ();
  }

  template <typename SourceTest, typename WL>
  GALOIS_ATTRIBUTE_PROF_NOINLINE void findNewSources (const SourceTest& srcTest, WL& wl) {

    for (auto n = nhood.begin ()
        , endn = nhood.end (); n != endn; ++n) {

      KDGaddRemContext* highest = (*n)->getHighestPriority ();
      if ((highest != NULL)
          && !bool (highest->onWL)
          && srcTest (highest)
          && highest->onWL.cas (false, true)) {

        // GALOIS_DEBUG ("Adding found source: %s\n", highest->str ().c_str ());
        wl.push (highest);
      }
    }
  }

  // TODO: combine with above method and reuse the code
  template <typename SourceTest, typename WL>
  GALOIS_ATTRIBUTE_PROF_NOINLINE void findSrcInNhood (const SourceTest& srcTest, WL& wl) {

    for (auto n = nhood.begin ()
        , endn = nhood.end (); n != endn; ++n) {

      KDGaddRemContext* highest = (*n)->getHighestPriority ();
      if ((highest != NULL)
          && !bool (highest->onWL)
          && srcTest (highest)
          && highest->onWL.cas (false, true)) {

        // GALOIS_DEBUG ("Adding found source: %s\n", highest->str ().c_str ());
        wl.push_back (highest);
      }
    }
  }

  void disableSrc (void) const {
    // XXX: nothing to do here. Added to reuse runCatching
  }


};



template <typename StableTest>
struct SourceTest {

  StableTest stabilityTest;

  explicit SourceTest (const StableTest& stabilityTest)
    : stabilityTest (stabilityTest) {}

  template <typename Ctxt>
  bool operator () (const Ctxt* ctxt) const {
    assert (ctxt != NULL);
    return ctxt->isSrc () && stabilityTest (ctxt->getActive());
  }
};

template <>
struct SourceTest <void> {

  template <typename Ctxt>
  bool operator () (const Ctxt* ctxt) const {
    assert (ctxt != NULL);
    return ctxt->isSrc ();
  }
};

// TODO: remove template parameters that can be passed to execute
template <typename T, typename Cmp, typename NhFunc, typename OpFunc, typename SourceTest, typename ArgsTuple, typename Ctxt>
class KDGaddRemAsyncExec: public IKDGbase<T, Cmp, NhFunc, internal::DummyExecFunc, OpFunc, ArgsTuple, Ctxt> {

  using ThisClass = KDGaddRemAsyncExec;

protected:
  using Base = IKDGbase<T, Cmp, NhFunc, internal::DummyExecFunc, OpFunc, ArgsTuple, Ctxt>;

  // important paramters
  // TODO: add capability to the interface to express these constants
  static const size_t DELETE_CONTEXT_SIZE = 1024;
  static const size_t UNROLL_FACTOR = OpFunc::UNROLL_FACTOR;

  static const unsigned DEFAULT_CHUNK_SIZE = 8;

  typedef galois::worklists::dChunkedFIFO<OpFunc::CHUNK_SIZE, Ctxt*> SrcWL_ty;
  // typedef galois::worklists::PerThreadChunkFIFO<CHUNK_SIZE, Ctxt*> SrcWL_ty;


  // typedef MapBasedNhoodMgr<T, Cmp> NhoodMgr;
  // typedef NhoodItem<T, Cmp, NhoodMgr> NItem;
  // typedef typename NItem::Ctxt Ctxt;

  using NhoodMgr =  typename Ctxt::NhoodMgr;

  using CtxtAlloc = typename Base::CtxtAlloc;
  using CtxtWL = typename Base::CtxtWL;

  using CtxtDelQ = PerThreadDeque<Ctxt*>; // XXX: can also use gdeque
  using CtxtLocalQ = PerThreadDeque<Ctxt*>;

  using UserCtxt = typename Base::UserCtxt;
  using PerThreadUserCtxt = typename Base::PerThreadUserCtxt;


  using Accumulator =  galois::GAccumulator<size_t>;

  struct CreateCtxtExpandNhood {
    KDGaddRemAsyncExec& exec;
    Accumulator& nInit;

    GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (const T& active) const {

      Ctxt* ctxt = exec.ctxtAlloc.allocate (1);
      assert (ctxt != NULL);
      // new (ctxt) Ctxt (active, nhmgr);
      //ctxtAlloc.construct (ctxt, Ctxt (active, nhmgr));
      exec.ctxtAlloc.construct (ctxt, active, exec.nhmgr);

      exec.addCtxtWL.get ().push_back (ctxt);

      nInit += 1;

      auto& uhand = *(exec.userHandles.getLocal ());
      runCatching (exec.nhFunc, ctxt, uhand);
    }

  };

  struct DummyWinWL {
    void push (const T&) const {
      std::abort ();
    }
  };


  template <typename WinWL>
  struct ApplyOperator {
    static const bool USE_WIN_WL = !std::is_same<WinWL, DummyWinWL>::value;
    typedef int tt_does_not_need_aborts;

    KDGaddRemAsyncExec& exec;
    WinWL& winWL;
    const galois::optional<T>& minWinWL;
    Accumulator& nsrc;
    Accumulator& ntotal;;

    template <typename WL>
    GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (Ctxt* const in_src, WL& wl) {
      assert (in_src != NULL);

      exec.ctxtLocalQ.get ().clear ();

      exec.ctxtLocalQ.get ().push_back (in_src);


      for (unsigned local_iter = 0;
          (local_iter < UNROLL_FACTOR) && !exec.ctxtLocalQ.get ().empty (); ++local_iter ) {

        Ctxt* src = exec.ctxtLocalQ.get ().front (); exec.ctxtLocalQ.get ().pop_front ();

        // GALOIS_DEBUG ("Processing source: %s\n", src->str ().c_str ());
        if (debug && !exec.sourceTest (src)) {
          std::cout << "Not found to be a source: " << src->str ()
            << std::endl;
          // abort ();
        }

        nsrc += 1;

        // addWL.get ().clear ();
        UserCtxt& userCtxt = *(exec.userHandles.getLocal ());

        if (ThisClass::NEEDS_PUSH) {
          userCtxt.resetPushBuffer ();
        }

        exec.opFunc (src->getActive (), userCtxt);


        if (ThisClass::NEEDS_PUSH) {

          exec.addCtxtWL.get ().clear ();
          CreateCtxtExpandNhood addCtxt {exec, ntotal};


          for (auto a = userCtxt.getPushBuffer ().begin ()
              , enda = userCtxt.getPushBuffer ().end (); a != enda; ++a) {

            if (!USE_WIN_WL || !minWinWL || exec.cmp (*a, *minWinWL)) {
              addCtxt (*a);
            } else {
              winWL.push (*a);
            }

          }

          for (auto c = exec.addCtxtWL.get ().begin ()
              , endc = exec.addCtxtWL.get ().end (); c != endc; ++c) {

            (*c)->findNewSources (exec.sourceTest, wl);
            // // if is source add to workList;
            // if (sourceTest (*c) && (*c)->onWL.cas (false, true)) {
            // // std::cout << "Adding new source: " << *c << std::endl;
            // wl.push (*c);
            // }
          }
        }

        src->removeFromNhood ();

        src->findSrcInNhood (exec.sourceTest, exec.ctxtLocalQ.get ());

        //TODO: use a ref count type wrapper for Ctxt;
        exec.ctxtDelQ.get ().push_back (src);

      }

      // add remaining to global wl
      // TODO: check onWL counter here
      for (auto c = exec.ctxtLocalQ.get ().begin ()
          , endc = exec.ctxtLocalQ.get ().end (); c != endc; ++c) {

        wl.push (*c);
      }

      while (exec.ctxtDelQ.get ().size () >= DELETE_CONTEXT_SIZE) {

        Ctxt* c = exec.ctxtDelQ.get ().front (); exec.ctxtDelQ.get ().pop_front ();
        exec.ctxtAlloc.destroy (c);
        exec.ctxtAlloc.deallocate (c, 1);
      }
    }

  };

private:
  NhoodMgr& nhmgr;
  SourceTest sourceTest;
  CtxtWL addCtxtWL;
  CtxtLocalQ ctxtLocalQ;
  CtxtDelQ ctxtDelQ;


public:

  KDGaddRemAsyncExec (
      const Cmp& cmp,
      const NhFunc& nhFunc,
      const OpFunc& opFunc,
      const ArgsTuple& argsTuple,
      NhoodMgr& nhmgr,
      const SourceTest& sourceTest)
    :
      Base (cmp, nhFunc, internal::DummyExecFunc (), opFunc, argsTuple),
      nhmgr (nhmgr),
      sourceTest (sourceTest)
  {}


  template <typename R>
  void expandNhoodpickSources (const R& range, CtxtWL& sources, Accumulator& nInit) {

    addCtxtWL.clear_all_parallel ();

    galois::runtime::do_all_gen (
        range,
        CreateCtxtExpandNhood {*this, nInit},
        std::make_tuple (
          galois::loopname ("create_contexts"),
          galois::chunk_size<NhFunc::CHUNK_SIZE> ()));

    galois::runtime::do_all_gen (makeLocalRange(this->addCtxtWL),
        [this, &sources] (Ctxt* ctxt) {
          if (sourceTest (ctxt) && ctxt->onWL.cas (false, true)) {
            sources.get ().push_back (ctxt);
          }
        },
        std::make_tuple (
          galois::loopname ("find_sources"),
          galois::chunk_size<DEFAULT_CHUNK_SIZE> ()));

  }

  template <typename A>
  void applyOperator (CtxtWL& sources, A op) {

    galois::for_each(galois::iterate(sources),
        op,
        galois::loopname("apply_operator"), galois::wl<SrcWL_ty>());

    galois::runtime::do_all_gen (makeLocalRange(ctxtDelQ),
        [this] (Ctxt* ctxt) {
          ThisClass::ctxtAlloc.destroy (ctxt);
          ThisClass::ctxtAlloc.deallocate (ctxt, 1);
        },
        std::make_tuple (
          galois::loopname ("delete_all_ctxt"),
          galois::chunk_size<DEFAULT_CHUNK_SIZE> ()));

    galois::on_each(
        [this, &sources] (const unsigned tid, const unsigned numT) {
          sources.get ().clear ();
          ctxtDelQ.get ().clear ();
        });

  }

  template <typename R>
  void execute (const R& range) {
    CtxtWL initSrc;

    Accumulator nInitCtxt;
    Accumulator nsrc;

    galois::TimeAccumulator t_create;
    galois::TimeAccumulator t_find;
    galois::TimeAccumulator t_for;
    galois::TimeAccumulator t_destroy;

    t_create.start ();
    expandNhoodpickSources (range, initSrc, nInitCtxt);
    t_create.stop ();

    // TODO: code to find global min goes here

    DummyWinWL winWL;
    galois::optional<T> minWinWL; // should remain uninitialized

    t_for.start ();
    applyOperator (initSrc, ApplyOperator<DummyWinWL> {*this, winWL, minWinWL, nsrc, nInitCtxt});
    t_for.stop ();

    reportStat_Single (Base::loopname, "Number of iterations: ", nsrc.reduce ());
    reportStat_Single (Base::loopname, "Time taken in creating intial contexts: ",   t_create.get ());
    reportStat_Single (Base::loopname, "Time taken in finding intial sources: ", t_find.get ());
    reportStat_Single (Base::loopname, "Time taken in for_each loop: ", t_for.get ());
    reportStat_Single (Base::loopname, "Time taken in destroying all the contexts: ", t_destroy.get ());
  }
};


template <typename T, typename Cmp, typename NhFunc, typename OpFunc, typename SourceTest, typename ArgsTuple, typename Ctxt>
class KDGaddRemWindowExec: public KDGaddRemAsyncExec<T, Cmp, NhFunc, OpFunc, SourceTest, ArgsTuple, Ctxt> {

  using Base = KDGaddRemAsyncExec<T, Cmp, NhFunc, OpFunc, SourceTest, ArgsTuple, Ctxt>;

  using WindowWL = typename std::conditional<Base::NEEDS_PUSH, PQwindowWL<T, Cmp>, SortedRangeWindowWL<T, Cmp> >::type;

  using CtxtWL = typename Base::CtxtWL;
  using Accumulator = typename Base::Accumulator;

  using ThisClass = KDGaddRemWindowExec;


  WindowWL winWL;
  PerThreadBag<T> pending;
  CtxtWL sources;
  Accumulator nsrc;

  void beginRound (void) {
    ThisClass::refillRound (winWL, pending);
  }

  void expandNhoodPending (void) {

    ThisClass::expandNhoodpickSources (makeLocalRange (pending), sources, ThisClass::roundTasks);
    pending.clear_all_parallel ();

  }

  void applyOperator (void) {

    galois::optional<T> minWinWL;

    if (ThisClass::NEEDS_PUSH && ThisClass::targetCommitRatio != 0.0) {
      minWinWL = winWL.getMin ();
    }

    using Op = typename ThisClass::template ApplyOperator<WindowWL>;
    Base::applyOperator (sources, Op {*this, winWL, minWinWL, ThisClass::roundCommits, ThisClass::roundTasks});
  }

  template <typename R>
  void push_initial (const R& range) {

    if (ThisClass::targetCommitRatio == 0.0) {

      galois::runtime::do_all_gen (range,
          [this] (const T& x) {
            pending.push (x);
          },
          std::make_tuple (
            galois::loopname ("init-fill"),
            chunk_size<NhFunc::CHUNK_SIZE> ()));


    } else {
      winWL.initfill (range);
    }
  }

public:

  KDGaddRemWindowExec (
      const Cmp& cmp,
      const NhFunc& nhFunc,
      const OpFunc& opFunc,
      const ArgsTuple& argsTuple,
      typename Base::NhoodMgr& nhmgr,
      const SourceTest& sourceTest)
    :
      Base (cmp, nhFunc, opFunc, argsTuple, nhmgr, sourceTest)
  {}

  template <typename R>
  void execute (const R& range) {

    push_initial (range);

    while (true) {

      beginRound ();

      expandNhoodPending ();

      if (sources.empty_all ()) {
        assert (pending.empty_all ());
        break;
      }


      applyOperator ();

      ThisClass::endRound ();


    }
  }

};


template <
    template <typename _one, typename _two, typename _three, typename _four, typename _five, typename _six, typename _seven> class Executor,
    typename R, typename Cmp, typename NhFunc, typename OpFunc, typename ST, typename ArgsTuple>
void for_each_ordered_ar_impl (const R& range, const Cmp& cmp, const NhFunc& nhFunc, const OpFunc& opFunc, const ST& sourceTest, const ArgsTuple& argsTuple) {

  typedef typename R::value_type T;

  auto argsT = std::tuple_cat (argsTuple,
      get_default_trait_values (argsTuple,
        std::make_tuple (loopname_tag {}, enable_parameter_tag {}),
        std::make_tuple (default_loopname {}, enable_parameter<false> {})));
  using ArgsT = decltype (argsT);

  typedef KDGaddRemContext<T, Cmp> Ctxt;
  typedef typename Ctxt::NhoodMgr NhoodMgr;
  typedef typename Ctxt::NItem NItem;
  typedef typename Ctxt::CtxtCmp  CtxtCmp;

  using Exec =  Executor<T, Cmp, NhFunc, OpFunc, ST, ArgsT, Ctxt>;

  std::cout << "sizeof(KDGaddRemContext) == " << sizeof(Ctxt) << std::endl;

  CtxtCmp ctxtcmp (cmp);
  typename NItem::Factory factory(ctxtcmp);
  NhoodMgr nhmgr (factory);

  Exec e (cmp, nhFunc, opFunc, argsT, nhmgr, sourceTest);
  e.execute (range);
}

template <typename R, typename Cmp, typename NhFunc, typename OpFunc, typename StableTest, typename ArgsTuple>
void for_each_ordered_ar (const R& range, const Cmp& cmp, const NhFunc& nhFunc, const OpFunc& opFunc, const StableTest& stabilityTest, const ArgsTuple& argsTuple) {

  if (addRemWinArg) {
    for_each_ordered_ar_impl<KDGaddRemWindowExec> (range, cmp, nhFunc, opFunc, SourceTest<StableTest> (stabilityTest), argsTuple);
  } else {
    for_each_ordered_ar_impl<KDGaddRemAsyncExec> (range, cmp, nhFunc, opFunc, SourceTest<StableTest> (stabilityTest), argsTuple);
  }
}

template <typename R, typename Cmp, typename NhFunc, typename OpFunc, typename ArgsTuple>
void for_each_ordered_ar (const R& range, const Cmp& cmp, const NhFunc& nhFunc, const OpFunc& opFunc, const ArgsTuple& argsTuple) {

  if (addRemWinArg) {
    for_each_ordered_ar_impl<KDGaddRemWindowExec> (range, cmp, nhFunc, opFunc, SourceTest<void> (), argsTuple);
  } else {
    for_each_ordered_ar_impl<KDGaddRemAsyncExec> (range, cmp, nhFunc, opFunc, SourceTest<void> (), argsTuple);
  }
}

} // end namespace runtime
} // end namespace galois

#endif //  GALOIS_RUNTIME_KDG_ADD_REM_H
