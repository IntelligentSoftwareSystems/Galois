/** TODO -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a framework to exploit
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
 * @section Description
 *
 * TODO 
 *
 * @author <ahassaan@ices.utexas.edu>
 */
#ifndef GALOIS_RUNTIME_KDG_ADD_REM_H
#define GALOIS_RUNTIME_KDG_ADD_REM_H

#include "Galois/GaloisForwardDecl.h"
#include "Galois/Accumulator.h"
#include "Galois/Atomic.h"
#include "Galois/gdeque.h"
#include "Galois/PriorityQueue.h"
#include "Galois/Timer.h"
#include "Galois/AltBag.h"
#include "Galois/PerThreadContainer.h"

#include "Galois/WorkList/WorkList.h"
#include "Galois/Runtime/Context.h"
#include "Galois/Runtime/OrderedLockable.h"
#include "Galois/Runtime/Executor_DoAll.h"
#include "Galois/Runtime/Range.h"
#include "Galois/Substrate/gio.h"
#include "Galois/Runtime/ThreadRWlock.h"
#include "Galois/Runtime/Mem.h"


#include "llvm/ADT/SmallVector.h"

#include <iostream>
#include <unordered_map>

namespace Galois {
namespace Runtime {

static const bool debug = false;

template <typename Ctxt, typename CtxtCmp>
class NhoodItem: public OrdLocBase<NhoodItem<Ctxt, CtxtCmp>, Ctxt, CtxtCmp> {
  using Base = OrdLocBase<NhoodItem, Ctxt, CtxtCmp>;

public:
  using PQ =  Galois::ThreadSafeOrderedSet<Ctxt*, CtxtCmp>;
  using Factory = OrdLocFactoryBase<NhoodItem, Ctxt, CtxtCmp>;

protected:
  PQ sharers;

public:
  NhoodItem (Lockable* l, const CtxtCmp& ctxtcmp):  Base (l), sharers (ctxtcmp) {}

  void add (const Ctxt* ctxt) {

    // assert (!sharers.find (const_cast<Ctxt*> (ctxt)));
    sharers.push (const_cast<Ctxt*> (ctxt));
  }

  bool isHighestPriority (const Ctxt* ctxt) const {
    return !sharers.empty () && (sharers.top () == ctxt);
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
  typedef Galois::GAtomic<bool> AtomicBool;
  // typedef Galois::gdeque<NItem*, 4> NhoodList;
  // typedef llvm::SmallVector<NItem*, 8> NhoodList;
  typedef typename gstl::Vector<NItem*> NhoodList;
  // typedef std::vector<NItem*> NhoodList;

  // TODO: fix visibility below
public:
  // FIXME: nhood should be a set instead of list
  NhoodList nhood;
  NhoodMgr& nhmgr;
  GALOIS_ATTRIBUTE_ALIGN_CACHE_LINE AtomicBool onWL;

public:

  KDGaddRemContext (const T& active, NhoodMgr& nhmgr)
    : 
      OrderedContextBase (active), // to make acquire call virtual function sub_acquire
      nhood (), 
      nhmgr (nhmgr), 
      onWL (false) 
  {}

  GALOIS_ATTRIBUTE_PROF_NOINLINE
  virtual void subAcquire (Lockable* l, Galois::MethodFlag) {
    NItem& nitem = nhmgr.getNhoodItem (l);

    assert (NItem::getOwner (l) == &nitem);

    if (std::find (nhood.begin (), nhood.end (), &nitem) == nhood.end ()) {
      nhood.push_back (&nitem);
      nitem.add (this);
    }
    
  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE bool isSrc () const {
    assert (!nhood.empty ()); // TODO: remove later

    bool ret = true;

    for (auto n = nhood.begin ()
        , endn = nhood.end (); n != endn; ++n) {

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


};



template <typename StableTest>
struct SourceTest {

  StableTest stabilityTest;

  explicit SourceTest (const StableTest& stabilityTest)
    : stabilityTest (stabilityTest) {}

  template <typename Ctxt>
  bool operator () (const Ctxt* ctxt) const {
    assert (ctxt != NULL);
    return ctxt->isSrc () && stabilityTest (ctxt->active);
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
template <typename T, typename Cmp, typename NhFunc, typename OpFunc, typename ArgsTuple, typename Ctxt, typename SourceTest>
class KDGaddRemAsyncExec: public OrderedExecutorBase<T, Cmp, NhFunc, HIDDEN::DummyExecFunc, OpFunc, ArgsTuple, Ctxt> {

  using Base = OrderedExecutorBase<T, Cmp, NhFunc, HIDDEN::DummyExecFunc, OpFunc, ArgsTuple, Ctxt>;

  // important paramters
  // TODO: add capability to the interface to express these constants
  static const size_t DELETE_CONTEXT_SIZE = 1024;
  static const size_t UNROLL_FACTOR = OpFunc::UNROLL_FACTOR;

  static const unsigned DEFAULT_CHUNK_SIZE = 8;



  // typedef MapBasedNhoodMgr<T, Cmp> NhoodMgr;
  // typedef NhoodItem<T, Cmp, NhoodMgr> NItem;
  // typedef typename NItem::Ctxt Ctxt;

  using NhoodMgr =  typename Ctxt::NhoodMgr;

  using CtxtAlloc = typenme Base::CtxtAlloc;
  using CtxtWL = typenme Base::CtxtWL;

  using CtxtDelQ = PerThreadBag<Ctxt*>;
  using CtxtLocalQ = PerThreadBag<Ctxt*>;

  using UserCtxt = typename Base::UserCtxt;
  using PerThreadUserCtxt = typename Base::PerThreadUserCtxt;


  using Accumulator =  Galois::GAccumulator<size_t>;

  struct CreateCtxtExpandNhood {
    NhFunc& nhFunc;
    NhoodMgr& nhmgr;
    CtxtAlloc& ctxtAlloc;
    CtxtWL& ctxtWL;

    CreateCtxtExpandNhood (
        NhFunc& nhFunc,
        NhoodMgr& nhmgr,
        CtxtAlloc& ctxtAlloc,
        CtxtWL& ctxtWL)
      :
        nhFunc (nhFunc),
        nhmgr (nhmgr),
        ctxtAlloc (ctxtAlloc),
        ctxtWL (ctxtWL)
    {}

    GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (const T& active) const {
      Ctxt* ctxt = ctxtAlloc.allocate (1);
      assert (ctxt != NULL);
      // new (ctxt) Ctxt (active, nhmgr);
      //ctxtAlloc.construct (ctxt, Ctxt (active, nhmgr));
      ctxtAlloc.construct (ctxt, active, nhmgr);

      ctxtWL.get ().push_back (ctxt);

      Galois::Runtime::setThreadContext (ctxt);
      int tmp=0;
      // TODO: nhFunc should take only one arg, 
      // 2nd arg being passed due to compatibility with Deterministic executor
      nhFunc (ctxt->active, tmp); 
      Galois::Runtime::setThreadContext (NULL);
    }

  };

  struct FindInitSources {
    const SourceTest& sourceTest;
    CtxtWL& initSrc;
    Accumulator& nsrc;

    FindInitSources (
        const SourceTest& sourceTest, 
        CtxtWL& initSrc,
        Accumulator& nsrc)
      : 
        sourceTest (sourceTest), 
        initSrc (initSrc),
        nsrc (nsrc)
    {}

    GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (Ctxt* ctxt) const {
      assert (ctxt != NULL);
      // assume nhood of ctxt is already expanded

      // if (ctxt->isSrc ()) {
        // std::cout << "Testing source: " << ctxt->str () << std::endl;
      // }
      // if (sourceTest (ctxt)) {
        // std::cout << "Initial source: " << ctxt->str () << std::endl;
      // }
      if (sourceTest (ctxt) && ctxt->onWL.cas (false, true)) {
        initSrc.get ().push_back (ctxt);
        nsrc += 1;
      }
    }
  };


  struct ApplyOperator {

    typedef int tt_does_not_need_aborts;

    NhFunc& nhFunc;
    OpFunc& opFunc;
    NhoodMgr& nhmgr;
    const SourceTest& sourceTest;
    CtxtAlloc& ctxtAlloc;
    CtxtWL& addCtxtWL;
    CtxtLocalQ& ctxtLocalQ;
    CtxtDelQ& ctxtDelQ;
    PerThreadUserCtxt& perThrdUserCtxt;
    Accumulator& niter;

    ApplyOperator (
        NhFunc& nhFunc,
        OpFunc& opFunc,
        NhoodMgr& nhmgr,
        const SourceTest& sourceTest,
        CtxtAlloc& ctxtAlloc,
        CtxtWL& addCtxtWL,
        CtxtLocalQ& ctxtLocalQ,
        CtxtDelQ& ctxtDelQ,
        PerThreadUserCtxt& perThrdUserCtxt,
        Accumulator& niter)
      :
        nhFunc (nhFunc),
        opFunc (opFunc),
        nhmgr (nhmgr),
        sourceTest (sourceTest),
        ctxtAlloc (ctxtAlloc),
        addCtxtWL (addCtxtWL),
        ctxtLocalQ (ctxtLocalQ),
        ctxtDelQ (ctxtDelQ),
        perThrdUserCtxt (perThrdUserCtxt),
        niter (niter) 
    {}


    template <typename WL>
    GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (Ctxt* const in_src, WL& wl) {
      assert (in_src != NULL);

      ctxtLocalQ.get ().clear ();

      ctxtLocalQ.get ().push_back (in_src);


      for (unsigned local_iter = 0; 
          (local_iter < UNROLL_FACTOR) && !ctxtLocalQ.get ().empty (); ++local_iter ) {

        Ctxt* src = ctxtLocalQ.get ().front (); ctxtLocalQ.get ().pop_front ();

        // GALOIS_DEBUG ("Processing source: %s\n", src->str ().c_str ());
        if (debug && !sourceTest (src)) {
          std::cout << "Not found to be a source: " << src->str ()
            << std::endl;
          // abort ();
        }

        niter += 1;

        // addWL.get ().clear ();
        UserCtxt& userCtxt = *(perThrdUserCtxt.getLocal ());

        if (true || DEPRECATED::ForEachTraits<OpFunc>::NeedsPush) {
          userCtxt.resetPushBuffer ();
          userCtxt.resetAlloc ();
        }

        opFunc (src->active, userCtxt.data ()); 


        if (true || DEPRECATED::ForEachTraits<OpFunc>::NeedsPush) {

          addCtxtWL.get ().clear ();
          CreateCtxtExpandNhood addCtxt (nhFunc, nhmgr, ctxtAlloc, addCtxtWL);

          for (auto a = userCtxt.getPushBuffer ().begin ()
              , enda = userCtxt.getPushBuffer ().end (); a != enda; ++a) {


            addCtxt (*a);
          }

          for (auto c = addCtxtWL.get ().begin ()
              , endc = addCtxtWL.get ().end (); c != endc; ++c) {

            (*c)->findNewSources (sourceTest, wl);
            // // if is source add to workList;
            // if (sourceTest (*c) && (*c)->onWL.cas (false, true)) {
            // // std::cout << "Adding new source: " << *c << std::endl;
            // wl.push (*c);
            // }
          }
        }

        src->removeFromNhood ();

        src->findSrcInNhood (sourceTest, ctxtLocalQ.get ());

        //TODO: use a ref count type wrapper for Ctxt;
        ctxtDelQ.get ().push_back (src);

      }


      // add remaining to global wl
      // TODO: check onWL counter here
      for (auto c = ctxtLocalQ.get ().begin ()
          , endc = ctxtLocalQ.get ().end (); c != endc; ++c) {

        wl.push (*c);
      }


      while (ctxtDelQ.get ().size () >= DELETE_CONTEXT_SIZE) {

        Ctxt* c = ctxtDelQ.get ().front (); ctxtDelQ.get ().pop_front ();
        ctxtAlloc.destroy (c);
        ctxtAlloc.deallocate (c, 1); 
      }
    }

  };

  struct DelCtxt {

    CtxtAlloc& ctxtAlloc;

    explicit DelCtxt (CtxtAlloc& ctxtAlloc): ctxtAlloc (ctxtAlloc) {}

    void operator () (Ctxt* ctxt) const {
      ctxtAlloc.destroy (ctxt);
      ctxtAlloc.deallocate (ctxt, 1);
    }
  };

private:
  NhFunc nhFunc;
  OpFunc opFunc;
  // TODO: make cmp function of nhmgr thread local as well.
  NhoodMgr& nhmgr;
  SourceTest sourceTest;


public:

  KDGaddRemAsyncExec (
      const Cmp& cmp,
      const NhFunc& nhFunc,
      const OpFunc& opFunc,
      const ArgsTuple& argsTuple, 
      NhoodMgr& nhmgr,
      const SourceTest& sourceTest)
    :
      Base (cmp, nhFunc, HIDDEN::DummyExecFunc (), opFunc, argsTuple),
      nhmgr (nhmgr),
      sourceTest (sourceTest)
  {}


  template <typename R>
  void execute (const R& range) {
    CtxtAlloc ctxtAlloc;
    CtxtWL initCtxt;
    CtxtWL initSrc;

    Accumulator nInitSrc;
    Accumulator niter;

    Galois::TimeAccumulator t_create;
    Galois::TimeAccumulator t_find;
    Galois::TimeAccumulator t_for;
    Galois::TimeAccumulator t_destroy;

    t_create.start ();
    Galois::do_all_choice (
        range, 
        CreateCtxtExpandNhood (nhFunc, nhmgr, ctxtAlloc, initCtxt),
        std::make_tuple (
          Galois::loopname ("create_initial_contexts"),
          Galois::chunk_size<DEFAULT_CHUNK_SIZE> ()));

    t_create.stop ();

    t_find.start ();
    Galois::do_all_choice (makeLocalRange(initCtxt),
        FindInitSources (sourceTest, initSrc, nInitSrc)
        std::make_tuple (
          Galois::loopname ("find_initial_sources"),
          Galois::chunk_size<DEFAULT_CHUNK_SIZE> ()));

    t_find.stop ();

    std::cout << "Number of initial sources found: " << nInitSrc.reduce () 
      << std::endl;

    // AddWL addWL;
    PerThreadUserCtxt perThrdUserCtxt;
    CtxtWL addCtxtWL;
    CtxtDelQ ctxtDelQ;
    CtxtLocalQ ctxtLocalQ;

    typedef Galois::WorkList::dChunkedFIFO<OpFunc::CHUNK_SIZE, Ctxt*> SrcWL_ty;
    // typedef Galois::WorkList::AltChunkedFIFO<CHUNK_SIZE, Ctxt*> SrcWL_ty;
    // TODO: code to find global min goes here

    t_for.start ();
    Galois::for_each_local(initSrc,
        ApplyOperator (
          nhFunc,
          opFunc,
          nhmgr,
          sourceTest,
          ctxtAlloc,
          addCtxtWL,
          ctxtLocalQ,
          ctxtDelQ,
          perThrdUserCtxt,
          niter),
        Galois::loopname("apply_operator"), Galois::wl<SrcWL_ty>());
    t_for.stop ();

    t_destroy.start ();
    Galois::do_all_choice (makeLocalRange(ctxtDelQ),
				DelCtxt (ctxtAlloc), 
        std::make_tuple (
          Galois::loopname ("delete_all_ctxt"),
          Galois::chunk_size<DEFAULT_CHUNK_SIZE> ()));
    t_destroy.stop ();

    reportStat (Base::loopname, "Number of iterations: ", niter.reduce (), 0);

    reportStat (Base::loopname, "Time taken in creating intial contexts: ",   t_create.get (), 0);
    reportStat (Base::loopname, "Time taken in finding intial sources: ", t_find.get (), 0);
    reportStat (Base::loopname, "Time taken in for_each loop: ", t_for.get (), 0);
    reportStat (Base::loopname, "Time taken in destroying all the contexts: ", t_destroy.get (), 0);
  }
};


template <typename T, typename Cmp, typename NhFunc, typename OpFunc, typename Ctxt, typename ST, typename ArgsTuple> 
class KDGaddRemWindowExec public OrderedExecutorBase<T, Cmp, NhFunc, HIDDEN::DummyExecFunc, OpFunc, ArgsTuple, Ctxt> {

  using Base = OrderedExecutorBase<T, Cmp, NhFunc, HIDDEN::DummyExecFunc, OpFunc, ArgsTuple, Ctxt>;

  // important paramters
  // TODO: add capability to the interface to express these constants
  static const size_t DELETE_CONTEXT_SIZE = 1024;
  static const size_t UNROLL_FACTOR = OpFunc::UNROLL_FACTOR;

  static const unsigned DEFAULT_CHUNK_SIZE = 8;



  // typedef MapBasedNhoodMgr<T, Cmp> NhoodMgr;
  // typedef NhoodItem<T, Cmp, NhoodMgr> NItem;
  // typedef typename NItem::Ctxt Ctxt;

  using NhoodMgr =  typename Ctxt::NhoodMgr;

  using CtxtAlloc = typenme Base::CtxtAlloc;
  using CtxtWL = typenme Base::CtxtWL;

  using CtxtDelQ = PerThreadBag<Ctxt*>;
  using CtxtLocalQ = PerThreadBag<Ctxt*>;

  using UserCtxt = typename Base::UserCtxt;
  using PerThreadUserCtxt = typename Base::PerThreadUserCtxt;


  using Accumulator =  Galois::GAccumulator<size_t>;


  CtxtWL pending;
  CtxtWL sources;

  void beginRound (void) {
  }

  void expandNhoodPending (void) {
  }

public:

  void execute (void) {
    while (true) {

      beginRound ();

      expandNhoodPending ();

      if (sources.empty_all ()) {
        break;
      }
      
      applyOperator ();


    }
  }

};


template <typename R, typename Cmp, typename NhFunc, typename OpFunc, typename ST, typename ArgsTuple>
void for_each_ordered_lc_impl (const R& range, const Cmp& cmp, const NhFunc& nhFunc, const OpFunc& opFunc, const ST& sourceTest, const ArgsTuple& argsTuple) {

  typedef typename R::value_type T;

  typedef KDGaddRemContext<T, Cmp> Ctxt;
  typedef typename Ctxt::NhoodMgr NhoodMgr;
  typedef typename Ctxt::NItem NItem;
  typedef typename Ctxt::CtxtCmp  CtxtCmp;

  typedef KDGaddRemAsyncExec<T, Cmp, NhFunc, OpFunc, Ctxt, ST, ArgsTuple> Exec;

  CtxtCmp ctxtcmp (cmp);
  typename NItem::Factory factory(ctxtcmp);
  NhoodMgr nhmgr (factory);

  Exec e (cmp, nhFunc, opFunc, nhmgr, sourceTest, argsTuple);
  e.execute (range);
}

template <typename R, typename Cmp, typename NhFunc, typename OpFunc, typename StableTest>
void for_each_ordered_lc (const R& range, const Cmp& cmp, const NhFunc& nhFunc, const OpFunc& opFunc, const StableTest& stabilityTest, const char* loopname) {

  for_each_ordered_lc_impl (range, cmp, nhFunc, opFunc, SourceTest<StableTest> (stabilityTest), loopname);
}

template <typename R, typename Cmp, typename NhFunc, typename OpFunc>
void for_each_ordered_lc (const R& range, const Cmp& cmp, const NhFunc& nhFunc, const OpFunc& opFunc, const char* loopname) {

  for_each_ordered_lc_impl (range, cmp, nhFunc, opFunc, SourceTest<void> (), loopname);
}

} // end namespace Runtime
} // end namespace Galois

#endif //  GALOIS_RUNTIME_KDG_ADD_REM_H
