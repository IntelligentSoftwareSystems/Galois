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

#ifndef GALOIS_RUNTIME_LC_ORDERED_H
#define GALOIS_RUNTIME_LC_ORDERED_H

#include "galois/GaloisForwardDecl.h"
#include "galois/Reduction.h"
#include "galois/Atomic.h"
#include "galois/gdeque.h"
#include "galois/PriorityQueue.h"
#include "galois/Timer.h"
#include "galois/AltBag.h"
#include "galois/PerThreadContainer.h"

#include "galois/worklists/WorkList.h"
#include "galois/runtime/Context.h"
#include "galois/runtime/OrderedLockable.h"
#include "galois/runtime/Executor_DoAll.h"
#include "galois/runtime/Range.h"
#include "galois/gIO.h"
#include "galois/runtime/Mem.h"


#include "llvm/ADT/SmallVector.h"

#include <iostream>
#include <unordered_map>

namespace galois {
namespace runtime {

static const bool debug = false;

template <typename Ctxt, typename CtxtCmp>
class NhoodItem: public OrdLocBase<NhoodItem<Ctxt, CtxtCmp>, Ctxt, CtxtCmp> {
  using Base = OrdLocBase<NhoodItem, Ctxt, CtxtCmp>;

public:
  using PQ =  galois::ThreadSafeOrderedSet<Ctxt*, CtxtCmp>;
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
class LCorderedContext: public SimpleRuntimeContext {

public:
  typedef T value_type;
  typedef LCorderedContext MyType;
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
  T active;
  // FIXME: nhood should be a set instead of list
  NhoodList nhood;
  NhoodMgr& nhmgr;
  GALOIS_ATTRIBUTE_ALIGN_CACHE_LINE AtomicBool onWL;

public:

  LCorderedContext (const T& active, NhoodMgr& nhmgr)
    :
      SimpleRuntimeContext (true), // to make acquire call virtual function sub_acquire
      active (active),
      nhood (),
      nhmgr (nhmgr),
      onWL (false)
  {}

  const T& getActive () const { return active; }

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
    assert (!nhood.empty ()); // TODO: remove later

    bool ret = true;

    // TODO: use const_iterator instead
    for (typename NhoodList::const_iterator n = nhood.begin ()
        , endn = nhood.end (); n != endn; ++n) {

      if (!(*n)->isHighestPriority (this)) {
        ret = false;
        break;
      }
    }

    return ret;
  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE void removeFromNhood () {
    for (typename NhoodList::iterator n = nhood.begin ()
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

    for (typename NhoodList::iterator n = nhood.begin ()
        , endn = nhood.end (); n != endn; ++n) {

      LCorderedContext* highest = (*n)->getHighestPriority ();
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

    for (typename NhoodList::iterator n = nhood.begin ()
        , endn = nhood.end (); n != endn; ++n) {

      LCorderedContext* highest = (*n)->getHighestPriority ();
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
template <typename OpFunc, typename NhoodFunc, typename Ctxt, typename SourceTest>
class LCorderedExec {

  // important paramters
  // TODO: add capability to the interface to express these constants
  static const size_t DELETE_CONTEXT_SIZE = 1024;
  static const size_t UNROLL_FACTOR = OpFunc::UNROLL_FACTOR;
  static const unsigned CHUNK_SIZE = OpFunc::CHUNK_SIZE;



  // typedef MapBasedNhoodMgr<T, Cmp> NhoodMgr;
  // typedef NhoodItem<T, Cmp, NhoodMgr> NItem;
  // typedef typename NItem::Ctxt Ctxt;

  typedef typename Ctxt::value_type T;
  typedef typename Ctxt::NhoodMgr NhoodMgr;

  typedef FixedSizeAllocator<Ctxt> CtxtAlloc;
  // typedef PerThreadBag<Ctxt*, 16> CtxtWL;
  typedef PerThreadVector<Ctxt*> CtxtWL;
  typedef PerThreadDeque<Ctxt*> CtxtDelQ;
  typedef PerThreadDeque<Ctxt*> CtxtLocalQ;
  // typedef galois::runtime::PerThreadVector<T> AddWL;
  typedef UserContextAccess<T> UserCtx;
  typedef substrate::PerThreadStorage<UserCtx> PerThreadUserCtx;


  typedef galois::GAccumulator<size_t> Accumulator;

  struct CreateCtxtExpandNhood {
    NhoodFunc& nhoodVisitor;
    NhoodMgr& nhmgr;
    CtxtAlloc& ctxtAlloc;
    CtxtWL& ctxtWL;

    CreateCtxtExpandNhood (
        NhoodFunc& nhoodVisitor,
        NhoodMgr& nhmgr,
        CtxtAlloc& ctxtAlloc,
        CtxtWL& ctxtWL)
      :
        nhoodVisitor (nhoodVisitor),
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

      galois::runtime::setThreadContext (ctxt);
      int tmp=0;
      // TODO: nhoodVisitor should take only one arg,
      // 2nd arg being passed due to compatibility with Deterministic executor
      nhoodVisitor (ctxt->active, tmp);
      galois::runtime::setThreadContext (NULL);
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

    OpFunc& op;
    NhoodFunc& nhoodVisitor;
    NhoodMgr& nhmgr;
    const SourceTest& sourceTest;
    CtxtAlloc& ctxtAlloc;
    CtxtWL& addCtxtWL;
    CtxtLocalQ& ctxtLocalQ;
    CtxtDelQ& ctxtDelQ;
    PerThreadUserCtx& perThUserCtx;
    Accumulator& niter;

    ApplyOperator (
        OpFunc& op,
        NhoodFunc& nhoodVisitor,
        NhoodMgr& nhmgr,
        const SourceTest& sourceTest,
        CtxtAlloc& ctxtAlloc,
        CtxtWL& addCtxtWL,
        CtxtLocalQ& ctxtLocalQ,
        CtxtDelQ& ctxtDelQ,
        PerThreadUserCtx& perThUserCtx,
        Accumulator& niter)
      :
        op (op),
        nhoodVisitor (nhoodVisitor),
        nhmgr (nhmgr),
        sourceTest (sourceTest),
        ctxtAlloc (ctxtAlloc),
        addCtxtWL (addCtxtWL),
        ctxtLocalQ (ctxtLocalQ),
        ctxtDelQ (ctxtDelQ),
        perThUserCtx (perThUserCtx),
        niter (niter)
    {}


    template <typename WL>
    GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (Ctxt* const in_src, WL& wl) {
      assert (in_src != NULL);

      ctxtLocalQ.get ().clear ();

      ctxtLocalQ.get ().push_back (in_src);

      unsigned local_iter = 0;

      while ((local_iter < UNROLL_FACTOR) && !ctxtLocalQ.get ().empty ()) {

        ++local_iter;

        Ctxt* src = ctxtLocalQ.get ().front (); ctxtLocalQ.get ().pop_front ();

        // GALOIS_DEBUG ("Processing source: %s\n", src->str ().c_str ());
        if (debug && !sourceTest (src)) {
          std::cout << "Not found to be a source: " << src->str ()
            << std::endl;
          // abort ();
        }

        niter += 1;

        // addWL.get ().clear ();
        UserCtx& userCtx = *(perThUserCtx.getLocal ());

        if (true || DEPRECATED::ForEachTraits<OpFunc>::NeedsPush) {
          userCtx.resetPushBuffer ();
          userCtx.resetAlloc ();
        }

        op (src->active, userCtx.data ());


        if (true || DEPRECATED::ForEachTraits<OpFunc>::NeedsPush) {

          addCtxtWL.get ().clear ();
          CreateCtxtExpandNhood addCtxt (nhoodVisitor, nhmgr, ctxtAlloc, addCtxtWL);

          // for (typename AddWL::local_iterator a = addWL.get ().begin ()
          // , enda = addWL.get ().end (); a != enda; ++a) {
          for (typename UserCtx::PushBufferTy::iterator a = userCtx.getPushBuffer ().begin ()
              , enda = userCtx.getPushBuffer ().end (); a != enda; ++a) {


            addCtxt (*a);
          }

          for (typename CtxtWL::local_iterator c = addCtxtWL.get ().begin ()
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
      for (typename CtxtLocalQ::local_iterator c = ctxtLocalQ.get ().begin ()
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
  NhoodFunc nhoodVisitor;
  OpFunc operFunc;
  // TODO: make cmp function of nhmgr thread local as well.
  NhoodMgr& nhmgr;
  SourceTest sourceTest;


public:

  LCorderedExec (
      const NhoodFunc& nhoodVisitor,
      const OpFunc& operFunc,
      NhoodMgr& nhmgr,
      const SourceTest& sourceTest)
    :
      nhoodVisitor (nhoodVisitor),
      operFunc (operFunc),
      nhmgr (nhmgr),
      sourceTest (sourceTest)
  {}

  template <typename R>
  void execute (const R& range, const char* loopname) {
    CtxtAlloc ctxtAlloc;
    CtxtWL initCtxt;
    CtxtWL initSrc;

    Accumulator nInitSrc;
    Accumulator niter;

    galois::TimeAccumulator t_create;
    galois::TimeAccumulator t_find;
    galois::TimeAccumulator t_for;
    galois::TimeAccumulator t_destroy;

    t_create.start ();
    galois::runtime::do_all_gen(
        range,
				CreateCtxtExpandNhood (nhoodVisitor, nhmgr, ctxtAlloc, initCtxt),
        std::make_tuple(
          galois::loopname("create_initial_contexts")));
    t_create.stop ();

    t_find.start ();
    galois::runtime::do_all_gen(makeLocalRange(initCtxt),
				 FindInitSources (sourceTest, initSrc, nInitSrc),
         std::make_tuple(
           galois::loopname("find_initial_sources")));
    //       "find_initial_sources");
    t_find.stop ();

    std::cout << "Number of initial sources found: " << nInitSrc.reduce ()
      << std::endl;

    // AddWL addWL;
    PerThreadUserCtx perThUserCtx;
    CtxtWL addCtxtWL;
    CtxtDelQ ctxtDelQ;
    CtxtLocalQ ctxtLocalQ;

    typedef galois::worklists::dChunkedFIFO<CHUNK_SIZE, Ctxt*> SrcWL_ty;
    // typedef galois::worklists::AltChunkedFIFO<CHUNK_SIZE, Ctxt*> SrcWL_ty;
    // TODO: code to find global min goes here

    t_for.start ();
    galois::for_each(galois::iterate(initSrc),
        ApplyOperator (
          operFunc,
          nhoodVisitor,
          nhmgr,
          sourceTest,
          ctxtAlloc,
          addCtxtWL,
          ctxtLocalQ,
          ctxtDelQ,
          perThUserCtx,
          niter),
        galois::loopname("apply_operator"),
        galois::wl<SrcWL_ty>());
    t_for.stop ();

    t_destroy.start ();
    galois::runtime::do_all_gen(makeLocalRange(ctxtDelQ),
				 DelCtxt (ctxtAlloc),
         std::make_tuple(galois::loopname("delete_all_ctxt"))); //, "delete_all_ctxt");
    t_destroy.stop ();

  }
};

template <typename R, typename Cmp, typename OpFunc, typename NhoodFunc, typename ST>
void for_each_ordered_lc_impl (const R& range, const Cmp& cmp, const NhoodFunc& nhoodVisitor, const OpFunc& operFunc, const ST& sourceTest, const char* loopname) {

  typedef typename R::value_type T;

  typedef LCorderedContext<T, Cmp> Ctxt;
  typedef typename Ctxt::NhoodMgr NhoodMgr;
  typedef typename Ctxt::NItem NItem;
  typedef typename Ctxt::CtxtCmp  CtxtCmp;

  typedef LCorderedExec<OpFunc, NhoodFunc, Ctxt, ST> Exec;

  CtxtCmp ctxtcmp (cmp);
  typename NItem::Factory factory(ctxtcmp);
  NhoodMgr nhmgr (factory);

  Exec e (nhoodVisitor, operFunc, nhmgr, sourceTest);
  // e.template execute<CHUNK_SIZE> (abeg, aend);
  e.execute (range, loopname);
}

template <typename R, typename Cmp, typename OpFunc, typename NhoodFunc, typename StableTest>
void for_each_ordered_lc (const R& range, const Cmp& cmp, const NhoodFunc& nhoodVisitor, const OpFunc& operFunc, const StableTest& stabilityTest, const char* loopname) {

  for_each_ordered_lc_impl (range, cmp, nhoodVisitor, operFunc, SourceTest<StableTest> (stabilityTest), loopname);
}

template <typename R, typename Cmp, typename OpFunc, typename NhoodFunc>
void for_each_ordered_lc (const R& range, const Cmp& cmp, const NhoodFunc& nhoodVisitor, const OpFunc& operFunc, const char* loopname) {

  for_each_ordered_lc_impl (range, cmp, nhoodVisitor, operFunc, SourceTest<void> (), loopname);
}

} // end namespace runtime
} // end namespace galois

#endif //  GALOIS_RUNTIME_LC_ORDERED_H
