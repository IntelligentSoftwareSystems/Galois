/** TODO -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
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
 * @section Description
 *
 * TODO 
 *
 * @author <ahassaan@ices.utexas.edu>
 */
#ifndef GALOIS_RUNTIME_LC_ORDERED_H
#define GALOIS_RUNTIME_LC_ORDERED_H

#include "Galois/config.h"
#include "Galois/Accumulator.h"
#include "Galois/Atomic.h"
#include "Galois/gdeque.h"
#include "Galois/PriorityQueue.h"
#include "Galois/Timer.h"

#include "Galois/Runtime/Context.h"
#include "Galois/Runtime/DoAll.h"
#include "Galois/Runtime/ParallelWork.h"
#include "Galois/Runtime/PerThreadWorkList.h"
#include "Galois/Runtime/Range.h"
#include "Galois/Runtime/ll/gio.h"
#include "Galois/Runtime/ll/ThreadRWlock.h"
#include "Galois/Runtime/mm/Mem.h"

#include "llvm/ADT/SmallVector.h"

#include <iostream>
#include GALOIS_CXX11_STD_HEADER(unordered_map)

namespace Galois {
namespace Runtime {

static const bool debug = false;

template <typename Ctxt, typename CtxtCmp>
class NhoodItem: public LockManagerBase {
  typedef LockManagerBase Base;

public:
  // typedef Galois::ThreadSafeMinHeap<Ctxt*, CtxtCmp> PQ;
  typedef Galois::ThreadSafeOrderedSet<Ctxt*, CtxtCmp> PQ;

protected:
  PQ sharers;
  Lockable* lockable;

public:
  NhoodItem (Lockable* l, const CtxtCmp& ctxcmp):  sharers (ctxcmp), lockable (l) {}

  void add (const Ctxt* ctx) {

    assert (!sharers.find (const_cast<Ctxt*> (ctx)));
    sharers.push (const_cast<Ctxt*> (ctx));
  }

  bool isHighestPriority (const Ctxt* ctx) const {
    return !sharers.empty () && (sharers.top () == ctx);
  }

  Ctxt* getHighestPriority () const {
    if (sharers.empty ()) { 
      return NULL;

    } else {
      return sharers.top ();
    }
  }

  void remove (const Ctxt* ctx) {
    sharers.remove (const_cast<Ctxt*> (ctx));
    // XXX: may fail in parallel execution
    assert (!sharers.find (const_cast<Ctxt*> (ctx)));
  }

  void print () const { 
    // TODO
  }
  
  bool tryMappingTo (Lockable* l) {
    return Base::CASowner (l, NULL);
  }

  void clearMapping () {
    // release requires having owned the lock
    bool r = Base::tryLock (lockable);
    assert (r);
    Base::release (lockable);
  }

  // just for debugging
  const Lockable* getMapping () const {
    return lockable;
  }

  static NhoodItem* getOwner (Lockable* l) {
    LockManagerBase* o = LockManagerBase::getOwner (l);
    // assert (dynamic_cast<NhoodItem*> (o) != nullptr);
    return static_cast<NhoodItem*> (o);
  }


  struct Factory {

    typedef NhoodItem<Ctxt, CtxtCmp> NItem;
    typedef Galois::Runtime::MM::FSBGaloisAllocator<NItem> NItemAlloc;

    NItemAlloc niAlloc;
    CtxtCmp ctxcmp;

    explicit Factory (const CtxtCmp& ctxcmp): ctxcmp (ctxcmp) {}

    NItem* create (Lockable* l) {
      NItem* ni = niAlloc.allocate (1);
      assert (ni != nullptr);
      // XXX(ddn): Forwarding still wonky on XLC
#if !defined(__IBMCPP__) || __IBMCPP__ > 1210
      niAlloc.construct (ni, l, ctxcmp);
#else
      niAlloc.construct (ni, NItem(l, ctxcmp));
#endif

      return ni;
    }

    void destroy (NItem* ni) {
      // delete ni; ni = NULL;
      niAlloc.destroy (ni);
      niAlloc.deallocate (ni, 1);
      ni = NULL;
    }
  };


};

template<typename NItem>
class PtrBasedNhoodMgr: boost::noncopyable {
public:
  typedef typename NItem::Factory NItemFactory;

  typedef Galois::Runtime::MM::FSBGaloisAllocator<NItem> NItemAlloc;
  typedef Galois::Runtime::PerThreadVector<NItem*> NItemWL;

protected:
  NItemFactory factory;
  NItemWL allNItems;
  
public:
  PtrBasedNhoodMgr(const NItemFactory& f): factory (f) {}

  NItem& getNhoodItem (Lockable* l) {

    if (NItem::getOwner (l) == NULL) {
      // NItem* ni = new NItem (l, cmp);
      NItem* ni = factory.create (l);

      if (ni->tryMappingTo (l)) {
        allNItems.get ().push_back (ni);

      } else {
        factory.destroy (ni);
      }

      assert (NItem::getOwner (l) != NULL);
    }

    NItem* ret = NItem::getOwner (l);
    assert (ret != NULL);
    return *ret;
  }

  LocalRange<NItemWL> getAllRange (void) {
    return makeLocalRange (allNItems);
  }

  ~PtrBasedNhoodMgr() {
    resetAllNItems();
  }

protected:
  struct Reset {
    PtrBasedNhoodMgr* self; 
    void operator()(NItem* ni) {
      ni->clearMapping();
      self->factory.destroy(ni);
    }
  };

  void resetAllNItems() {
    Reset fn = { this };
    do_all_impl(makeLocalRange(allNItems), fn);
  }
};

template <typename NItem>
class MapBasedNhoodMgr: public PtrBasedNhoodMgr<NItem> {
public:
  typedef MapBasedNhoodMgr MyType;

  // typedef std::tr1::unordered_map<Lockable*, NItem> NhoodMap; 
  //
  typedef MM::SimpleBumpPtrWithMallocFallback<MM::FreeListHeap<MM::SystemBaseAlloc> > BasicHeap;
  typedef MM::ThreadAwarePrivateHeap<BasicHeap> PerThreadHeap;
  typedef MM::ExternRefGaloisAllocator<std::pair<Lockable*, NItem*>, PerThreadHeap> PerThreadAllocator;

  typedef std::unordered_map<
      Lockable*,
      NItem*,
      std::hash<Lockable*>,
      std::equal_to<Lockable*>,
      PerThreadAllocator
    > NhoodMap;

  typedef Galois::Runtime::LL::ThreadRWlock Lock_ty;
  typedef PtrBasedNhoodMgr<NItem> Base;

protected:
  PerThreadHeap heap;
  NhoodMap nhoodMap;
  Lock_ty map_mutex;

public:

  MapBasedNhoodMgr (const typename Base::NItemFactory& f): 
    Base (f),
    heap (),
    nhoodMap (8, std::hash<Lockable*> (), std::equal_to<Lockable*> (), PerThreadAllocator (&heap))

  {}

  NItem& getNhoodItem (Lockable* l) {

    map_mutex.readLock ();
      typename NhoodMap::iterator i = nhoodMap.find (l);

      if (i == nhoodMap.end ()) {
        // create the missing entry

        map_mutex.readUnlock ();

        map_mutex.writeLock ();
          // check again to avoid over-writing existing entry
          if (nhoodMap.find (l) == nhoodMap.end ()) {
            NItem* ni = Base::factory.create (l);
            Base::allNItems.get ().push_back (ni);
            nhoodMap.insert (std::make_pair (l, ni));

          }
        map_mutex.writeUnlock ();

        // read again now
        map_mutex.readLock ();
        i = nhoodMap.find (l);
      }

    map_mutex.readUnlock ();
    assert (i != nhoodMap.end ());
    assert (i->second != nullptr);

    return *(i->second);
    
  }


  ~MapBasedNhoodMgr () {
    Base::resetAllNItems ();
  }

};

template <typename T, typename Cmp>
class LCorderedContext: public SimpleRuntimeContext {

public:
  struct Comparator;

  typedef T value_type;
  typedef LCorderedContext MyType;
  typedef NhoodItem<MyType, typename MyType::Comparator> NItem;
  typedef PtrBasedNhoodMgr<NItem> NhoodMgr;
  typedef Galois::GAtomic<bool> AtomicBool;
  typedef Galois::gdeque<NItem*, 4> NhoodList;
  // typedef llvm::SmallVector<NItem*, 4> NhoodList;
  // typedef std::vector<NItem*> NhoodList;

  // TODO: fix visibility below
public:
  T active;
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

  GALOIS_ATTRIBUTE_PROF_NOINLINE virtual void subAcquire (Lockable* l) {
    NItem& nitem = nhmgr.getNhoodItem (l);

    assert (NItem::getOwner (l) == &nitem);

    nhood.push_back (&nitem);
    nitem.add (this);
    
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

  struct Comparator {
    const Cmp& cmp;

    explicit Comparator (const Cmp& cmp): cmp (cmp) {}

    inline bool operator () (const LCorderedContext* left, const LCorderedContext* right) const {
      assert (left != NULL);
      assert (right != NULL);
      return cmp (left->active, right->active);
    }
  };

};



template <typename StableTest>
struct SourceTest {

  StableTest stabilityTest;

  explicit SourceTest (const StableTest& stabilityTest)
    : stabilityTest (stabilityTest) {}

  template <typename Ctxt>
  bool operator () (const Ctxt* ctx) const {
    assert (ctx != NULL);
    return ctx->isSrc () && stabilityTest (ctx->active);
  }
};

template <>
struct SourceTest <void> {

  template <typename Ctxt>
  bool operator () (const Ctxt* ctx) const {
    assert (ctx != NULL);
    return ctx->isSrc ();
  }
};

// TODO: remove template parameters that can be passed to execute
template <typename OperFunc, typename NhoodFunc, typename Ctxt, typename SourceTest>
class LCorderedExec {

  // important paramters
  // TODO: add capability to the interface to express these constants
  static const size_t DELETE_CONTEXT_SIZE = 1024;
  static const size_t UNROLL_FACTOR = OperFunc::UNROLL_FACTOR;
  static const unsigned CHUNK_SIZE = OperFunc::CHUNK_SIZE;



  // typedef MapBasedNhoodMgr<T, Cmp> NhoodMgr;
  // typedef NhoodItem<T, Cmp, NhoodMgr> NItem;
  // typedef typename NItem::Ctxt Ctxt;

  typedef typename Ctxt::value_type T;
  typedef typename Ctxt::NhoodMgr NhoodMgr;

  typedef Galois::Runtime::MM::FSBGaloisAllocator<Ctxt> CtxtAlloc;
  typedef Galois::Runtime::PerThreadVector<Ctxt*> CtxtWL;
  typedef Galois::Runtime::PerThreadDeque<Ctxt*> CtxtDelQ;
  typedef Galois::Runtime::PerThreadDeque<Ctxt*> CtxtLocalQ;
  // typedef Galois::Runtime::PerThreadVector<T> AddWL;
  typedef Galois::Runtime::UserContextAccess<T> UserCtx;
  typedef Galois::Runtime::PerThreadStorage<UserCtx> PerThreadUserCtx;


  typedef Galois::GAccumulator<size_t> Accumulator;

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

    GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (const T& active) {
      Ctxt* ctx = ctxtAlloc.allocate (1);
      assert (ctx != NULL);
      // new (ctx) Ctxt (active, nhmgr);
      //ctxtAlloc.construct (ctx, Ctxt (active, nhmgr));
      ctxtAlloc.construct (ctx, active, nhmgr);

      ctxtWL.get ().push_back (ctx);

      Galois::Runtime::setThreadContext (ctx);
      int tmp=0;
      // TODO: nhoodVisitor should take only one arg, 
      // 2nd arg being passed due to compatibility with Deterministic executor
      nhoodVisitor (ctx->active, tmp); 
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

    GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (Ctxt* ctx) {
      assert (ctx != NULL);
      // assume nhood of ctx is already expanded

      // if (ctx->isSrc ()) {
        // std::cout << "Testing source: " << ctx->str () << std::endl;
      // }
      // if (sourceTest (ctx)) {
        // std::cout << "Initial source: " << ctx->str () << std::endl;
      // }
      if (sourceTest (ctx) && ctx->onWL.cas (false, true)) {
        initSrc.get ().push_back (ctx);
        nsrc += 1;
      }
    }
  };


  struct ApplyOperator {

    typedef int tt_does_not_need_aborts;

    OperFunc& op;
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
        OperFunc& op,
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
        userCtx.resetPushBuffer ();
        userCtx.resetAlloc ();
        op (src->active, userCtx.data ()); 

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

    void operator () (Ctxt* ctx) {
      ctxtAlloc.destroy (ctx);
      ctxtAlloc.deallocate (ctx, 1);
    }
  };

private:
  NhoodFunc nhoodVisitor;
  OperFunc operFunc;
  // TODO: make cmp function of nhmgr thread local as well.
  NhoodMgr& nhmgr;
  SourceTest sourceTest;


public:

  LCorderedExec (
      const NhoodFunc& nhoodVisitor,
      const OperFunc& operFunc,
      NhoodMgr& nhmgr,
      const SourceTest& sourceTest)
    :
      nhoodVisitor (nhoodVisitor),
      operFunc (operFunc),
      nhmgr (nhmgr),
      sourceTest (sourceTest)
  {}

  template <typename AI>
  void execute (AI abeg, AI aend, const char* loopname) {
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
    Galois::Runtime::do_all_impl(makeStandardRange(abeg, aend), 
				 CreateCtxtExpandNhood (nhoodVisitor, nhmgr, ctxtAlloc, initCtxt));
    //        "create_initial_contexts");
    t_create.stop ();

    t_find.start ();
    Galois::Runtime::do_all_impl(makeLocalRange(initCtxt),
				 FindInitSources (sourceTest, initSrc, nInitSrc));
    //       "find_initial_sources");
    t_find.stop ();

    std::cout << "Number of initial sources found: " << nInitSrc.reduce () 
      << std::endl;

    // AddWL addWL;
    PerThreadUserCtx perThUserCtx;
    CtxtWL addCtxtWL;
    CtxtDelQ ctxtDelQ;
    CtxtLocalQ ctxtLocalQ;

    typedef Galois::WorkList::dChunkedFIFO<CHUNK_SIZE, Ctxt*> SrcWL_ty;
    // TODO: code to find global min goes here

    t_for.start ();
    Galois::Runtime::for_each_impl<SrcWL_ty> (
        Galois::Runtime::makeLocalRange(initSrc),
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
        "apply_operator");
    t_for.stop ();

    t_destroy.start ();
    Galois::Runtime::do_all_impl(makeLocalRange(ctxtDelQ),
				 DelCtxt (ctxtAlloc)); //, "delete_all_ctxt");
    t_destroy.stop ();

    std::cout << "Number of iterations: " << niter.reduce () << std::endl;

    std::cout << "Time taken in creating intial contexts: " << t_create.get () << std::endl;
    std::cout << "Time taken in finding intial sources: " << t_find.get () << std::endl;
    std::cout << "Time taken in for_each loop: " << t_for.get () << std::endl;
    std::cout << "Time taken in destroying all the contexts: " << t_destroy.get () << std::endl;
  }
};

template <typename AI, typename Cmp, typename OperFunc, typename NhoodFunc, typename ST>
void for_each_ordered_lc_impl (AI abeg, AI aend, const Cmp& cmp, const NhoodFunc& nhoodVisitor, const OperFunc& operFunc, const ST& sourceTest, const char* loopname) {

  typedef typename std::iterator_traits<AI>::value_type T;

  typedef LCorderedContext<T, Cmp> Ctxt;
  typedef typename Ctxt::NhoodMgr NhoodMgr;
  typedef typename Ctxt::NItem NItem;
  typedef typename Ctxt::Comparator CtxtCmp;

  typedef LCorderedExec<OperFunc, NhoodFunc, Ctxt, ST> Exec;

  CtxtCmp ctxcmp (cmp);
  typename NItem::Factory factory(ctxcmp);
  NhoodMgr nhmgr (factory);

  Exec e (nhoodVisitor, operFunc, nhmgr, sourceTest);
  // e.template execute<CHUNK_SIZE> (abeg, aend);
  e.execute (abeg, aend, loopname);
}

template <typename AI, typename Cmp, typename OperFunc, typename NhoodFunc, typename StableTest>
void for_each_ordered_lc (AI abeg, AI aend, const Cmp& cmp, const NhoodFunc& nhoodVisitor, const OperFunc& operFunc, const StableTest& stabilityTest, const char* loopname) {

  for_each_ordered_lc_impl (abeg, aend, cmp, nhoodVisitor, operFunc, SourceTest<StableTest> (stabilityTest), loopname);
}

template <typename AI, typename Cmp, typename OperFunc, typename NhoodFunc>
void for_each_ordered_lc (AI abeg, AI aend, const Cmp& cmp, const NhoodFunc& nhoodVisitor, const OperFunc& operFunc, const char* loopname) {

  for_each_ordered_lc_impl (abeg, aend, cmp, nhoodVisitor, operFunc, SourceTest<void> (), loopname);
}

} // end namespace Runtime
} // end namespace Galois

#endif //  GALOIS_RUNTIME_LC_ORDERED_H
