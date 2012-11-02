/** TODO -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
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

#include "Galois/Accumulator.h"
#include "Galois/Galois.h"
#include "Galois/Timer.h"
#include "Galois/Atomic.h"
#include "Galois/gdeque.h"
#include "Galois/PriorityQueue.h"


#include "Galois/Runtime/PerThreadWorkList.h"
#include "Galois/Runtime/DoAllCoupled.h"
#include "Galois/Runtime/mm/Mem.h"
#include "Galois/Runtime/Context.h"
#include "Galois/Runtime/ThreadRWlock.h"
#include "Galois/Runtime/ll/gio.h"

#include <iostream>
#include <tr1/unordered_map>


namespace GaloisRuntime {

template <typename Ctxt, typename CtxtCmp>
class NhoodItem {

public:
  // typedef Galois::ThreadSafeMinHeap<Ctxt*, CtxtCmp> PQ;
  typedef Galois::ThreadSafeOrderedSet<Ctxt*, CtxtCmp> PQ;

protected:
  Lockable* lockable;
  PQ sharers;

public:

  template <typename Cmp>
  NhoodItem (Lockable* l, const Cmp& cmp): lockable (l), sharers (CtxtCmp (cmp)) {}

  Lockable* getLockable () const { return lockable; }

  void add (const Ctxt* ctxt) {

    assert (!sharers.find (const_cast<Ctxt*> (ctxt)));
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


template <typename T, typename Cmp, typename NhoodMgr_tp>
class LCorderedContext: public SimpleRuntimeContext {


public:
  struct Comparator;

  typedef T value_type;
  typedef NhoodMgr_tp NhoodMgr;
  typedef LCorderedContext MyType;
  typedef NhoodItem<MyType, typename MyType::Comparator> NItem;
  // typedef Galois::gdeque<NItem*> NhoodList;
  typedef std::vector<NItem*> NhoodList;
  typedef Galois::GAtomic<bool> AtomicBool;

  // TODO: fix visibility below
public:
  GALOIS_ATTRIBUTE_ALIGN_CACHE_LINE AtomicBool onWL;
  T active;
  NhoodMgr& nhmgr;
  NhoodList nhood;


public:

  LCorderedContext (const T& active, NhoodMgr& nhmgr)
    : 
      SimpleRuntimeContext (true), // to make acquire call virtual function sub_acquire
      active (active), 
      nhmgr (nhmgr), 
      nhood (), 
      onWL (false) 
  {}

  virtual void sub_acquire (Lockable* l) {
    NItem& nitem = nhmgr.getNhoodItem (l);

    assert (nitem.getLockable () == l);

    nhood.push_back (&nitem);
    nitem.add (this);
    
  }

  bool isSrc () const {
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

  void removeFromNhood () {
    for (typename NhoodList::iterator n = nhood.begin ()
        , endn = nhood.end (); n != endn; ++n) {

      (*n)->remove (this);
    }
  }

  std::string str () const {
    std::stringstream ss;
    ss << "[" << this << ": " << active << "]";
    return ss.str ();
  }

  template <typename SourceTest, typename WL>
  void findNewSources (const SourceTest& srcTest, WL& wl) {

    for (typename NhoodList::iterator n = nhood.begin ()
        , endn = nhood.end (); n != endn; ++n) {

      LCorderedContext* highest = (*n)->getHighestPriority ();
      if ((highest != NULL) 
          && srcTest (highest) 
          && highest->onWL.cas (false, true)) {

        // GALOIS_DEBUG_PRINT ("Adding found source: %s\n", highest->str ().c_str ());
        wl.push (highest);
      }
    }
  }

  struct Comparator {
    Cmp cmp;

    explicit Comparator (const Cmp& cmp): cmp (cmp) {}

    inline bool operator () (const LCorderedContext* left, const LCorderedContext* right) const {
      assert (left != NULL);
      assert (right != NULL);
      return cmp (left->active, right->active);
    }
  };

};


template <typename T, typename Cmp>
class PtrBasedNhoodMgr {
public:
  typedef PtrBasedNhoodMgr MyType;
  typedef LCorderedContext<T, Cmp, MyType> Ctxt;
  typedef typename Ctxt::NItem NItem;

  typedef GaloisRuntime::MM::FSBGaloisAllocator<NItem> NItemAlloc;
  typedef GaloisRuntime::PerThreadVector<NItem*> NItemWL;


  Cmp cmp;
  NItemAlloc niAlloc;
  NItemWL allNItems;
  

  PtrBasedNhoodMgr (const Cmp& cmp): cmp (cmp) {}

  NItem& getNhoodItem (Lockable* l) {

    if (l->auxPtr.getValue () == NULL) {
      NItem* nv = niAlloc.allocate (1);
      assert (nv != NULL);
      niAlloc.construct (nv, NItem (l, cmp));
      // NItem* nv = new NItem (l, cmp);

      if (l->auxPtr.CAS (NULL, nv)) {
        allNItems.get ().push_back (nv);
      } else {
        // delete nv; nv = NULL;
        niAlloc.destroy (nv);
        niAlloc.deallocate (nv, 1);
        nv = NULL;
      }


      assert (l->auxPtr.getValue () != NULL);
    }

    NItem* ret = static_cast<NItem*> (l->auxPtr.getValue ());
    assert (ret != NULL);
    return *ret;
  }

private:

  struct Reset {
    NItemAlloc& niAlloc;

    Reset (NItemAlloc& niAlloc): niAlloc (niAlloc) {}

    void operator () (NItem* ni) {
      ni->getLockable ()->auxPtr.setValue (NULL);
      niAlloc.destroy (ni);
      niAlloc.deallocate (ni, 1);
    }
  };

  void resetAll () {
    Galois::do_all (allNItems.begin_all (), allNItems.end_all (),
        Reset (niAlloc), "reset_NItems");
  }

public:
  ~PtrBasedNhoodMgr () {
    resetAll ();
  }



};



template <typename T, typename Cmp>
class MapBasedNhoodMgr {
public:
  typedef MapBasedNhoodMgr MyType;
  typedef LCorderedContext<T, Cmp, MyType> Ctxt;
  typedef typename Ctxt::NItem NItem;
  // typedef std::tr1::unordered_map<Lockable*, NItem> NhoodMap; 
  //
  typedef MM::SimpleBumpPtrWithMallocFallback<MM::FreeListHeap<MM::SystemBaseAlloc> > BasicHeap;
  typedef MM::ThreadAwarePrivateHeap<BasicHeap> PerThreadHeap;
  typedef MM::ExternRefGaloisAllocator<std::pair<Lockable*, NItem>, PerThreadHeap> PerThreadAllocator;

  typedef std::tr1::unordered_map<
      Lockable*,
      NItem,
      std::tr1::hash<Lockable*>,
      std::equal_to<Lockable*>,
      PerThreadAllocator
    > NhoodMap;

  typedef GaloisRuntime::ThreadRWlock Lock_ty;

protected:
  Cmp cmp;
  PerThreadHeap heap;
  NhoodMap nhoodMap;

  Lock_ty map_mutex;

public:

  MapBasedNhoodMgr (const Cmp& cmp): 
    cmp (cmp),
    heap (),
    nhoodMap (8, std::tr1::hash<Lockable*> (), std::equal_to<Lockable*> (), PerThreadAllocator (&heap))

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
            nhoodMap.insert (std::make_pair (l, NItem (l, cmp)));
          }
        map_mutex.writeUnlock ();

        // read again now
        map_mutex.readLock ();
        i = nhoodMap.find (l);
      }

    map_mutex.readUnlock ();

    return i->second;
    
  }

  // NItem& getNhoodItem (Lockable* l) {
// 
    // map_mutex.lock ();
      // typename NhoodMap::iterator i = nhoodMap.find (l);
      // if (i == nhoodMap.end ()) {
        // nhoodMap.insert (std::make_pair (l, NItem (cmp)));
        // i = nhoodMap.find (l);
        // assert (i != nhoodMap.end ());
      // }
// 
    // map_mutex.unlock ();
// 
    // return i->second;
// 
  // }

};




template <typename Ctxt, typename StableTest>
struct UnstableSourceTest {

  StableTest stabilityTest;

  explicit UnstableSourceTest (const StableTest& stabilityTest)
    : stabilityTest (stabilityTest) {}

  bool operator () (const Ctxt* ctxt) const {
    assert (ctxt != NULL);
    return ctxt->isSrc () && stabilityTest (ctxt->active);
  }
};

template <typename Ctxt>
struct StableSourceTest {
  bool operator () (const Ctxt* ctxt) const {
    assert (ctxt != NULL);
    return ctxt->isSrc ();
  }
};


// TODO: remove template parameters that can be passed to execute
template <typename OperFunc, typename NhoodFunc, typename Ctxt, typename SourceTest>
class LCorderedExec {


  // typedef MapBasedNhoodMgr<T, Cmp> NhoodMgr;
  // typedef NhoodItem<T, Cmp, NhoodMgr> NItem;
  // typedef typename NItem::Ctxt Ctxt;

  typedef typename Ctxt::value_type T;
  typedef typename Ctxt::NhoodMgr NhoodMgr;

  typedef GaloisRuntime::MM::FSBGaloisAllocator<Ctxt> CtxtAlloc;
  typedef GaloisRuntime::PerThreadVector<Ctxt*> CtxtWL;
  typedef GaloisRuntime::PerThreadVector<T> AddWL;


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

    void operator () (const T& active) {
      Ctxt* ctxt = ctxtAlloc.allocate (1);
      assert (ctxt != NULL);
      // new (ctxt) Ctxt (active, nhmgr);
      ctxtAlloc.construct (ctxt, Ctxt (active, nhmgr));

      ctxtWL.get ().push_back (ctxt);

      GaloisRuntime::setThreadContext (ctxt);
      nhoodVisitor (ctxt->active);
      GaloisRuntime::setThreadContext (NULL);
    }

  };

  struct FindInitSources {

    SourceTest& sourceTest;
    CtxtWL& initSrc;
    Accumulator& nsrc;

    FindInitSources (
        SourceTest& sourceTest, 
        CtxtWL& initSrc,
        Accumulator& nsrc)
      : 
        sourceTest (sourceTest), 
        initSrc (initSrc),
        nsrc (nsrc)
    {}

    void operator () (Ctxt* ctxt) {
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

    OperFunc& op;
    NhoodFunc& nhoodVisitor;
    NhoodMgr& nhmgr;
    SourceTest& sourceTest;
    CtxtAlloc& ctxtAlloc;
    CtxtWL& delCtxtWL;
    CtxtWL& addCtxtWL;
    AddWL& addWL;
    Accumulator& niter;

    ApplyOperator (
        OperFunc& op,
        NhoodFunc& nhoodVisitor,
        NhoodMgr& nhmgr,
        SourceTest& sourceTest,
        CtxtAlloc& ctxtAlloc,
        CtxtWL& delCtxtWL,
        CtxtWL& addCtxtWL,
        AddWL& addWL,
        Accumulator& niter)
      :
        op (op),
        nhoodVisitor (nhoodVisitor),
        nhmgr (nhmgr),
        sourceTest (sourceTest),
        ctxtAlloc (ctxtAlloc),
        delCtxtWL (delCtxtWL),
        addCtxtWL (addCtxtWL),
        addWL (addWL),
        niter (niter) 
    {}


    template <typename WL>
    void operator () (Ctxt* src, WL& wl) {
      assert (src != NULL);


      // GALOIS_DEBUG_PRINT ("Processing source: %s\n", src->str ().c_str ());
      if (sourceTest (src)) {
        niter += 1;

        addWL.get ().clear ();
        op (src->active, addWL.get ()); 

        addCtxtWL.get ().clear ();
        CreateCtxtExpandNhood addCtxt (nhoodVisitor, nhmgr, ctxtAlloc, addCtxtWL);

        for (typename AddWL::local_iterator a = addWL.get ().begin ()
            , enda = addWL.get ().end (); a != enda; ++a) {

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
        src->findNewSources (sourceTest, wl);

        // ctxtAlloc.destroy (src);
        // ctxtAlloc.deallocate (src, 1); 
        // src = NULL;
        delCtxtWL.get ().push_back (src);

      } else {
        std::cout << "Not found to be a source: " << src->str ()
           << std::endl;
        // abort ();
      }

    }

  };

  struct DelCtxt {

    CtxtAlloc& ctxtAlloc;

    explicit DelCtxt (CtxtAlloc& ctxtAlloc): ctxtAlloc (ctxtAlloc) {}

    void operator () (Ctxt* ctxt) {
      ctxtAlloc.destroy (ctxt);
      ctxtAlloc.deallocate (ctxt, 1);
    }
  };

private:
  OperFunc operFunc;
  NhoodFunc nhoodVisitor;
  NhoodMgr& nhmgr;
  SourceTest sourceTest;


public:

  LCorderedExec (
      OperFunc operFunc,
      NhoodFunc nhoodVisitor,
      NhoodMgr& nhmgr,
      SourceTest sourceTest)
    :
      operFunc (operFunc),
      nhoodVisitor (nhoodVisitor),
      nhmgr (nhmgr),
      sourceTest (sourceTest)
  {}

  template <const unsigned CHUNK_SIZE, typename AI>
  void execute (AI abeg, AI aend) {
    CtxtAlloc ctxtAlloc;
    CtxtWL initCtxt;
    CtxtWL initSrc;

    Accumulator nInitSrc;
    Accumulator niter;

    Galois::do_all (abeg, aend, 
        CreateCtxtExpandNhood (nhoodVisitor, nhmgr, ctxtAlloc, initCtxt),
        "create_initial_contexts");

    Galois::do_all (initCtxt.begin_all (), initCtxt.end_all (),
        FindInitSources (sourceTest, initSrc, nInitSrc),
        "find_initial_sources");

    std::cout << "Number of initial sources found: " << nInitSrc.reduce () 
      << std::endl;

    AddWL addWL;
    CtxtWL addCtxtWL;
    CtxtWL delCtxtWL;

    typedef GaloisRuntime::WorkList::dChunkedFIFO<CHUNK_SIZE, Ctxt*> SrcWL_ty;
    // TODO: code to find global min goes here
    Galois::for_each<SrcWL_ty> (initSrc.begin_all (), initSrc.end_all (),
        ApplyOperator (
          operFunc,
          nhoodVisitor,
          nhmgr,
          sourceTest,
          ctxtAlloc,
          addCtxtWL,
          delCtxtWL,
          addWL,
          niter),
        "apply_operator");

    Galois::do_all (delCtxtWL.begin_all (), delCtxtWL.end_all (),
        DelCtxt (ctxtAlloc), "delete_all_ctxt");

    std::cout << "Number of iterations: " << niter.reduce () << std::endl;

  }


  

};

template <const unsigned CHUNK_SIZE, typename AI, typename Cmp, typename OperFunc, typename NhoodFunc>
void for_each_ordered_lc (AI abeg, AI aend, Cmp cmp, OperFunc operFunc, NhoodFunc nhoodVisitor) {

  typedef typename std::iterator_traits<AI>::value_type T;

  // typedef MapBasedNhoodMgr<T, Cmp> NhoodMgr;
  typedef PtrBasedNhoodMgr<T, Cmp> NhoodMgr;

  typedef typename NhoodMgr::Ctxt Ctxt;
  typedef typename NhoodMgr::NItem NItem;
  typedef StableSourceTest<Ctxt> SourceTest;

  typedef LCorderedExec<OperFunc, NhoodFunc, Ctxt, SourceTest> Exec;

  NhoodMgr nhmgr (cmp);

  Exec e (operFunc, nhoodVisitor, nhmgr, SourceTest ());
  e.template execute<CHUNK_SIZE> (abeg, aend);
}

template <const unsigned CHUNK_SIZE, typename AI, typename Cmp, typename OperFunc, typename NhoodFunc, typename StableTest>
void for_each_ordered_lc (AI abeg, AI aend, Cmp cmp, OperFunc operFunc, NhoodFunc nhoodVisitor, StableTest stabilityTest) {

  typedef typename std::iterator_traits<AI>::value_type T;

  // typedef MapBasedNhoodMgr<T, Cmp> NhoodMgr;
  typedef PtrBasedNhoodMgr<T, Cmp> NhoodMgr;

  typedef typename NhoodMgr::Ctxt Ctxt;
  typedef typename NhoodMgr::NItem NItem;
  typedef UnstableSourceTest<Ctxt, StableTest> SourceTest;

  typedef LCorderedExec<OperFunc, NhoodFunc, Ctxt, SourceTest> Exec;

  NhoodMgr nhmgr (cmp);

  Exec e (operFunc, nhoodVisitor, nhmgr, SourceTest (stabilityTest));
  e.template execute<CHUNK_SIZE> (abeg, aend);
}

}
#endif //  GALOIS_RUNTIME_LC_ORDERED_H
