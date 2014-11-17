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
 
#ifndef GALOIS_RUNTIME_DAGEXEC_H
#define GALOIS_RUNTIME_DAGEXEC_H

#include "Galois/config.h"
#include "Galois/GaloisForwardDecl.h"
#include "Galois/Accumulator.h"
#include "Galois/Atomic.h"
#include "Galois/gdeque.h"
#include "Galois/PriorityQueue.h"
#include "Galois/Timer.h"
#include "Galois/DoAllWrap.h"

#include "Galois/Runtime/Context.h"
#include "Galois/Runtime/Executor_DoAll.h"
#include "Galois/Runtime/LCordered.h"
#include "Galois/Runtime/PerThreadContainer.h"
#include "Galois/Runtime/ll/gio.h"
#include "Galois/Runtime/ll/ThreadRWlock.h"
#include "Galois/Runtime/mm/Mem.h"

#include "llvm/Support/CommandLine.h"

#include <atomic>

namespace Galois {
namespace Runtime {

namespace {

template <typename Ctxt>
struct DAGnhoodItem: public LockManagerBase {

public:
  typedef LockManagerBase Base;
  typedef Galois::ThreadSafeOrderedSet<Ctxt*, std::less<Ctxt*> > SharerSet;

  Lockable* lockable;
  SharerSet sharers;

public:
  explicit DAGnhoodItem (Lockable* l): lockable (l), sharers () {}

  void addSharer (Ctxt* ctxt) {
    sharers.push (ctxt);
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

  static DAGnhoodItem* getOwner (Lockable* l) {
    LockManagerBase* o = LockManagerBase::getOwner (l);
    // assert (dynamic_cast<DAGnhoodItem*> (o) != nullptr);
    return static_cast<DAGnhoodItem*> (o);
  }

  struct Factory {

    typedef DAGnhoodItem<Ctxt> NItem;
    typedef MM::FixedSizeAllocator<NItem> NItemAlloc;

    NItemAlloc niAlloc;

    NItem* create (Lockable* l) {
      NItem* ni = niAlloc.allocate (1);
      assert (ni != nullptr);
      niAlloc.construct (ni, l);
      return ni;
    }

    void destroy (NItem* ni) {
      // delete ni; ni = NULL;
      niAlloc.destroy (ni);
      niAlloc.deallocate (ni, 1);
    }
  };
  
};


template <typename Ctxt>
struct DAGnhoodItemRW: public LockManagerBase {

public:
  typedef LockManagerBase Base;
  typedef Galois::ThreadSafeOrderedSet<Ctxt*, std::less<Ctxt*> > SharerSet;

  Lockable* lockable;
  SharerSet writers;
  SharerSet readers;

  explicit DAGnhoodItemRW (Lockable* l): lockable (l) {}

  void addReader (Ctxt* ctxt) {
    readers.push (ctxt);
  }

  void addWriter (Ctxt* ctxt) { 
    writers.push (ctxt);
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

  static DAGnhoodItemRW* getOwner (Lockable* l) {
    LockManagerBase* o = LockManagerBase::getOwner (l);
    // assert (dynamic_cast<DAGnhoodItemRW*> (o) != nullptr);
    return static_cast<DAGnhoodItemRW*> (o);
  }

  struct Factory {

    typedef DAGnhoodItemRW<Ctxt> NItem;
    typedef MM::FixedSizeAllocator<NItem> NItemAlloc;

    NItemAlloc niAlloc;

    NItem* create (Lockable* l) {
      NItem* ni = niAlloc.allocate (1);
      assert (ni != nullptr);
      niAlloc.construct (ni, l);
      return ni;
    }

    void destroy (NItem* ni) {
      // delete ni; ni = NULL;
      niAlloc.destroy (ni);
      niAlloc.deallocate (ni, 1);
    }
  };

};


template <typename T, typename Derived, typename NItem_tp>
struct DAGcontextBase: public SimpleRuntimeContext {

  typedef NItem_tp NItem;
  typedef PtrBasedNhoodMgr<NItem> NhoodMgr;

public:
  typedef Galois::ThreadSafeOrderedSet<Derived*, std::less<Derived*> > AdjSet;
  // TODO: change AdjList to array for quicker iteration
  typedef Galois::gdeque<Derived*, 64> AdjList;
  typedef std::atomic<int> ParCounter;

  // GALOIS_ATTRIBUTE_ALIGN_CACHE_LINE ParCounter inDeg;
  ParCounter inDeg;
  int origInDeg;
  NhoodMgr& nhmgr;
  T elem;
  unsigned outDeg;
  Derived** outNeighbors;

  AdjSet adjSet;

public:
  explicit DAGcontextBase (const T& t, NhoodMgr& nhmgr): 
    SimpleRuntimeContext (true), // true to call subAcquire
    inDeg (0),
    origInDeg (0), 
    nhmgr (nhmgr),
    elem (t),
    outDeg (0),
    outNeighbors (nullptr)
  {}

  const T& getElem () const { return elem; }

  //! returns true on success
  bool addOutNeigh (Derived* that) {
    return adjSet.push (that);
  }

  void addInNeigh (Derived* that) {
    int* x = const_cast<int*> (&origInDeg);
    ++(*x);
    ++inDeg;
  }

  template <typename A>
  void finalizeAdj (A& allocator) {
    assert (outDeg == 0);
    assert (outNeighbors == nullptr);

    outDeg = adjSet.size ();

    if (outDeg != 0) {
      outNeighbors = allocator.allocate (outDeg);

      unsigned j = 0;
      for (auto i = adjSet.begin (), i_end = adjSet.end (); 
          i != i_end; ++i) {

        outNeighbors[j++] = *i;
      }

      assert (j == outDeg);
      adjSet.clear ();
    }
  }


  Derived** neighbor_begin (void) {
    return &outNeighbors[0];
  }

  Derived** neighbor_end (void) {
    return &outNeighbors[outDeg];
  }

  bool removeLastInNeigh (Derived* that) {
    assert (inDeg >= 0);
    return ((--inDeg) == 0);
  }

  bool isSrc (void) const {
    return inDeg == 0;
  }

  void reset (void) {
    inDeg = origInDeg;
  }

  template <typename A>
  void reclaimAlloc (A& allocator) {
    if (outDeg != 0) {
      allocator.deallocate (outNeighbors, outDeg);
      outNeighbors = nullptr;
    }
  }
};

template <typename T>
struct DAGcontext: public DAGcontextBase<T, DAGcontext<T>, DAGnhoodItem<DAGcontext<T> > > {

  typedef DAGcontextBase<T, DAGcontext, DAGnhoodItem<DAGcontext> > Base;
  typedef typename Base::NItem NItem;
  typedef typename Base::NhoodMgr NhoodMgr;

  explicit DAGcontext (const T& t, typename Base::NhoodMgr& nhmgr)
    : Base (t, nhmgr) 
  {}

  GALOIS_ATTRIBUTE_PROF_NOINLINE
  virtual void subAcquire (Lockable* l, Galois::MethodFlag) {
    NItem& nitem = Base::nhmgr.getNhoodItem (l);

    assert (NItem::getOwner (l) == &nitem);

    nitem.addSharer (this);
    
  }
};


template <typename T>
struct DAGcontextRW: public DAGcontextBase<T, DAGcontextRW<T>, DAGnhoodItemRW<DAGcontextRW<T> > > {

  typedef DAGcontextBase<T, DAGcontextRW, DAGnhoodItemRW<DAGcontextRW> > Base;
  typedef typename Base::NItem NItem;
  typedef typename Base::NhoodMgr NhoodMgr;

  explicit DAGcontextRW (const T& t, NhoodMgr& nhmgr)
    : Base (t, nhmgr) 
  {}

  GALOIS_ATTRIBUTE_PROF_NOINLINE
  virtual void subAcquire (Lockable* l, Galois::MethodFlag f) {

    NItem& nitem = Base::nhmgr.getNhoodItem (l);

    assert (NItem::getOwner (l) == &nitem);

    switch (f) {
      case Galois::MethodFlag::READ:
        nitem.addReader (this);
        break;

      case Galois::MethodFlag::WRITE:
        nitem.addWriter (this);
        break;

      default:
        GALOIS_DIE ("can't handle flag type\n");
        break;
    }
  }
};

} // end namespace anonymous


template <typename T, typename Cmp, typename OpFunc, typename NhoodFunc, typename Ctxt_tp>
class DAGexecutorBase {

protected:
  typedef Ctxt_tp  Ctxt;
  typedef typename Ctxt::NhoodMgr NhoodMgr;
  typedef typename Ctxt::NItem NItem;

  typedef MM::FixedSizeAllocator<Ctxt> CtxtAlloc;
  typedef MM::PowOf_2_BlockAllocator<Ctxt*> CtxtAdjAlloc;
  typedef PerThreadBag<Ctxt*> CtxtWL;
  typedef UserContextAccess<T> UserCtx;
  typedef PerThreadStorage<UserCtx> PerThreadUserCtx;


  struct ApplyOperator {

    typedef int tt_does_not_need_aborts;

    DAGexecutorBase& outer;

    template <typename W>
    void operator () (Ctxt* src, W& wl) {
      assert (src->isSrc ());

      // printf ("processing source: %p, item: %d\n", src, src->elem);

      UserCtx& uctx = *(outer.userCtxts.getLocal ());
      outer.opFunc (src->getElem (), uctx);

      for (auto i = src->neighbor_begin (), i_end = src->neighbor_end ();
          i != i_end; ++i) {

        if ((*i)->removeLastInNeigh (src)) {
          wl.push (*i);
          outer.numPush += 1;
        }
      }
    }
  };


  static const unsigned DEFAULT_CHUNK_SIZE = 16;

  Cmp cmp;
  NhoodFunc nhVisitor;
  OpFunc opFunc;
  std::string loopname;
  NhoodMgr nhmgr;

  CtxtAlloc ctxtAlloc;
  CtxtAdjAlloc ctxtAdjAlloc;
  CtxtWL allCtxts;
  CtxtWL initSources;
  PerThreadUserCtx userCtxts;
  Galois::GAccumulator<size_t> numPush;

public:

  DAGexecutorBase (
      const Cmp& cmp, 
      const NhoodFunc& nhVisitor, 
      const OpFunc& opFunc,
      const char* loopname)
    :
      cmp (cmp),
      nhVisitor (nhVisitor),
      opFunc (opFunc),
      loopname (loopname),
      nhmgr (typename NItem::Factory ())
  {}

  ~DAGexecutorBase (void) {
    Galois::do_all_choice (Galois::Runtime::makeLocalRange (allCtxts),
        [this] (Ctxt* ctxt) {
          ctxt->reclaimAlloc (ctxtAdjAlloc);
          ctxtAlloc.destroy (ctxt);
          ctxtAlloc.deallocate (ctxt, 1);
        }, 
       "free_ctx", Galois::doall_chunk_size<DEFAULT_CHUNK_SIZE> ());
  }

  void createEdge (Ctxt* a, Ctxt* b) {
    assert (a != nullptr);
    assert (b != nullptr);

    if (a == b) { 
      // no self edges
      return; 
    }

    // a < b ? a : b
    Ctxt* src = cmp (a->getElem () , b->getElem ()) ? a : b;
    Ctxt* dst = (src == a) ? b : a;

    // avoid adding same edge multiple times
    if (src->addOutNeigh (dst)) {
      dst->addInNeigh (src);
    }
  }

  virtual void createAllEdges (void) = 0;

  template <typename R>
  void initialize (const R& range) {
    // 
    // 1. create contexts and expand neighborhoods and create graph nodes
    // 2. go over nhood items and create edges
    // 3. Find initial sources and run for_each
    //

    Galois::StatTimer t_init ("Time to create the DAG: ");

    std::printf ("total number of tasks: %ld\n", std::distance (range.begin (), range.end ()));

    t_init.start ();
    Galois::do_all_choice (range,
        [this] (const T& x) {
          Ctxt* ctxt = ctxtAlloc.allocate (1);
          assert (ctxt != NULL);
          ctxtAlloc.construct (ctxt, x, nhmgr);

          allCtxts.get ().push_back (ctxt);

          Galois::Runtime::setThreadContext (ctxt);

          UserCtx& uctx = *(userCtxts.getLocal ());
          nhVisitor (ctxt->getElem (), uctx);
          Galois::Runtime::setThreadContext (NULL);

          // printf ("Created context:%p for item: %d\n", ctxt, x);

        }, "create_ctxt", Galois::doall_chunk_size<DEFAULT_CHUNK_SIZE> ());

    // virtual call implemented by derived classes
    createAllEdges ();


    Galois::do_all_choice (Galois::Runtime::makeLocalRange (allCtxts),
        [this] (Ctxt* ctxt) {
          ctxt->finalizeAdj (ctxtAdjAlloc);
          // std::printf ("ctxt: %p, indegree=%d\n", ctxt, ctxt->origInDeg);
          if (ctxt->isSrc ()) {
            initSources.get ().push_back (ctxt);
          }
        }, "finalize", Galois::doall_chunk_size<DEFAULT_CHUNK_SIZE> ());

    std::printf ("Number of initial sources: %ld\n", initSources.size_all ());

    // printStats ();

    t_init.stop ();
  }

  void execute (void) {

    StatTimer t_exec ("Time to execute the DAG: ");

    typedef Galois::WorkList::dChunkedFIFO<OpFunc::CHUNK_SIZE, Ctxt*> SrcWL_ty;

    t_exec.start ();

    Galois::for_each_local (initSources,
        ApplyOperator {*this}, Galois::loopname("apply_operator"), Galois::wl<SrcWL_ty>());

    // std::printf ("Number of pushes: %zd\n, (#pushes + #init) = %zd\n", 
        // numPush.reduceRO (), numPush.reduceRO () + initSources.size_all  ());
    t_exec.stop ();
  }

  void reinitDAG (void) {
    Galois::StatTimer t_reset ("Time to reset the DAG: ");

    t_reset.start ();
    Galois::do_all_choice (Galois::Runtime::makeLocalRange (allCtxts),
        [] (Ctxt* ctxt) {
          ctxt->reset();
        },
        "reinitDAG", Galois::doall_chunk_size<DEFAULT_CHUNK_SIZE> ());
    t_reset.stop ();
  }

  void printStats (void) {
    Galois::GAccumulator<size_t> numEdges;
    Galois::GAccumulator<size_t> numNodes;

    printf ("WARNING: timing affected by measuring DAG stats\n");

    Galois::do_all_choice (Galois::Runtime::makeLocalRange (allCtxts),
        [&numNodes,&numEdges] (Ctxt* ctxt) {
          numNodes += 1;
          numEdges += std::distance (ctxt->neighbor_begin (), ctxt->neighbor_end ());
        },
        "dag_stats", Galois::doall_chunk_size<DEFAULT_CHUNK_SIZE> ());

    printf ("DAG created with %zd nodes, %zd edges\n", 
        numNodes.reduceRO (), numEdges.reduceRO ());
  }

};

template <typename T, typename Cmp, typename OpFunc, typename NhoodFunc> 
struct DAGexecutor: public DAGexecutorBase<T, Cmp, OpFunc, NhoodFunc, DAGcontext<T> > {

  typedef DAGexecutorBase<T, Cmp, OpFunc, NhoodFunc, DAGcontext<T> > Base;
  typedef typename Base::NItem NItem;

  DAGexecutor(
      const Cmp& cmp, 
      const NhoodFunc& nhoodVisitor, 
      const OpFunc& opFunc, 
      const char* loopname)
    : 
      Base (cmp, nhoodVisitor, opFunc, loopname) 
  {}

  virtual void createAllEdges (void) {
    Galois::do_all_choice (Base::nhmgr.getAllRange(),
        [this] (NItem* nitem) {
          // std::printf ("Nitem: %p, num sharers: %ld\n", nitem, nitem->sharers.size ());

          for (auto i = nitem->sharers.begin ()
            , i_end = nitem->sharers.end (); i != i_end; ++i) {

            auto j = i;
            ++j;
            for (; j != i_end; ++j) {
              Base::createEdge (*i, *j);
            }
          }
        }, "create_ctxt_edges", Galois::doall_chunk_size<Base::DEFAULT_CHUNK_SIZE> ());
  }
};

template <typename T, typename Cmp, typename OpFunc, typename NhoodFunc> 
struct DAGexecutorRW: public DAGexecutorBase<T, Cmp, OpFunc, NhoodFunc, DAGcontextRW<T> > {

  typedef DAGexecutorBase<T, Cmp, OpFunc, NhoodFunc, DAGcontextRW<T> > Base;
  typedef typename Base::NItem NItem;

  DAGexecutorRW (
      const Cmp& cmp, 
      const NhoodFunc& nhoodVisitor, 
      const OpFunc& opFunc, 
      const char* loopname)
    : 
      Base (cmp, nhoodVisitor, opFunc, loopname) 
  {}

  virtual void createAllEdges (void) {

    Galois::do_all_choice (Base::nhmgr.getAllRange(),
        [this] (NItem* nitem) {
          // std::printf ("Nitem: %p, num sharers: %ld\n", nitem, nitem->sharers.size ());

          for (auto w = nitem->writers.begin ()
            , w_end = nitem->writers.end (); w != w_end; ++w) {

            auto v = w;
            ++v;
            for (; v != w_end; ++v) {
              Base::createEdge (*w, *v);
            }

            for (auto r = nitem->readers.begin ()
              , r_end = nitem->readers.end (); r != r_end; ++r) {
              Base::createEdge (*w, *r);
            }
          }
        }, "create_ctxt_edges", Galois::doall_chunk_size<Base::DEFAULT_CHUNK_SIZE> ());
  }

};

template <typename R, typename Cmp, typename OpFunc, typename NhoodFunc>
DAGexecutorRW<typename R::value_type, Cmp, OpFunc, NhoodFunc>* make_dag_executor (const R& range, const Cmp& cmp, const NhoodFunc& nhVisitor, const OpFunc& opFunc, const char* loopname=nullptr) {

  return new DAGexecutorRW<typename R::value_type, Cmp, OpFunc, NhoodFunc> (cmp, nhVisitor, opFunc, loopname);
}

template <typename R, typename Cmp, typename OpFunc, typename NhoodFunc>
void destroy_dag_executor (DAGexecutorRW<typename R::value_type, Cmp, OpFunc, NhoodFunc>*& exec_ptr) {
  delete exec_ptr; exec_ptr = nullptr;
}

template <typename R, typename Cmp, typename OpFunc, typename NhoodFunc>
void for_each_ordered_dag (const R& range, const Cmp& cmp, const NhoodFunc& nhVisitor, const OpFunc& opFunc, const char* loopname=nullptr) {

  typedef typename R::value_type T;
  typedef DAGexecutorRW<T, Cmp, OpFunc, NhoodFunc> Exec_ty;
  // typedef DAGexecutor<T, Cmp, OpFunc, NhoodFunc> Exec_ty;
  
  Exec_ty exec (cmp, nhVisitor, opFunc);

  exec.initialize (range);

  exec.execute ();

}



} // end namespace Runtime
} // end namespace Galois


#endif // GALOIS_RUNTIME_DAG_H
