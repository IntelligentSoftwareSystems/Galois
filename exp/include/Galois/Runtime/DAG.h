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
 
#ifndef GALOIS_RUNTIME_DAG_H
#define GALOIS_RUNTIME_DAG_H

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
#include "Galois/Runtime/ll/gio.h"
#include "Galois/Runtime/ll/ThreadRWlock.h"
#include "Galois/Runtime/mm/Mem.h"

#include <boost/iterator/filtering_iterator.hpp>

#include <atomic>

namespace Galois {
namespace Runtime {


template <typename T>
struct DAGcontext {

protected:
  typedef Galois::gdeque<DAGcontext*, 8> AdjList;
  typedef std::atomic<int> ParCounter;

  T item;
  const int origInDeg;
  ParCounter inDeg;
  AdjList outNeighbors;

public:
  explicit DAGcontext (const T& t): item (t) origInDeg (0), inDeg (0) {}

  void addOutNeigh (DAGcontext* that) {
    assert (std::find (outNeighbors.begin (), outNeighbors.end (), that) == outNeighbors.end ());
    outNeighbors.push_back (that);

  }

  void addInNeigh (DAGcontext* that) {
    int* x = const_cast<int*> (&origInDeg);
    ++(*x);
    ++inDeg;
  }

  bool removeInNeigh (DAGcontext* that) {
    assert (inDeg >= 0);
    return ((--inDeg) == 0);
  }

  void reset (void) {
    inDeg = origInDeg;
  }

  typename AdjList::iterator neighbor_begin (void) {
    return outNeighbors.begin ();
  }

  typename AdjList::iterator neighbor_end (void) {
    return outNeighbors.end ();
  }

};


struct DAGnhoodItem: public LockManagerBase {

  typedef LockManagerBase Base;

  Lockable* lockable;

  explicit DAGnhoodItem (Lockable* l): lockable (l) {}

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
  
};


class DAGexecutor {

  typedef Galois::Runtime::MM::FSBGaloisAllocator<Ctxt> CtxtAlloc;
  typedef Galois::Runtime::PerThreadVector<Ctxt*> CtxtWL;
  typedef Galois::Runtime::UserContextAccess<T> UserCtx;


  struct ApplyOperator {

    typedef int tt_does_not_need_aborts;

    OpFunc& opFunc;

    explicit ApplyOperator (OpFunc& opFunc): opFunc (opFunc) {}

    template <typename W>
    void operator () (Ctxt* src, W& wl) {
      assert (src->isSrc);

      opFunc (src->item);

      for (auto i = src->neighbor_begin (), i_end = src->neighbor_end;
          i != i_end; ++i) {

        if ((*i)->decInCount() == 0) {
          wl.push (*i);
        }
      }
    }
  };


  CtxtAlloc ctxtAlloc;
  CtxtWL allCtxts;

  template <typename R>
  void initialize (const R& range) {


    // 
    // 1. create contexts and expand neighborhoods and create graph nodes
    // 2. go over nhood items and create edges
    // 3. Find initial sources and run for_each

    Galois::Runtime::do_all_impl (range,
        [this] (const T& active) {
          Ctxt* ctx = ctxtAlloc.allocate (1);
          assert (ctx != NULL);
          // new (ctx) Ctxt (active, nhmgr);
          //ctxtAlloc.construct (ctx, Ctxt (active, nhmgr));
          ctxtAlloc.construct (ctx, active, nhmgr);

          allCtxts.get ().push_back (ctx);

          Galois::Runtime::setThreadContext (ctx);

          nhoodVisitor (ctx->item);
          Galois::Runtime::setThreadContext (NULL);
        }, "create_ctxt");


    Galois::Runtime::do_all_impl (makeLocalRange (allNItems),
        [&] (NItem* nitem) {
          for (auto i = nitem.sharers.begin (), i_end = nitem.sharers.end (); i != i_end; ++i) {
            auto j = i;
            ++j;
            for (; j != i_end; ++j) {
              createEdge (*i, *j);
            }
          }
        }, "create_ctxt_edges");

  }

  void execute (void) {

    auto isSrcFunc = [&] (const Ctxt* ctx) {
      return ctx->incoming == 0;
    };

    auto beg = boost::make_filtering_iterator (isSrcFunc, allCtxts.begin_all (), allCtxts.end_all ());
    auto end = boost::make_filtering_iterator (isSrcFunc, allCtxts.end_all (), allCtxts.end_all ());

    const unsigned CHUNK_SIZE = OpFunc::CHUNK_SIZE;
    typedef Galois::WorkList::dChunkedFIFO<CHUNK_SIZE, Ctxt*> SrcWL_ty;


    Galois::Runtime::for_each_impl<SrcWL_ty> ( 
        Galois::Runtime::makeStandardRange (beg, end),
        ApplyOperator (opFunc), "apply_operator");

  }

  void resetDAG (void) {
    Galois::Runtime::do_all_impl (makeLocalRange (allCtxts),
        [] (Ctxt* ctx) {
          ctx->reset();
        },
        "reset_dag");
  }

};

} // end namespace Runtime
} // end namespace Galois


#endif // GALOIS_RUNTIME_DAG_H
