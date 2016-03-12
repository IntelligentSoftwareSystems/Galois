/** ?? -*- C++ -*-
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
 */

#ifndef GALOIS_RUNTIME_ORDERED_SPECULATION_H
#define GALOIS_RUNTIME_ORDERED_SPECULATION_H

#include "Galois/PerThreadContainer.h"
#include "Galois/PriorityQueue.h"
#include "Galois/DoAllWrap.h"
#include "Galois/Atomic.h"
#include "Galois/Accumulator.h"
#include "Galois/GaloisForwardDecl.h"
#include "Galois/optional.h"

#include "Galois/Runtime/Context.h"
#include "Galois/Runtime/OrderedLockable.h"
#include "Galois/Runtime/IKDGbase.h"
#include "Galois/Runtime/WindowWorkList.h"
#include "Galois/Runtime/UserContextAccess.h"
#include "Galois/Runtime/Mem.h"

#include "Galois/Substrate/gio.h"
#include "Galois/Substrate/PerThreadStorage.h"
#include "Galois/Substrate/CompilerSpecific.h"

namespace Galois {

namespace Runtime {

enum class ContextState: int {
    UNSCHEDULED = 0,
    SCHEDULED,
    READY_TO_COMMIT,
    ABORT_SELF,
    ABORT_HELP,
    COMMITTING, 
    COMMIT_DONE,
    READY_TO_ABORT,
    ABORTING,
    ABORT_DONE,
    ABORTED_CHILD,
    RECLAIM,
};

const char* ContextStateNames[] = {
    "UNSCHEDULED",
    "SCHEDULED",
    "READY_TO_COMMIT",
    "ABORT_SELF",
    "ABORT_HELP",
    "COMMITTING", 
    "COMMIT_DONE",
    "READY_TO_ABORT",
    "ABORTING",
    "ABORT_DONE",
    "ABORTED_CHILD",
    "RECLAIM",
};


template <typename Ctxt, typename CtxtCmp>
struct OptimNhoodItem: public OrdLocBase<OptimNhoodItem<Ctxt, CtxtCmp>, Ctxt, CtxtCmp> {

  using Base = OrdLocBase<OptimNhoodItem, Ctxt, CtxtCmp>;
  using Factory = OrdLocFactoryBase<OptimNhoodItem, Ctxt, CtxtCmp>;

  using Sharers = Galois::gstl::List<Ctxt*>;
  using Lock_ty = Galois::Substrate::SimpleLock;


  const CtxtCmp& ctxtCmp;
  GAtomic<Ctxt*> minCtxt;
  Sharers sharers;


  OptimNhoodItem (Lockable* l, const CtxtCmp& ctxtCmp): 
    Base (l), 
    ctxtCmp (ctxtCmp),
    minCtxt (nullptr)
  {}


  bool markMin (Ctxt* ctxt) {
    assert (ctxt);

    Ctxt* other = nullptr;

    do {

      other = minCtxt;

      if (other == ctxt) {
        return true;
      }

      if (other) {

        if (ctxtCmp (other, ctxt)) {

          ctxt->disableSrc ();
          return false;
        }
      }
      
    } while (!minCtxt.cas (other, ctxt));

    if (other) {
      other->disableSrc ();
    }

    return true;
  }

  Ctxt* getMin (void) const {
    return minCtxt;
  }

  void resetMin (Ctxt* c) {

    assert (getMin () == c);
    minCtxt = nullptr;
  }

  void addToHistory (Ctxt* ctxt) {

    assert (ctxt && ctxt->isSrc () && ctxt->hasState (ContextState::READY_TO_COMMIT));
    assert (std::find (sharers.begin (), sharers.end (), ctxt) == sharers.end ());

    if (!sharers.empty ()) {
      assert (sharers.back ()->hasState (ContextState::READY_TO_COMMIT));
    }
    sharers.push_back (ctxt);
  }

  Ctxt* getHistHead (void) const {

    if (sharers.empty ()) {
      return nullptr;

    } else {
      return sharers.front ();
    }
  }

  Ctxt* getHistTail (void) const {
    
    if (sharers.empty ()) {
      return nullptr;

    } else { 
      return sharers.back ();

    }
  }

  template <typename WL>
  bool findAborts (Ctxt* ctxt, WL& abortWL) {

    assert (getMin () == ctxt);

    bool ret = false;

    for (auto i = sharers.end (), beg_i = sharers.begin (); beg_i != i; ) {
      --i;
      if (ctxtCmp (ctxt, *i)) {
        dbg::print (ctxt, " causing sharer to abort ", *i);
        ret = true;
        (*i)->markForAbortRecursive (abortWL);

      } else {
        break;
      }
    }

    return ret;
  }

  //! mark all sharers later than ctxt for abort
  template <typename WL>
  void markForAbort (Ctxt* ctxt, WL& abortWL) {

    assert (std::find (sharers.begin (), sharers.end (), ctxt) != sharers.end ());

    bool succ = false;

    for (auto i = sharers.end (), beg_i = sharers.begin (); beg_i != i; ) {
      --i;
      if (ctxt == *i) {
        succ = true;
        break;

      } else {
        dbg::print (ctxt, " causing sharer to abort ", *i);
        (*i)->markForAbortRecursive (abortWL);
      }
    }

    assert (succ);

  }

  // TODO: re-implement
  void removeAbort (Ctxt* ctxt) {

    assert (!sharers.empty ());
    assert (std::find (sharers.begin (), sharers.end (), ctxt) != sharers.end ());

    assert (ctxt->hasState (ContextState::ABORTING));

    if (sharers.back () != ctxt) { 
      GALOIS_DIE ("invalid state");
    }

    assert (sharers.back () == ctxt);
    sharers.pop_back ();

    assert (std::find (sharers.begin (), sharers.end (), ctxt) == sharers.end ());

  }

  void removeCommit (Ctxt* ctxt) {


    assert (!sharers.empty ());
    assert (std::find (sharers.begin (), sharers.end (), ctxt) != sharers.end ());
    assert (sharers.front () == ctxt);

    sharers.pop_front ();

    assert (std::find (sharers.begin (), sharers.end (), ctxt) == sharers.end ());

  }

};


template <typename T, typename Cmp, typename Exec>
struct OptimContext: public OrderedContextBase<T> {

  using Base = OrderedContextBase<T>;

  using CtxtCmp = ContextComparator<OptimContext, Cmp>;
  using NItem = OptimNhoodItem<OptimContext, CtxtCmp>;
  using NhoodMgr = PtrBasedNhoodMgr<NItem>;
  using NhoodList = typename gstl::Vector<NItem*>;
  using ChildList = typename gstl::Vector<OptimContext*>;
  using Lock_ty = Galois::Substrate::SimpleLock;

  bool source;
  std::atomic<ContextState> state;
  Exec& exec;
  bool addBack; // set to false by parent when parent is marked for abort, see markAbortRecursive
  Galois::GAtomic<bool> onWL;
  unsigned execRound;
  NhoodList nhood;

  // TODO: avoid using UserContextAccess and per-iteration allocator
  // use Pow of 2 block allocator instead. 
  UserContextAccess<T> userHandle;
  ChildList children;


  explicit OptimContext (const T& x, const ContextState& s, Exec& exec)
  :
    Base (x), 
    source (true), 
    state (s), 
    exec (exec),
    addBack (false),
    onWL (false),
    execRound (0)
  {}


  bool hasState (const ContextState& s) const { return state == s; } 

  void setState (const ContextState& s) { state = s; } 

  bool casState (ContextState s_old, const ContextState& s_new) { 
    // return state.cas (s_old, s_new);
    return state.compare_exchange_strong (s_old, s_new);
  }

  void markExecRound (unsigned r) {
    assert (r >= execRound);
    execRound = r;
  }

  unsigned getExecRound (void) const {
    return execRound;
  }

  ContextState getState (void) const { return state; }

  // bool isRunning (void) const {
    // return hasState (ContextState::SCHEDULED) || hasState (ContextState::ABORT_SELF);
  // }
// 
  // void waitFor (OptimContext* that) const {
    // assert (that);
// 
    // while (that->isRunning ()) {
      // Substrate::asmPause ();
    // }
  // }


  void disableSrc (void) {
    source = false;
  }

  bool isSrc (void) const {
    return source; 
  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE
  virtual void subAcquire (Lockable* l, Galois::MethodFlag m) {

    NItem& nitem = exec.nhmgr.getNhoodItem (l);
    assert (NItem::getOwner (l) == &nitem);

    if (std::find (nhood.begin (), nhood.end (), &nitem) == nhood.end ()) {
      nhood.push_back (&nitem);
      nitem.markMin (this);
    }
  }

  void schedule (void) {
    setState (ContextState::SCHEDULED); 
    onWL = false;
    source = true;
    addBack = true;
    nhood.clear ();
    userHandle.reset ();
    children.clear ();
  }

  void publishChanges (void) {


    // for (auto i = userHandle.getPushBuffer ().begin (), 
        // end_i = userHandle.getPushBuffer ().end (); i != end_i; ++i) {
// 
      // OptimContext* child = exec.push (*i);
      // dbg::print (this, " creating child ", child);
// 
      // children.push_back (child);
    // }
  }

  void addChild (OptimContext* child) {

    assert (std::find (children.begin (), children.end (), child) == children.end ());

    dbg::print (this, " creating child ", child);

    children.push_back (child);

  }

  void doCommit () {

    assert (hasState (ContextState::COMMITTING));

    dbg::print (this, " committing with item ", this->getActive ());

    userHandle.commit (); // TODO: rename to retire

    for (NItem* n: nhood) {
      n->removeCommit (this);
    }

    setState (ContextState::COMMIT_DONE);
  }

  void doAbort () {
    // this can be in states READY_TO_COMMIT, ABORT_SELF
    // children can be in UNSCHEDULED, READY_TO_COMMIT, ABORT_DONE

    // first abort all the children recursively
    // then abort self.
    //

    assert (hasState (ContextState::ABORTING));

    dbg::print (this, " aborting with item ", this->getActive ());

    userHandle.rollback ();

    for (NItem* ni: nhood) {
      ni->removeAbort (this);
    }

    if (this->addBack) {

      setState (ContextState::ABORT_DONE);
      exec.push_abort (this);

    } else {
      // is an aborted child whose parent also aborted
      setState (ContextState::ABORTED_CHILD);
    }

  }

  bool isCommitSrc (void) const {

    for (const NItem* ni: nhood) {

      if (ni->getHistHead () != this) {
        return false;
      }
    }

    return true;
  }

  template <typename WL>
  void findCommitSrc (const OptimContext* gvt, WL& wl) const {

    for (const NItem* ni: nhood) {

      OptimContext* c = ni->getHistHead ();
      assert (c != this);

      if (c && (!gvt || exec.ctxtCmp (c, gvt)) 
          && c->isCommitSrc () 
          && c->onWL.cas (false, true)) {
        wl.push (c);
      }
    }
  }

  bool isAbortSrc (void) const {

    if (!this->hasState (ContextState::READY_TO_ABORT)) {
      return false;
    }

    for (const NItem* ni: nhood) {

      if (ni->getHistTail () != this) {
        return false;
      }
    }

    return true;
  }

  template <typename WL>
  void findAbortSrc (WL& wl) const {

    // XXX: if a task has children that don't share neighborhood with
    // it, should it be an abort source? Yes, because the end goal in 
    // finding abort sources is that tasks may abort and restore state
    // in isolation. 
    
    for (const NItem* ni: nhood) {

      OptimContext* c = ni->getHistTail ();

      if (c && c->isAbortSrc () && c->onWL.cas (false, true)) {
        wl.push (c);
      }
    }
  }

  bool isSrcSlowCheck (void) const {
    
    for (const NItem* ni: nhood) {

      if (ni->getMin () != this) {
        return false;
      }
    }

    return true;
  }

  template <typename WL>
  bool findAborts (WL& abortWL) {

    assert (isSrcSlowCheck ());

    bool ret = false;

    for (NItem* ni: nhood) {
      ret = ni->findAborts (this, abortWL) || ret;
    }

    return ret;
  }


  template <typename WL>
  void markForAbortRecursive (WL& abortWL) {
    if (casState (ContextState::READY_TO_COMMIT, ContextState::READY_TO_ABORT)) {

      for (NItem* ni: nhood) {
        ni->markForAbort (this, abortWL);
      }

      if (isAbortSrc () && onWL.cas (false, true)) {
        abortWL.push (this);
      }

      for (OptimContext* c: children) {
        dbg::print (this, " causing abort on child ", c);
        c->markForAbortRecursive (abortWL);
        c->addBack = false;
      }

    } else if (casState (ContextState::SCHEDULED, ContextState::ABORTED_CHILD)) {
      // a SCHEDULED task can only be aborted recursively if it's a child

    } else if (casState (ContextState::UNSCHEDULED, ContextState::ABORTED_CHILD)) {

    } else {
      assert (hasState (ContextState::READY_TO_ABORT) || hasState (ContextState::ABORTED_CHILD));
    }

    assert (hasState (ContextState::READY_TO_ABORT) || hasState (ContextState::ABORTED_CHILD));

  }

  void resetMarks (void) {

    for (NItem* ni: nhood) {
      if (ni->getMin () == this) {
        ni->resetMin (this);
      }
    }
  }

  void addToHistory (void) {

    for (NItem* ni: nhood) {
      ni->addToHistory (this);
    }
  }

};

template <typename T, typename Cmp, typename NhFunc, typename ExFunc, typename  OpFunc, typename ArgsTuple>
class OptimOrdExecutor: public IKDGbase<T, Cmp, 
  OptimContext<T, Cmp, OptimOrdExecutor<T, Cmp, NhFunc, ExFunc, OpFunc, ArgsTuple> >,
  NhFunc, ExFunc, OpFunc, ArgsTuple> {

public:

  friend struct OptimContext<T, Cmp, OptimOrdExecutor>;
  using Ctxt = OptimContext<T, Cmp, OptimOrdExecutor>;
  using Base = IKDGbase <T, Cmp, Ctxt, NhFunc, ExFunc, OpFunc, ArgsTuple>;

  using NhoodMgr = typename Ctxt::NhoodMgr;
  using CtxtCmp = typename Ctxt::CtxtCmp;
  using NItemFactory = typename Ctxt::NItem::Factory;
  using CtxtWL = typename Base::CtxtWL;

  using WindowWL = typename std::conditional<Base::NEEDS_PUSH, PQbasedWindowWL<Ctxt*, CtxtCmp>, SortedRangeWindowWL<Ctxt*, CtxtCmp> >::type;

  using CommitQ = Galois::PerThreadVector<Ctxt*>;
  using ExecutionRecords = std::vector<ParaMeter::StepStats>;

  static const unsigned DEFAULT_CHUNK_SIZE = 4;

  struct CtxtMaker {
    OptimOrdExecutor& outer;

    Ctxt* operator () (const T& x) {

      Ctxt* ctxt = outer.ctxtAlloc.allocate (1);
      assert (ctxt);
      outer.ctxtAlloc.construct (ctxt, x, ContextState::UNSCHEDULED, outer);

      return ctxt;
    }
  };


  NItemFactory nitemFactory;
  NhoodMgr nhmgr;
  WindowWL winWL;
  CtxtMaker ctxtMaker;


  GAccumulator<size_t> totalRetires;
  CommitQ commitQ;
  ExecutionRecords execRcrds;

  TimeAccumulator t_expandNhood;
  TimeAccumulator t_beginRound;
  TimeAccumulator t_executeSources;
  TimeAccumulator t_applyOperator;
  TimeAccumulator t_computeGVT;
  TimeAccumulator t_serviceAborts;
  TimeAccumulator t_performCommits;
  TimeAccumulator t_reclaimMemory;

public:
  OptimOrdExecutor (const Cmp& cmp, const NhFunc& nhFunc, const ExFunc& exFunc, const OpFunc& opFunc, const ArgsTuple& argsTuple)
    : 
      Base (cmp, nhFunc, exFunc, opFunc, argsTuple),
      nitemFactory (Base::ctxtCmp),
      nhmgr (nitemFactory),
      winWL (Base::ctxtCmp),
      ctxtMaker {*this}
  {
  }

  ~OptimOrdExecutor (void) {
    dumpStats ();
  }

  // on each thread
  template <typename R>
  void push_initial (const R& range) {

    StatTimer t ("push_initial");

    t.start ();

    Galois::do_all_choice (range,
        [this] (const T& x) {

          Ctxt* c = ctxtMaker (x);
          Base::getNextWL ().push (c);

        }, 
        std::make_tuple (
          Galois::loopname ("init-fill"),
          chunk_size<DEFAULT_CHUNK_SIZE> ()));

    if (Base::targetCommitRatio != 0.0) {

      winWL.initfill (makeLocalRange (Base::getNextWL ()));
      Base::getNextWL ().clear_all_parallel ();
    }

    t.stop ();

  }

  void operator () (void) {
    execute ();
  }
  
  void execute () {

    StatTimer t ("executorLoop");

    typename Base::CtxtWL sources;

    t.start ();

    while (true) {

      beginRound ();

      if (Base::getCurrWL ().empty_all ()) {
        break;
      }

      expandNhood ();

      serviceAborts (sources);

      executeSources (sources);

      applyOperator (sources);

      performCommits ();

      reclaimMemory (sources);

      Base::endRound ();

    }

    t.stop ();
  }

private:

  void dumpParaMeterStats (void) {

    for (const ParaMeter::StepStats& s: execRcrds) {
      s.dump (ParaMeter::getStatsFile (), Base::loopname);
    }

    ParaMeter::closeStatsFile ();
  }

  void dumpStats (void) {

    reportStat (Base::loopname, "retired", totalRetires.reduce ());
    reportStat (Base::loopname, "efficiency", double (totalRetires.reduce ()) / Base::totalTasks);
    reportStat (Base::loopname, "avg. parallelism", double (totalRetires.reduce ()) / Base::rounds);

    if (Base::ENABLE_PARAMETER) {
      dumpParaMeterStats ();
    }

    reportStat ("NULL", "t_expandNhood",    t_expandNhood.get ());
    reportStat ("NULL", "t_beginRound",     t_beginRound.get ());
    reportStat ("NULL", "t_executeSources", t_executeSources.get ());
    reportStat ("NULL", "t_applyOperator",  t_applyOperator.get ());
    reportStat ("NULL", "t_computeGVT",     t_computeGVT.get ());
    reportStat ("NULL", "t_serviceAborts",  t_serviceAborts.get ());
    reportStat ("NULL", "t_performCommits", t_performCommits.get ());
    reportStat ("NULL", "t_reclaimMemory",  t_reclaimMemory.get ());

  }

  Ctxt* push (const T& x) {
    
    Ctxt* ctxt = ctxtMaker (x);

    if (Base::targetCommitRatio == 0.0) {

      Base::getNextWL ().push (ctxt);

    } else {
      winWL.push (ctxt);
          
    }

    return ctxt;
  }

  void push_abort (Ctxt* ctxt) {
    assert (ctxt);
    assert (ctxt->hasState (ContextState::ABORT_DONE));

    ctxt->setState (ContextState::UNSCHEDULED);
    Base::getNextWL ().push (ctxt);
  }


  GALOIS_ATTRIBUTE_PROF_NOINLINE void beginRound () {
    
    t_beginRound.start ();
    Base::beginRound (winWL);

    if (Base::ENABLE_PARAMETER) {
      execRcrds.emplace_back (Base::rounds, Base::getCurrWL ().size_all ());
    }

    t_beginRound.stop ();
  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE void expandNhood (void) {

    t_expandNhood.start ();
    Galois::do_all_choice (makeLocalRange (Base::getCurrWL ()),
        [this] (Ctxt* c) {

          if (!c->hasState (ContextState::ABORTED_CHILD)) {

            assert (!c->hasState (ContextState::RECLAIM));
            c->schedule ();

            dbg::print ("scheduling: ", c, " with item: ", c->getActive ());

            typename Base::UserCtxt& uhand = c->userHandle;

            // nhFunc (c, uhand);
            runCatching (Base::nhFunc, c, uhand);

            Base::roundTasks += 1;
          }
        },
        std::make_tuple (
          Galois::loopname ("expandNhood"),
          chunk_size<NhFunc::CHUNK_SIZE> ()));

    t_expandNhood.stop ();

  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE void executeSources (CtxtWL& sources) {

    if (Base::HAS_EXEC_FUNC) {


      t_executeSources.start ();

      Galois::do_all_choice (makeLocalRange (sources),
        [this] (Ctxt* ctxt) {
          assert (ctxt->isSrc ());
          assert (!ctxt->hasState (ContextState::RECLAIM));
          assert (!ctxt->hasState (ContextState::ABORTED_CHILD));

          Base::exFunc (ctxt->getActive (), ctxt->userHandle);
        },
        std::make_tuple (
          Galois::loopname ("executeSources"),
          Galois::chunk_size<ExFunc::CHUNK_SIZE> ()));

      t_executeSources.stop ();
    }
  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE void applyOperator (CtxtWL& sources) {
    t_applyOperator.start ();

    Ctxt* minElem = nullptr;

    if (Base::NEEDS_PUSH) {
      if (Base::targetCommitRatio != 0.0 && !winWL.empty ()) {
        minElem = *winWL.getMin();
      }
    }



    Galois::do_all_choice (makeLocalRange (sources),
        [this, minElem] (Ctxt* c) {

          typename Base::UserCtxt& uhand = c->userHandle;

          assert (c->isSrc ());
          assert (!c->hasState (ContextState::RECLAIM));
          assert (!c->hasState (ContextState::ABORTED_CHILD));

          bool commit = true;

          if (Base::OPERATOR_CAN_ABORT) {
            runCatching (Base::opFunc, c, uhand);
            commit = c->isSrc (); // in case opFunc signalled abort

          } else {
            Base::opFunc (c->getActive (), uhand);
            commit = true;
          }

          if (commit) {

            if (Base::NEEDS_PUSH) {

              for (auto i = uhand.getPushBuffer ().begin ()
                  , endi = uhand.getPushBuffer ().end (); i != endi; ++i) {

                Ctxt* child = ctxtMaker (*i);
                c->addChild (child);

                if (!minElem || !Base::ctxtCmp (minElem, child)) {
                  // if *i >= *minElem
                  Base::getNextWL ().push_back (child);
                } else {
                  winWL.push (child);
                } 
              }
            } else {

              assert (uhand.getPushBuffer ().begin () == uhand.getPushBuffer ().end ());
            }

            bool b = c->casState (ContextState::SCHEDULED, ContextState::READY_TO_COMMIT);

            assert (b && "CAS shouldn't have failed");
            Base::roundCommits += 1;

            c->publishChanges ();
            c->addToHistory ();
            commitQ.get ().push_back (c);

            if (Base::ENABLE_PARAMETER) {
              c->markExecRound (Base::rounds);
            }

          } else {

            if (c->casState (ContextState::SCHEDULED, ContextState::ABORTING)) {
              c->doAbort ();

            } else {
              assert (c->hasState (ContextState::ABORTING) || c->hasState (ContextState::ABORT_DONE));
            }
          }
        },
        std::make_tuple (
          Galois::loopname ("applyOperator"),
          Galois::chunk_size<OpFunc::CHUNK_SIZE> ()));

    t_applyOperator.stop ();

  }

  Ctxt* computeGVT (void) {

    t_computeGVT.start ();

    Substrate::PerThreadStorage<Ctxt*> perThrdMin;

    on_each_impl ([this, &perThrdMin] (const unsigned tid, const unsigned numT) {
          
          for (auto i = Base::getNextWL ().local_begin ()
            , end_i = Base::getNextWL ().local_end (); i != end_i; ++i) {

            Ctxt*& lm = *(perThrdMin.getLocal ());

            if (!lm || Base::ctxtCmp (*i, lm)) {
              lm = *i;
            }
          }

          
        });

    Ctxt* ret = nullptr;

    for (unsigned i = 0; i < perThrdMin.size (); ++i) {

      Ctxt* lm = *(perThrdMin.getRemote (i));

      if (lm) {
        if (!ret || Base::ctxtCmp (lm, ret)) {
          ret = lm;
        }
      }
    }

    Ctxt* const* minElem = winWL.getMin ();

    if (minElem) {
      if (!ret || Base::ctxtCmp (*minElem, ret)) {
        ret = *minElem;
      }
    }

    if (ret) {
      dbg::print ("GVT computed as: ", ret, ", with elem: ", ret->getActive ());

    } else {
      dbg::print ("GVT computed as NULL");
    }

    t_computeGVT.stop ();

    return ret;

  }

  void quickAbort (Ctxt* c) {
    assert (c);
    bool b= c->hasState (ContextState::SCHEDULED) || c->hasState (ContextState::ABORTED_CHILD) || c->hasState (ContextState::ABORT_DONE);

    assert (b);

    if (c->casState (ContextState::SCHEDULED, ContextState::ABORT_DONE)) {
      push_abort (c);

    } else {
      assert (c->hasState (ContextState::ABORTED_CHILD)); 
    }
  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE void serviceAborts (CtxtWL& sources) {

    t_serviceAborts.start ();
    
    CtxtWL abortWL;

    Galois::do_all_choice (makeLocalRange (Base::getCurrWL ()),
        [this, &abortWL] (Ctxt* c) {

          if (c->isSrc ()) {

            if (c->findAborts (abortWL)) {
              // XXX: c does not need to abort if it's neighborhood
              // isn't dependent on values computed by other tasks

              c->disableSrc ();
            }

          } 
        },
        std::make_tuple (
          Galois::loopname ("mark-aborts"),
          Galois::chunk_size<DEFAULT_CHUNK_SIZE> ()));


    Galois::Runtime::for_each_gen (
        makeLocalRange (abortWL),
        [this] (Ctxt* c, UserContext<Ctxt*>& wlHandle) {

          if (c->casState (ContextState::READY_TO_ABORT, ContextState::ABORTING)) {
            c->doAbort ();
            c->findAbortSrc (wlHandle);
          
          } else {
            assert (c->hasState (ContextState::ABORTING) || c->hasState (ContextState::ABORT_DONE));
          }
        },
        std::make_tuple (
          Galois::loopname ("handle-aborts"),
          Galois::does_not_need_aborts_tag (),
          Galois::wl<Galois::WorkList::dChunkedFIFO<NhFunc::CHUNK_SIZE> > ()));
    


    Galois::do_all_choice (makeLocalRange (Base::getCurrWL ()),

        [this, &sources] (Ctxt* c) {
          if (c->isSrc () && !c->hasState (ContextState::ABORTED_CHILD)) {
            assert (c->hasState (ContextState::SCHEDULED));

            sources.push (c);

          } else if (c->hasState (ContextState::ABORTED_CHILD)) {
            commitQ.get ().push_back (c); // for reclaiming memory 

          } else {
            assert (!c->hasState (ContextState::ABORTED_CHILD));
            quickAbort (c);
          }

          c->resetMarks ();
        },
        std::make_tuple ( 
          Galois::loopname ("collect-sources"),
          Galois::chunk_size<DEFAULT_CHUNK_SIZE> ()));

    t_serviceAborts.stop ();

  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE void performCommits () {

    t_performCommits.start ();

    CtxtWL commitSources;

    Ctxt* gvt = computeGVT ();


    Galois::do_all_choice (makeLocalRange (commitQ),
        [this, gvt, &commitSources] (Ctxt* c) {

          assert (c);

          if (c->hasState (ContextState::READY_TO_COMMIT) 
              && (!gvt || Base::ctxtCmp (c, gvt))
              && c->isCommitSrc ()) {

            commitSources.push (c);
          }
        },
        std::make_tuple (
          Galois::loopname ("find-commit-srcs"),
          Galois::chunk_size<DEFAULT_CHUNK_SIZE> ()));
        

    Galois::Runtime::for_each_gen (
        makeLocalRange (commitSources),
        [this, gvt] (Ctxt* c, UserContext<Ctxt*>& wlHandle) {

          bool b = c->casState (ContextState::READY_TO_COMMIT, ContextState::COMMITTING);

          if (b) {

            assert (c->isCommitSrc ());
            c->doCommit ();
            c->findCommitSrc (gvt, wlHandle);
            totalRetires += 1;

            if (Base::ENABLE_PARAMETER) {
              assert (c->getExecRound () < execRcrds.size ());
              execRcrds[c->getExecRound ()].parallelism += 1;
            }

          } else {
            assert (c->hasState (ContextState::COMMIT_DONE));
          }
        },
        std::make_tuple (
          Galois::loopname ("retire"),
          Galois::does_not_need_aborts_tag (),
          Galois::wl<Galois::WorkList::dChunkedFIFO<DEFAULT_CHUNK_SIZE> > ()));

    t_performCommits.stop ();


  }

  void freeCtxt (Ctxt* ctxt) {
    Base::ctxtAlloc.destroy (ctxt);
    Base::ctxtAlloc.deallocate (ctxt, 1);
  }

  void reclaimMemory (CtxtWL& sources) {

    t_reclaimMemory.start ();

    sources.clear_all_parallel ();

    // XXX: the following memory free relies on the fact that 
    // per-thread fixed allocators are being used. Otherwise, mem-free
    // should be done in a separate loop, after enforcing set semantics
    // among all threads


    Galois::Runtime::on_each_impl (
        [this] (const unsigned tid, const unsigned numT) {
          
          auto& localQ = commitQ.get ();
          auto new_end = std::partition (localQ.begin (), 
            localQ.end (), 
            [] (Ctxt* c) {
              assert (c);
              return c->hasState (ContextState::READY_TO_COMMIT);
            });


          for (auto i = new_end, end_i = localQ.end (); i != end_i; ++i) {

            if ((*i)->casState (ContextState::ABORTED_CHILD, ContextState::RECLAIM)
              || (*i)->casState (ContextState::COMMIT_DONE, ContextState::RECLAIM)) {
              dbg::print ("Ctxt destroyed from commitQ: ", *i);
              freeCtxt (*i);
            }
          }

          localQ.erase (new_end, localQ.end ());
        });

    t_reclaimMemory.stop ();

  }




};


template <typename R, typename Cmp, typename NhFunc, typename ExFunc, typename OpFunc, typename _ArgsTuple>
void for_each_ordered_optim (const R& range, const Cmp& cmp, const NhFunc& nhFunc, const ExFunc& exFunc, const OpFunc& opFunc, const _ArgsTuple& argsTuple) {


  auto argsT = std::tuple_cat (argsTuple, 
      get_default_trait_values (argsTuple,
        std::make_tuple (loopname_tag {}, enable_parameter_tag {}),
        std::make_tuple (default_loopname {}, enable_parameter<false> {})));
  using ArgsT = decltype (argsT);

  using T = typename R::value_type;
  

  using Exec = OptimOrdExecutor<T, Cmp, NhFunc, ExFunc, OpFunc, ArgsT>;
  
  Exec e (cmp, nhFunc, exFunc, opFunc, argsT);

  Substrate::getThreadPool().burnPower (Galois::getActiveThreads ());

  e.push_initial (range);
  e.execute ();

  Substrate::getThreadPool().beKind();
}

template <typename R, typename Cmp, typename NhFunc, typename OpFunc, typename _ArgsTuple>
void for_each_ordered_optim (const R& range, const Cmp& cmp, const NhFunc& nhFunc, const OpFunc& opFunc, const _ArgsTuple& argsTuple) {


  for_each_ordered_optim (range, cmp, nhFunc, HIDDEN::DummyExecFunc (), opFunc, argsTuple);
}



} // end namespace Runtime
} // end namespace Galois


#endif // GALOIS_RUNTIME_ORDERED_SPECULATION_H
