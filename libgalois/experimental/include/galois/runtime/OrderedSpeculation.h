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

#include "galois/PerThreadContainer.h"
#include "galois/PriorityQueue.h"
#include "galois/DoAllWrap.h"
#include "galois/Atomic.h"
#include "galois/Accumulator.h"
#include "galois/GaloisForwardDecl.h"
#include "galois/optional.h"

#include "galois/runtime/Context.h"
#include "galois/runtime/OrderedLockable.h"
#include "galois/runtime/IKDGbase.h"
#include "galois/runtime/WindowWorkList.h"
#include "galois/runtime/UserContextAccess.h"
#include "galois/runtime/Executor_ParaMeter.h"
#include "galois/runtime/Mem.h"

#include "galois/gIO.h"
#include "galois/substrate/PerThreadStorage.h"
#include "galois/substrate/CompilerSpecific.h"

#include <iostream>

namespace galois {

namespace runtime {

enum class SpecMode {
  OPTIM, PESSIM
};

namespace cll = llvm::cl;


cll::opt<SpecMode> specMode (
    cll::desc ("Speculation mode"),
    cll::values (
      clEnumVal (SpecMode::OPTIM, "SpecMode::OPTIM"),
      clEnumVal (SpecMode::PESSIM, "SpecMode::PESSIM"),
      clEnumValEnd),
    cll::init (SpecMode::OPTIM));

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

template <typename NItem, typename Ctxt, typename CtxtCmp>
struct OptimNItemFunctions {

  static void addToHistory (NItem* ni, Ctxt* ctxt) {

    assert (ctxt && ctxt->isSrc() && ctxt->hasState (ContextState::READY_TO_COMMIT));
    assert (std::find (ni->histList.begin(), ni->histList.end(), ctxt) == ni->histList.end());

    if (!ni->histList.empty()) {
      assert (ni->histList.back()->hasState (ContextState::READY_TO_COMMIT));
    }
    ni->histList.push_back (ctxt);
  }

  static Ctxt* getHistHead (const NItem* ni) {

    if (ni->histList.empty()) {
      return nullptr;

    } else {
      return ni->histList.front();
    }
  }

  static Ctxt* getHistTail (const NItem* ni) {
    
    if (ni->histList.empty()) {
      return nullptr;

    } else { 
      return ni->histList.back();

    }
  }

  template <typename WL>
  static void findAborts (NItem* ni, Ctxt* ctxt, WL& abortRoots) {

    assert (ni->getMin() == ctxt);

    Ctxt* next = nullptr;

    for (auto i = ni->histList.end(), beg_i = ni->histList.begin(); beg_i != i; ) {
      --i;
      if (ni->ctxtCmp (ctxt, *i)) {
        dbg::print (ctxt, " causing sharer to abort ", *i);
        next = *i;
        // (*i)->markForAbortRecursive (abortRoots);

      } else {
        break;
      }
    }

    if (next) {
      next->markForAbortRecursive (abortRoots);
    }

    // return ret;
  }

  //! mark all histList later than ctxt for abort
  template <typename WL>
  static void markForAbort (NItem* ni, Ctxt* ctxt, WL& abortRoots) {

    assert (std::find (ni->histList.begin(), ni->histList.end(), ctxt) != ni->histList.end());

    bool succ = false;

    Ctxt* next = nullptr;

    for (auto i = ni->histList.end(), beg_i = ni->histList.begin(); beg_i != i; ) {
      --i;
      if (ctxt == *i) {
        succ = true;
        break;

      } else {
        dbg::print (ctxt, " causing sharer to abort ", *i);
        // (*i)->markForAbortRecursive (abortRoots);
        next = *i;
      }
    }

    assert (succ);

    if (next) {
      next->markForAbortRecursive (abortRoots);
    }

  }

  // TODO: re-implement
  static void removeAbort (NItem* ni, Ctxt* ctxt) {

    assert (!ni->histList.empty());
    assert (std::find (ni->histList.begin(), ni->histList.end(), ctxt) != ni->histList.end());

    assert (ctxt->hasState (ContextState::ABORTING));

    if (ni->histList.back() != ctxt) { 
      GALOIS_DIE ("invalid state");
    }

    assert (ni->histList.back() == ctxt);
    ni->histList.pop_back();

    assert (std::find (ni->histList.begin(), ni->histList.end(), ctxt) == ni->histList.end());

  }

  static void removeCommit (NItem* ni, Ctxt* ctxt) {


    assert (!ni->histList.empty());
    assert (std::find (ni->histList.begin(), ni->histList.end(), ctxt) != ni->histList.end());
    assert (ni->histList.front() == ctxt);

    ni->histList.pop_front();

    assert (std::find (ni->histList.begin(), ni->histList.end(), ctxt) == ni->histList.end());

  }

  static void detectAborts (NItem* ni, Ctxt* ctxt) {
    // ctxt is the winner, we check for aborts
    Ctxt* t = ni->getHistTail();
    if (t && ni->ctxtCmp (ctxt, t)) {
      ctxt->addAbortLocation (ni);
    }
  }

};

template <typename Ctxt, typename CtxtCmp>
struct OptimNItem: public OrdLocBase<OptimNItem<Ctxt, CtxtCmp>, Ctxt, CtxtCmp> {

  using Base = OrdLocBase<OptimNItem, Ctxt, CtxtCmp>;

  using Factory = OrdLocFactoryBase<OptimNItem, Ctxt, CtxtCmp>;
  using HistList = galois::gstl::List<Ctxt*>;
  using Lock_ty = galois::substrate::SimpleLock;
  using NF = OptimNItemFunctions<OptimNItem, Ctxt, CtxtCmp>;


  const CtxtCmp& ctxtCmp;
  HistList histList;
  GAtomic<Ctxt*> minCtxt;


  OptimNItem (Lockable* l, const CtxtCmp& ctxtCmp): 
    Base (l),
    ctxtCmp (ctxtCmp),
    histList(),
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

          ctxt->disableSrc();
          return false;
        }
      }
      
    } while (!minCtxt.cas (other, ctxt));

    if (other) {
      other->disableSrc();
    }

    detectAborts (ctxt);

    return true;
  }

  Ctxt* getMin (void) const {
    return minCtxt;
  }

  void resetMin (Ctxt* c) {

    assert (getMin() == c);
    minCtxt = nullptr;
  }

  void addToHistory (Ctxt* ctxt) {
    NF::addToHistory (this, ctxt);
  }

  Ctxt* getHistHead (void) const {
    return NF::getHistHead (this);
  }

  Ctxt* getHistTail (void) const {
    return NF::getHistTail (this);
  }

  template <typename WL>
  void findAborts (Ctxt* ctxt, WL& abortRoots) {
    NF::findAborts (this, ctxt, abortRoots);
  }

  template <typename WL>
  void markForAbort (Ctxt* ctxt, WL& abortRoots) {
    NF::markForAbort (this, ctxt, abortRoots);
  }

  void removeAbort (Ctxt* ctxt) {
    NF::removeAbort (this, ctxt);
  }

  void removeCommit (Ctxt* ctxt) {
    NF::removeCommit (this, ctxt);
  }

  void detectAborts (Ctxt* ctxt) {
    NF::detectAborts (this, ctxt);
  }
};


template <typename T, typename Cmp, typename Exec>
struct SpecContextBase: public OrderedContextBase<T> {

  using Base = OrderedContextBase<T>;
  using CtxtCmp = ContextComparator<SpecContextBase, Cmp>;
  using Executor = Exec;


  bool source;
  std::atomic<ContextState> state;
  Exec& exec;
  unsigned execRound;
  UserContextAccess<T> userHandle;

  explicit SpecContextBase (const T& x, const ContextState& s, Exec& exec)
  :
    Base (x), 
    source (true), 
    state (s), 
    exec (exec),
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

  void disableSrc (void) {
    source = false;
  }

  bool isSrc (void) const {
    return source; 
  }

  void enableSrc (void) {
    source = true;
  }

  void schedule (void) {
    source = true;

    assert (hasState (ContextState::UNSCHEDULED) || hasState (ContextState::ABORT_DONE));
    setState (ContextState::SCHEDULED); 

    userHandle.reset();
  }
};

template <typename Ctxt, typename NItem>
struct OptimContextFunctions {

  static void addChild (Ctxt* c, Ctxt* child) {

    assert (std::find (c->children.begin(), c->children.end(), child) == c->children.end());

    dbg::print (c, " creating child ", child);

    c->children.push_back (child);

  }

  static void doCommit(Ctxt* c) {

    assert (c->hasState (ContextState::COMMITTING));

    dbg::print (c, " committing with item ", c->getActive());

    c->userHandle.commit(); // TODO: rename to retire

    for (NItem* n: c->nhood) {
      n->removeCommit (c);
    }

    c->setState (ContextState::COMMIT_DONE);
  }

  static void doAbort(Ctxt* c) {
    // this can be in states READY_TO_COMMIT, ABORT_SELF
    // children can be in UNSCHEDULED, READY_TO_COMMIT, ABORT_DONE

    // first abort all the children recursively
    // then abort self.
    //

    assert (c->hasState (ContextState::ABORTING));

    dbg::print (c, " aborting with item ", c->getActive());

    c->userHandle.rollback();

    for (NItem* ni: c->nhood) {
      ni->removeAbort (c);
    }

    if (c->addBack) {

      c->setState (ContextState::ABORT_DONE);
      c->exec.push_abort (c);

    } else {
      // is an aborted child whose parent also aborted
      c->setState (ContextState::ABORTED_CHILD);
    }

  }

  static bool isCommitSrc (const Ctxt* c) {

    for (const NItem* ni: c->nhood) {

      if (ni->getHistHead() != c) {
        return false;
      }
    }

    return true;
  }

  template <typename WL>
  static void findCommitSrc (const Ctxt* ctxt, const Ctxt* gvt, WL& wl) {

    for (const NItem* ni: ctxt->nhood) {

      Ctxt* c = ni->getHistHead();
      assert (c != ctxt);

      if (c && (!gvt || ctxt->exec.getCtxtCmp() (c, gvt)) 
          && c->isCommitSrc() 
          && c->onWL.cas (false, true)) {
        wl.push (c);
      }
    }
  }

  static bool isAbortSrc (const Ctxt* c) {

    if (!c->hasState (ContextState::READY_TO_ABORT)) {
      return false;
    }

    for (const NItem* ni: c->nhood) {

      if (ni->getHistTail() != c) {
        return false;
      }
    }

    return true;
  }

  template <typename WL>
  static void findAbortSrc (const Ctxt* ctxt, WL& wl) {

    // XXX: if a task has children that don't share neighborhood with
    // it, should it be an abort source? Yes, because the end goal in 
    // finding abort sources is that tasks may abort and restore state
    // in isolation. 
    
    for (const NItem* ni: ctxt->nhood) {

      Ctxt* c = ni->getHistTail();

      if (c && c->isAbortSrc() && c->onWL.cas (false, true)) {
        wl.push (c);
      }
    }
  }

  static bool isSrcSlowCheck (const Ctxt* c) {
    
    for (const NItem* ni: c->nhood) {

      if (!ni->isMin(c)) {
        return false;
      }
    }

    return true;
  }

  static void addAbortLocation (Ctxt* c, NItem* ni) {
    assert (ni);
    assert (std::find (c->nhood.begin(), c->nhood.end(), ni) != c->nhood.end());
    c->exec.addAbortLocation (ni);
  }

  /*
  template <typename WL>
  bool findAborts (WL& abortRoots) {

    assert (isSrcSlowCheck());

    bool ret = false;

    for (NItem* ni: nhood) {
      ret = ni->findAborts (this, abortRoots) || ret;
    }

    return ret;
  }
  */


  template <typename WL>
  static void markForAbortRecursive (Ctxt* ctxt, WL& abortRoots) {
    if (ctxt->casState (ContextState::READY_TO_COMMIT, ContextState::READY_TO_ABORT)) {

      for (NItem* ni: ctxt->nhood) {
        ni->markForAbort (ctxt, abortRoots);
      }

      if (ctxt->isAbortSrc() && ctxt->onWL.cas (false, true)) {
        abortRoots.push (ctxt);
      }

      for (Ctxt* c: ctxt->children) {
        dbg::print (ctxt, " causing abort on child ", c);
        c->markForAbortRecursive (abortRoots);
        c->addBack = false;
      }

    } else if (ctxt->casState (ContextState::SCHEDULED, ContextState::ABORTED_CHILD)) {
      // a SCHEDULED task can only be aborted recursively if it's a child

    } else if (ctxt->casState (ContextState::UNSCHEDULED, ContextState::ABORTED_CHILD)) {

    } else {
      assert (ctxt->hasState (ContextState::READY_TO_ABORT) || ctxt->hasState (ContextState::ABORTED_CHILD));
    }

    assert (ctxt->hasState (ContextState::READY_TO_ABORT) || ctxt->hasState (ContextState::ABORTED_CHILD));

  }

  static void addToHistory (Ctxt* c) {

    for (NItem* ni: c->nhood) {
      ni->addToHistory (c);
    }
  }
};

template <typename T, typename Cmp, typename Exec>
struct OptimContext: public SpecContextBase<T, Cmp, Exec> {

  using Base = SpecContextBase<T, Cmp, Exec>;

  using CtxtCmp = typename Base::CtxtCmp;
  using NItem = OptimNItem<OptimContext, CtxtCmp>;
  using NhoodMgr = PtrBasedNhoodMgr<NItem>;
  using NhoodList = typename gstl::Vector<NItem*>;
  using ChildList = typename gstl::Vector<OptimContext*>;

  using CF = OptimContextFunctions<OptimContext, NItem>;

  galois::GAtomic<bool> onWL;
  bool addBack; // set to false by parent when parent is marked for abort, see markAbortRecursive
  NhoodList nhood;
  ChildList children;

  explicit OptimContext (const T& x, const ContextState& s, Exec& exec)
  :
    Base (x, s, exec),
    onWL (false),
    addBack (true)
  {}

  GALOIS_ATTRIBUTE_PROF_NOINLINE
  virtual void subAcquire (Lockable* l, galois::MethodFlag m) {

    NItem& nitem = Base::exec.getNhoodMgr().getNhoodItem (l);
    assert (NItem::getOwner (l) == &nitem);

    if (std::find (nhood.begin(), nhood.end(), &nitem) == nhood.end()) {
      nhood.push_back (&nitem);
      nitem.markMin (this);
    }
  }

  virtual bool owns (Lockable* l, MethodFlag m) const {
    NItem& nitem = Base::exec.getNhoodMgr().getNhoodItem (l);
    assert (NItem::getOwner (l) == &nitem);
    return (nitem.getMin () ==  this);
  }

  void schedule (void) {
    Base::schedule();
    onWL = false;
    addBack = true;
    nhood.clear();
    children.clear();
  }

  void resetMarks (void) {

    for (NItem* ni: nhood) {
      if (ni->getMin() == this) {
        ni->resetMin (this);
      }
    }
  }

  
  void addChild (OptimContext* child) {
    CF::addChild (this, child);
  }

  void doCommit (void) {
    CF::doCommit (this);
  }

  void doAbort (void) {
    CF::doAbort (this);
  }

  bool isCommitSrc (void) const {
    return CF::isCommitSrc (this);
  }

  template <typename WL>
  void findCommitSrc (const OptimContext* gvt, WL& wl) const {
    CF::findCommitSrc (this, gvt, wl);
  }

  bool isAbortSrc (void) const {
    return CF::isAbortSrc (this);
  }

  template <typename WL>
  void findAbortSrc (WL& wl) const {
    CF::findAbortSrc (this, wl);
  }

  bool isSrcSlowCheck (void) const {
    return CF::isSrcSlowCheck (this);
  }

  void addAbortLocation (NItem* ni) {
    CF::addAbortLocation (this, ni);
  }

  template <typename WL>
  void markForAbortRecursive (WL& abortRoots) {
    CF::markForAbortRecursive (this, abortRoots);
  }

  void addToHistory (void) {
    CF::addToHistory (this);
  }

};

template <typename T, typename Cmp, typename NhFunc, typename ExFunc, typename OpFunc, typename ArgsTuple, typename Ctxt>
class OrdSpecExecBase: public IKDGbase<T, Cmp, NhFunc, ExFunc, OpFunc, ArgsTuple, Ctxt> {

 protected:

  using Base = IKDGbase <T, Cmp, NhFunc, ExFunc, OpFunc, ArgsTuple, Ctxt>;
  using Derived = typename Ctxt::Executor;

  using CtxtCmp = typename Ctxt::CtxtCmp;
  using CtxtWL = typename Base::CtxtWL;

  using ThisClass = OrdSpecExecBase;

  using CommitQ = galois::PerThreadVector<Ctxt*>;
  using ExecutionRecords = std::vector<ParaMeter::StepStats>;

  static const unsigned DEFAULT_CHUNK_SIZE = 4;

  struct CtxtMaker {
    OrdSpecExecBase& outer;

    Ctxt* operator() (const T& x) {

      Ctxt* ctxt = outer.ctxtAlloc.allocate (1);
      assert (ctxt);
      outer.ctxtAlloc.construct (ctxt, x, ContextState::UNSCHEDULED, static_cast<Derived&> (outer));

      return ctxt;
    }
  };


  CtxtMaker ctxtMaker;

  GAccumulator<size_t> totalRetires;
  GAccumulator<size_t> totalAborts;

  CommitQ commitQ;
  ExecutionRecords execRcrds;


public:
  OrdSpecExecBase (const Cmp& cmp, const NhFunc& nhFunc, const ExFunc& exFunc, const OpFunc& opFunc, const ArgsTuple& argsTuple)
    : 
      Base (cmp, nhFunc, exFunc, opFunc, argsTuple),
      ctxtMaker {*this}
  {
  }

  ~OrdSpecExecBase (void) {
    dumpStats();
  }

  CtxtMaker& getCtxtMaker(void) {
    return ctxtMaker;
  }

protected:

  void dumpParaMeterStats (void) {
    // remove last record if its 0
    if (!execRcrds.empty() && execRcrds.back().parallelism.reduceRO() == 0) {
      execRcrds.pop_back();
    }

    for (const ParaMeter::StepStats& s: execRcrds) {
      s.dump (ParaMeter::getStatsFile(), this->loopname);
    }

    ParaMeter::closeStatsFile();
  }

  void dumpStats (void) {

    reportStat_Serial (this->loopname, "retired", totalRetires.reduce());
    reportStat_Serial (this->loopname, "efficiency%", double (100 * totalRetires.reduce()) / this->totalTasks);
    reportStat_Serial (this->loopname, "avg. parallelism", double (totalRetires.reduce()) / this->rounds);

    if (ThisClass::ENABLE_PARAMETER) {
      dumpParaMeterStats();
    }
  }



  template <typename W>
  GALOIS_ATTRIBUTE_PROF_NOINLINE void beginRound(W& winWL) {

    this->beginRound (winWL);

    if (ThisClass::ENABLE_PARAMETER) {
      execRcrds.emplace_back (this->rounds, this->getCurrWL().size_all());
    }
  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE void expandNhood (void) {

    galois::do_all_choice (makeLocalRange (this->getCurrWL()),
        [this] (Ctxt* c) {

          if (!c->hasState (ContextState::ABORTED_CHILD)) {

            dbg::print ("scheduling: ", c, " with item: ", c->getActive());

            assert (!c->hasState (ContextState::RECLAIM));
            c->schedule();

            typename Base::UserCtxt& uhand = c->userHandle;

            // nhFunc (c, uhand);
            runCatching (this->nhFunc, c, uhand);

            this->roundTasks += 1;
          }
        },
        std::make_tuple (
          galois::loopname ("expandNhood"),
          chunk_size<NhFunc::CHUNK_SIZE>()));

  }


  void freeCtxt (Ctxt* ctxt) {
    this->ctxtAlloc.destroy (ctxt);
    this->ctxtAlloc.deallocate (ctxt, 1);
  }

  // FOR DEBUGGING
  /*
  Ctxt* computeGVT (void) {

    // t_computeGVT.start();

    substrate::PerThreadStorage<Ctxt*> perThrdMin;

    on_each_impl ([this, &perThrdMin] (const unsigned tid, const unsigned numT) {
          
          for (auto i = Base::getNextWL().local_begin()
            , end_i = Base::getNextWL().local_end(); i != end_i; ++i) {

            Ctxt*& lm = *(perThrdMin.getLocal());

            if (!lm || Base::ctxtCmp (*i, lm)) {
              lm = *i;
            }
          }

          
        });

    Ctxt* ret = nullptr;

    for (unsigned i = 0; i < perThrdMin.size(); ++i) {

      Ctxt* lm = *(perThrdMin.getRemote (i));

      if (lm) {
        if (!ret || Base::ctxtCmp (lm, ret)) {
          ret = lm;
        }
      }
    }

    Ctxt* const* minWinWL = winWL.getMin();

    if (minWinWL) {
      if (!ret || Base::ctxtCmp (*minWinWL, ret)) {
        ret = *minWinWL;
      }
    }

    // t_computeGVT.stop();

    return ret;

  }
  */
};

template <typename T, typename Cmp, typename NhFunc, typename ExFunc, typename  OpFunc, typename ArgsTuple, typename Ctxt>
class OptimOrderedExecBase: public OrdSpecExecBase<T, Cmp, NhFunc, ExFunc, OpFunc, ArgsTuple, Ctxt> {

protected:

  using Base = OrdSpecExecBase<T, Cmp, NhFunc, ExFunc, OpFunc, ArgsTuple, Ctxt>;
  using ThisClass = OptimOrderedExecBase;

  using NhoodMgr = typename Ctxt::NhoodMgr;
  using CtxtCmp = typename Ctxt::CtxtCmp;
  using NItem = typename Ctxt::NItem;
  using NItemFactory = typename NItem::Factory;
  using CtxtWL = typename Base::CtxtWL;

  using WindowWL = PQwindowWL<Ctxt*, CtxtCmp>;

  WindowWL winWL;
  Ctxt* minWinWL;
  NItemFactory nitemFactory;
  NhoodMgr nhmgr;
  PerThreadBag<NItem*> abortLocations;
  substrate::PerThreadStorage<Ctxt*> currMinPending; // reset at the beginning of each round

  CtxtWL abortRoots;
  CtxtWL commitRoots;

public:

  OptimOrderedExecBase (const Cmp& cmp, const NhFunc& nhFunc, const ExFunc& exFunc, const OpFunc& opFunc, const ArgsTuple& argsTuple)
    : 
      Base (cmp, nhFunc, exFunc, opFunc, argsTuple),
      winWL (this->ctxtCmp),
      nitemFactory (this->ctxtCmp),
      nhmgr (nitemFactory)
  {
  }

  template <typename R>
  void push_initial (const R& range) {

    StatTimer t ("push_initial");
    t.start();

    const bool USE_WIN = (this->targetCommitRatio != 0.0);
    galois::do_all_choice (range,
        [this, USE_WIN] (const T& x) {

          Ctxt* c = this->ctxtMaker (x);
          if (USE_WIN) {
            winWL.push (c);
          } else {
            this->getNextWL().push (c);
          }

        }, 
        std::make_tuple (
          galois::loopname ("init-fill"),
          chunk_size<ThisClass::DEFAULT_CHUNK_SIZE>()));

    t.stop();

  }

  NhoodMgr& getNhoodMgr (void) {
    return nhmgr;
  }

  void addAbortLocation (NItem* ni) {
    if (!ThisClass::NEEDS_CUSTOM_LOCKING) {
      assert (ni);
      abortLocations.push (ni);
    }
  }

  void push_abort (Ctxt* ctxt) {
    assert (ctxt);
    assert (ctxt->hasState (ContextState::ABORT_DONE));

    ctxt->setState (ContextState::UNSCHEDULED);

    if (!minWinWL || this->ctxtCmp(ctxt, minWinWL)) {
      updateCurrMinPending (ctxt);
      this->getNextWL().push (ctxt);

    } else {
      winWL.push(ctxt);
    }

  }

protected:

  // invoked after beginRound
  void resetMinWinWL (void) {
    minWinWL = nullptr;

    if (this->targetCommitRatio != 0.0) {
      galois::optional<Ctxt*> m = winWL.getMin();
      if (m) {
        minWinWL = *m;
      }
    }
  }

  void resetCurrMinPending(void) {
    // reset currMinPending
    on_each_impl (
        [this] (const unsigned tid, const unsigned numT) {
          *(currMinPending.getLocal()) = nullptr;
        });

  }

  void updateCurrMinPending (Ctxt* c) {
    Ctxt*& minPending = *currMinPending.getLocal();

    if (!minPending || this->ctxtCmp (c, minPending)) {
      minPending = c;
    }
  }


  Ctxt* getMinPending (void) {
    Ctxt* m = minWinWL;

    for (unsigned i = 0; i < galois::getActiveThreads(); ++i) {
      Ctxt* c = *currMinPending.getRemote (i);

      if (!c) { continue; }

      if (!m || this->ctxtCmp (c, m)) {
        m = c;
      }
    }

    return m;
  }

  void beginRound(void) {
    this->beginRound(winWL);
    resetCurrMinPending();
    resetMinWinWL();
  }

  Ctxt* push_commit (const T& x) {

    Ctxt* c = this->ctxtMaker (x); 
    assert(c);


    if (!minWinWL || this->ctxtCmp (c, minWinWL)) {
      this->getNextWL().push_back (c);

      dbg::print ("Child going to nextWL, c: ", c, ", with active: ", c->getActive());

      updateCurrMinPending (c);

    } else {
      assert (!this->ctxtCmp (c, minWinWL));
      winWL.push (c);

      dbg::print ("Child going to winWL, c: ", c, ", with active: ", c->getActive());
    }

    return c;
  }


  void quickAbort (Ctxt* c) {
    assert (c);
    bool b= c->hasState (ContextState::SCHEDULED) 
      || c->hasState (ContextState::UNSCHEDULED)
      || c->hasState (ContextState::ABORTED_CHILD) 
      || c->hasState (ContextState::ABORT_DONE);

    assert (b);

    if (c->casState (ContextState::SCHEDULED, ContextState::ABORT_DONE)) {
      push_abort (c);
      dbg::print("Quick Abort c: ", c, ", with active: ", c->getActive());

    } 
  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE void serviceAborts (void) {

    assert (abortRoots.empty_all());

    if (abortLocations.empty_all ()) { 
      assert (abortRoots.empty_all ());
      return; 
    }

    galois::do_all_choice (makeLocalRange (abortLocations),
        [this] (NItem* ni) {

          Ctxt* c = ni->getMin();
          ni->findAborts (c, abortRoots);

          // quickAbort (c); // not needed as applyOperator aborts 
          // c->disableSrc();
        },
        std::make_tuple (
          galois::loopname ("mark-aborts"),
          galois::chunk_size<ThisClass::DEFAULT_CHUNK_SIZE>()));


    galois::runtime::for_each_gen (
        makeLocalRange (abortRoots),
        [this] (Ctxt* c, UserContext<Ctxt*>& wlHandle) {

          if (c->casState (ContextState::READY_TO_ABORT, ContextState::ABORTING)) {
            c->doAbort();
            c->findAbortSrc (wlHandle);
          
          } else {
            assert (c->hasState (ContextState::ABORTING) || c->hasState (ContextState::ABORT_DONE));
          }

          dbg::print("aborted after execution:", c, " with active: ", c->getActive());
        },
        std::make_tuple (
          galois::loopname ("handle-aborts"),
          galois::does_not_need_aborts<>(),
          galois::combine_stats_by_name<>(),
          galois::wl<galois::worklists::AltChunkedFIFO<ThisClass::DEFAULT_CHUNK_SIZE> >()));


    galois::runtime::on_each_impl(
        [this] (const unsigned tid, const unsigned numT) {
          abortRoots.get().clear();
          abortLocations.get().clear();
        });
  }

  /*
  GALOIS_ATTRIBUTE_PROF_NOINLINE void serviceAborts (CtxtWL& sources) {

    abortRoots.clear_all_parallel();

    galois::do_all_choice (makeLocalRange (Base::getCurrWL()),
        [this, &abortRoots] (Ctxt* c) {

          if (c->isSrc()) {

            assert (c->isSrcSlowCheck());

            if (c->findAborts (abortRoots)) {
              // XXX: c does not need to abort if it's neighborhood
              // isn't dependent on values computed by other tasks
              

              c->disableSrc();
              dbg::print("Causing rollbacks:", c, " with active: ", c->getActive());
            }

          } 
        },
        std::make_tuple (
          galois::loopname ("mark-aborts"),
          galois::chunk_size<Base::DEFAULT_CHUNK_SIZE>()));


    galois::runtime::for_each_gen (
        makeLocalRange (abortRoots),
        [this] (Ctxt* c, UserContext<Ctxt*>& wlHandle) {

          if (c->casState (ContextState::READY_TO_ABORT, ContextState::ABORTING)) {
            c->doAbort();
            c->findAbortSrc (wlHandle);
          
          } else {
            assert (c->hasState (ContextState::ABORTING) || c->hasState (ContextState::ABORT_DONE));
          }

          dbg::print("aborted after execution:", c, " with active: ", c->getActive());
        },
        std::make_tuple (
          galois::loopname ("handle-aborts"),
          galois::does_not_need_aborts_tag(),
          galois::wl<galois::worklists::dChunkedFIFO<NhFunc::CHUNK_SIZE> >()));
    


    galois::do_all_choice (makeLocalRange (Base::getCurrWL()),

        [this, &sources] (Ctxt* c) {
          if (c->isSrc() && !c->hasState (ContextState::ABORTED_CHILD)) {
            assert (c->hasState (ContextState::SCHEDULED));

            sources.push (c);

          } else if (c->hasState (ContextState::ABORTED_CHILD)) {
            Base::commitQ.get().push_back (c); // for reclaiming memory 

          } else {
            assert (!c->hasState (ContextState::ABORTED_CHILD));
            quickAbort (c);
          }

          c->resetMarks();
        },
        std::make_tuple ( 
          galois::loopname ("collect-sources"),
          galois::chunk_size<Base::DEFAULT_CHUNK_SIZE>()));

  }
  */

  GALOIS_ATTRIBUTE_PROF_NOINLINE void performCommits() {


    Ctxt* gvt = getMinPending();

    // TODO: remove this after debugging
    // Ctxt* gvtAlt = Base::computeGVT();
// 
    // assert (gvt == gvtAlt);

    if (gvt) {
      dbg::print ("GVT computed as: ", gvt, ", with elem: ", gvt->getActive());
    } else {
      dbg::print ("GVT computed as NULL");
    }


    assert (commitRoots.empty_all());

    galois::do_all_choice (makeLocalRange (this->commitQ),
        [this, gvt] (Ctxt* c) {

          assert (c);

          if (c->hasState (ContextState::READY_TO_COMMIT) 
              && (!gvt || this->ctxtCmp (c, gvt))
              && c->isCommitSrc()) {

            commitRoots.push (c);
          }
        },
        std::make_tuple (
          galois::loopname ("find-commit-srcs"),
          galois::chunk_size<ThisClass::DEFAULT_CHUNK_SIZE>()));
        

    galois::runtime::for_each_gen (
        makeLocalRange (commitRoots),
        [this, gvt] (Ctxt* c, UserContext<Ctxt*>& wlHandle) {

          bool b = c->casState (ContextState::READY_TO_COMMIT, ContextState::COMMITTING);

          if (b) {

            assert (c->isCommitSrc());
            if (gvt) {
              assert (this->ctxtCmp (c, gvt));
            }

            c->doCommit();
            c->findCommitSrc (gvt, wlHandle);
            this->totalRetires += 1;

            if (ThisClass::ENABLE_PARAMETER) {
              assert (c->getExecRound() < this->execRcrds.size());
              this->execRcrds[c->getExecRound()].parallelism += 1;
            }

          } else {
            assert (c->hasState (ContextState::COMMIT_DONE));
          }
        },
        std::make_tuple (
          galois::loopname ("retire"),
          galois::does_not_need_aborts<>(),
          galois::combine_stats_by_name<>(),
          galois::wl<galois::worklists::AltChunkedFIFO<Base::DEFAULT_CHUNK_SIZE> >()));

    commitRoots.clear_all_parallel();
  }

  void freeCtxt (Ctxt* ctxt) {
    this->ctxtAlloc.destroy (ctxt);
    this->ctxtAlloc.deallocate (ctxt, 1);
  }

  void reclaimMemory (void) {

    // XXX: the following memory free relies on the fact that 
    // per-thread fixed allocators are being used. Otherwise, mem-free
    // should be done in a separate loop, after enforcing set semantics
    // among all threads


    galois::runtime::on_each_impl (
        [this] (const unsigned tid, const unsigned numT) {
          
          auto& localQ = this->commitQ.get();
          auto new_end = std::partition (localQ.begin(), 
            localQ.end(), 
            [] (Ctxt* c) {
              assert (c);
              return c->hasState (ContextState::READY_TO_COMMIT);
            });


          for (auto i = new_end, end_i = localQ.end(); i != end_i; ++i) {

            if ((*i)->casState (ContextState::ABORTED_CHILD, ContextState::RECLAIM)
              || (*i)->casState (ContextState::COMMIT_DONE, ContextState::RECLAIM)) {
              dbg::print ("Ctxt destroyed from commitQ: ", *i);
              freeCtxt (*i);
            }
          }

          localQ.erase (new_end, localQ.end());
        });

  }



};

template <typename T, typename Cmp, typename NhFunc, typename ExFunc, typename  OpFunc, typename ArgsTuple>
class OptimOrdExecutor: public OptimOrderedExecBase<T, Cmp, NhFunc, ExFunc, OpFunc, ArgsTuple,
  OptimContext<T, Cmp, OptimOrdExecutor<T, Cmp, NhFunc, ExFunc, OpFunc, ArgsTuple> > > {

    using ThisClass = OptimOrdExecutor;

protected:

  friend struct OptimContext<T, Cmp, OptimOrdExecutor>;
  using Ctxt = OptimContext<T, Cmp, OptimOrdExecutor>;
  using Base = OptimOrderedExecBase<T, Cmp, NhFunc, ExFunc, OpFunc, ArgsTuple, Ctxt>;

  using NhoodMgr = typename Ctxt::NhoodMgr;
  using CtxtCmp = typename Ctxt::CtxtCmp;
  using NItem = typename Ctxt::NItem;
  using NItemFactory = typename NItem::Factory;
  using CtxtWL = typename Base::CtxtWL;




public:
  OptimOrdExecutor (const Cmp& cmp, const NhFunc& nhFunc, const ExFunc& exFunc, const OpFunc& opFunc, const ArgsTuple& argsTuple)
    : 
      Base (cmp, nhFunc, exFunc, opFunc, argsTuple)

  {
  }


  void operator() (void) {
    execute();
  }
  
  void execute() {

    StatTimer t ("executorLoop");

    t.start();

    while (true) {

      this->t_beginRound.start();
      this->beginRound();
      // resetCurrMinPending(); already invoked by OptimOrderedExecBase::beginRound
      this->t_beginRound.stop();

      if (this->getCurrWL().empty_all()) {
        break;
      }

      this->t_expandNhood.start();
      this->expandNhood();
      this->t_expandNhood.stop();

      this->t_serviceAborts.start();
      this->serviceAborts();
      this->t_serviceAborts.stop();

      this->t_executeSources.start();
      executeSources();
      this->t_executeSources.stop();

      this->t_applyOperator.start();
      applyOperator();
      this->t_applyOperator.stop();

      this->t_performCommits.start();
      this->performCommits();
      this->t_performCommits.stop();

      this->t_reclaimMemory.start();
      this->reclaimMemory();
      this->t_reclaimMemory.stop();

      this->endRound();

    }

    t.stop();
  }

protected:

  GALOIS_ATTRIBUTE_PROF_NOINLINE void executeSources (void) {

    if (ThisClass::HAS_EXEC_FUNC) {


      galois::do_all_choice (makeLocalRange (this->getCurrWL()),
        [this] (Ctxt* ctxt) {
          bool _x = ctxt->hasState (ContextState::ABORTED_CHILD) 
                 || ctxt->hasState (ContextState::SCHEDULED)
                 || ctxt->hasState (ContextState::UNSCHEDULED);

          assert (_x);

          assert (ctxt);
          if (ctxt->isSrc () && ctxt->hasState (ContextState::SCHEDULED)) {
            this->exFunc (ctxt->getActive(), ctxt->userHandle);
          }
        },
        std::make_tuple (
          galois::loopname ("executeSources"),
          galois::chunk_size<ExFunc::CHUNK_SIZE>()));

    }
  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE void applyOperator (void) {

    // Ctxt in currWL can be in SCHEDULED, UNSCHEDULED (after having been aborted),
    // ABORTED_CHILD

    galois::do_all_choice (makeLocalRange (this->getCurrWL()),
        [this] (Ctxt* c) {

          bool _x = c->hasState (ContextState::ABORTED_CHILD) 
                 || c->hasState (ContextState::SCHEDULED)
                 || c->hasState (ContextState::UNSCHEDULED);

          assert (_x);

          if (c->hasState (ContextState::SCHEDULED)) {


            if (this->NEEDS_CUSTOM_LOCKING || c->isSrc ()) {
              bool commit = c->isSrc ();

              typename Base::UserCtxt& uhand = c->userHandle;
              if (this->NEEDS_CUSTOM_LOCKING) {
                c->enableSrc ();
                runCatching (this->opFunc, c, uhand);
                commit = c->isSrc(); // in case opFunc signalled abort

              } else {
                this->opFunc (c->getActive(), uhand);
                commit = true;
              }

              if (commit) {
                if (this->NEEDS_PUSH) {

                  for (auto i = uhand.getPushBuffer().begin()
                      , endi = uhand.getPushBuffer().end(); i != endi; ++i) {

                    Ctxt* child = this->push_commit (*i);
                    c->addChild (child);

                  }
                  uhand.getPushBuffer().clear();
                } else {

                  assert (uhand.getPushBuffer().begin() == uhand.getPushBuffer().end());
                }

                bool b = c->casState (ContextState::SCHEDULED, ContextState::READY_TO_COMMIT);

                assert (b && "CAS shouldn't have failed");
                this->roundCommits += 1;

                c->addToHistory();
                this->commitQ.get().push_back (c);

                if (this->ENABLE_PARAMETER) {
                  c->markExecRound (this->rounds);
                }

              } else {
                  // uhand.rollback();
                  this->quickAbort (c);
              } // end if commit                

            } else { // if !isSrc
              this->quickAbort (c);
            }
          } else if (c->hasState (ContextState::ABORTED_CHILD)) {

            this->commitQ.get().push_back (c); // for reclaiming memory 

          } else {
            assert (c->hasState (ContextState::UNSCHEDULED));
            // do nothing. UNSCHEDULED was reached because c was aborted in serviceAborts
          }

          c->resetMarks();
        },
        std::make_tuple (
          galois::loopname ("applyOperator"),
          galois::chunk_size<OpFunc::CHUNK_SIZE>()));

  }



};


template <typename T, typename Cmp, typename Exec>
class PessimOrdContext: public SpecContextBase<T, Cmp, Exec> {

public:

  using Base = SpecContextBase<T, Cmp, Exec>;
  using NhoodList =  galois::gstl::Vector<Lockable*>;
  using CtxtCmp = typename Base::CtxtCmp;
  using Executor = Exec;

  
  NhoodList nhood;

  explicit PessimOrdContext (const T& x, const ContextState& s, Exec& e)
    : 
      Base (x, s, e)

  {}

  void schedule() {
    Base::schedule();
    nhood.clear();
  }


  bool priorityAcquire (Lockable* l) {
    PessimOrdContext* other = nullptr;

    do {
      other = static_cast<PessimOrdContext*> (Base::getOwner (l));

      if (other == this) {
        return true;
      }

      if (other) {
        bool conflict = Base::exec.getCtxtCmp() (other, this); // *other < *this
        if (conflict) {
          // A lock that I want but can't get
          this->disableSrc();
          dbg::print (this, " lost to ", other);
          return false; 
        }
      }
    } while (!this->CASowner(l, other));

    // Disable loser
    if (other) {
      other->disableSrc();

      if (other->casState (ContextState::READY_TO_COMMIT, ContextState::ABORT_HELP)) {

        Base::exec.markForAbort (other);
        dbg::print (this, " marking for abort: ", other);
        // this->disableSrc();// abort self to recompute after other has abortedthis->disableSrc();
      } else if (other->hasState (ContextState::ABORT_HELP)) {
        // this->disableSrc(); // abort self to recompute after other has aborted
      }

    }

    return true;
  }

  virtual bool owns (Lockable* l, MethodFlag m) const {
    return static_cast<PessimOrdContext*> (Base::getOwner (l)) == this;
  }

  // TODO: Refactor common code with TwoPhaseContext::subAcquire
  virtual void subAcquire (Lockable* l, galois::MethodFlag) {

    dbg::print (this, " trying to acquire ", l);

    if (std::find (nhood.cbegin(), nhood.cend(), l) == nhood.cend()) {

      nhood.push_back (l);

      bool succ = priorityAcquire (l);

      if (succ) {
        // dbg::print (this, " acquired lock ", l);
      } else {
        assert (!this->isSrc());
        // dbg::print (this, " failed to acquire lock ", l);
      }


    } // end if find

    return;
  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE void doCommit() {
    assert (Base::hasState (ContextState::COMMITTING));

    // executor must already have pushed new work from userHandle.getPushBuffer
    // release locks
    dbg::print (this, " committing with item ", this->getActive());

    Base::userHandle.commit();
    releaseLocks();
    Base::setState (ContextState::COMMIT_DONE);
  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE void doAbort() {
    assert (Base::hasState (ContextState::ABORTING));
    // perform undo actions in reverse order
    // release locks
    // add active element to worklist
    dbg::print (this, " aborting with item ", this->getActive());

    Base::userHandle.rollback();
    releaseLocks();
    Base::setState (ContextState::ABORT_DONE);
    Base::exec.push_abort (this);

  }

private:

  void releaseLocks() {
    for (Lockable* l: nhood) {
      assert (l != nullptr);
      if (static_cast<PessimOrdContext*> (Base::getOwner (l)) == this) {
        // dbg::print (this, " releasing lock ", l);
        bool b = Base::tryLock (l); // release requires having had the lock
        assert (b);
        Base::release (l);
      }
    }

  }


};

template <typename T, typename Cmp, typename NhFunc, typename ExFunc, typename  OpFunc, typename ArgsTuple>
class PessimOrdExecutor: public OrdSpecExecBase<T, Cmp, NhFunc, ExFunc, OpFunc, ArgsTuple,
  PessimOrdContext<T, Cmp, PessimOrdExecutor<T, Cmp, NhFunc, ExFunc, OpFunc, ArgsTuple> > > {

    using ThisClass = PessimOrdExecutor;

protected:

  friend class PessimOrdContext<T, Cmp, PessimOrdExecutor>;
  using Ctxt = PessimOrdContext<T, Cmp, PessimOrdExecutor>;
  using Base = OrdSpecExecBase<T, Cmp, NhFunc, ExFunc, OpFunc, ArgsTuple, Ctxt>;

  using CtxtCmp = typename Ctxt::CtxtCmp;
  using CtxtWL = typename Base::CtxtWL;

  typename Base::template WindowWLwrapper<PessimOrdExecutor> winWL;
  galois::optional<T> minWinWL;

  substrate::PerThreadStorage<galois::optional<T> > currMinPending; // reset at the beginning of each round
  CtxtWL abortRoots;
  CtxtWL freeWL;


public:

  PessimOrdExecutor (const Cmp& cmp, const NhFunc& nhFunc, const ExFunc& exFunc, const OpFunc& opFunc, const ArgsTuple& argsTuple)
    : 
      Base (cmp, nhFunc, exFunc, opFunc, argsTuple),
      winWL (*this, cmp)
  {
  }

  template <typename R>
  void push_initial (const R& range) {
    if (this->targetCommitRatio == 0.0) {

      galois::do_all_choice (range,
          [this] (const T& x) {
            this->getNextWL ().push_back (this->ctxtMaker (x));
          }, 
          std::make_tuple (
            galois::loopname ("init-fill"),
            chunk_size<NhFunc::CHUNK_SIZE> ()));

    } else {
      winWL.initfill (range);
          
    }
  }

  void markForAbort (Ctxt* c) {
    assert (c);
    abortRoots.push (c);
  }

  void execute (void) {
    StatTimer t ("executorLoop");

    t.start();

    while (true) {

      this->t_beginRound.start();
      this->beginRound(winWL);
      resetCurrMinPending();
      resetMinWinWL();
      this->t_beginRound.stop();

      if (this->getCurrWL().empty_all()) {
        break;
      }

      this->t_expandNhood.start();
      this->expandNhood();
      this->t_expandNhood.stop();

      this->t_serviceAborts.start();
      serviceAborts();
      this->t_serviceAborts.stop();

      this->t_executeSources.start();
      executeSources();
      this->t_executeSources.stop();

      this->t_applyOperator.start();
      applyOperator();
      this->t_applyOperator.stop();

      this->t_performCommits.start();
      performCommits();
      this->t_performCommits.stop();

      this->endRound();

    }

    t.stop();

  }

protected:


  void resetCurrMinPending(void) {
    // reset currMinPending
    on_each_impl (
        [this] (const unsigned tid, const unsigned numT) {
          *(currMinPending.getLocal()) = galois::optional<T>();
        });
  }

  void updateCurrMinPending (const T& elem) {
    galois::optional<T>& minPending = *currMinPending.getLocal();

    if (!minPending || this->getItemCmp()(elem, *minPending)) {
      minPending = elem;
    }
  }

  // invoked after beginRound
  void resetMinWinWL (void) {
    minWinWL = galois::optional<T>();
    if (this->targetCommitRatio != 0.0) {
      minWinWL = winWL.getMin();
    }
  }

  galois::optional<T> getMinPending (void) {
    galois::optional<T> m = minWinWL;

    for (unsigned i = 0; i < galois::getActiveThreads(); ++i) {
      galois::optional<T> c = *currMinPending.getRemote (i);

      if (!c) { continue; }

      if (!m || this->getItemCmp()(*c, *m)) {
        m = c;
      }
    }

    return m;
  }

  void push_commit (const T& x) {


    if (!minWinWL || this->getItemCmp()(x, *minWinWL)) {

      Ctxt* c = this->ctxtMaker (x); 
      assert (c);
      dbg::print ("Child going to nextWL, c: ", c, ", with active: ", c->getActive());
      this->getNextWL().push_back (c);

      updateCurrMinPending (x);

    } else {
      dbg::print ("Child going to winWL, active: ", x);
      assert(!this->getItemCmp()(x, *minWinWL));
      winWL.push(x);
    }
  }

  void push_abort (Ctxt* ctxt) {
    assert (ctxt);
    assert (ctxt->hasState (ContextState::ABORT_DONE));

    ctxt->setState (ContextState::UNSCHEDULED);

    if (!minWinWL || this->getItemCmp()(ctxt->getActive(), *minWinWL)) {
      updateCurrMinPending (ctxt->getActive());
      this->getNextWL().push (ctxt);

    } else {
      winWL.push(ctxt->getActive());
      this->freeCtxt(ctxt);
    }

  }

  void serviceAborts() {

    galois::do_all_choice (makeLocalRange (abortRoots),
        [this] (Ctxt* c) {

          bool b = c->hasState (ContextState::ABORT_HELP) 
            || c->hasState (ContextState::ABORTING)
            || c->hasState (ContextState::ABORT_DONE);
          assert (b);

          if (c->casState (ContextState::ABORT_HELP, ContextState::ABORTING)) {
            c->doAbort();
          }

        },
        std::make_tuple (
          galois::loopname ("abort-marked"),
          galois::chunk_size<ThisClass::DEFAULT_CHUNK_SIZE>()));

    abortRoots.clear_all_parallel();

  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE void executeSources (void) {

    if (ThisClass::HAS_EXEC_FUNC) {

      galois::do_all_choice (makeLocalRange (this->getCurrWL()),
        [this] (Ctxt* ctxt) {

          if (ctxt->isSrc()) {
            assert (ctxt->hasState (ContextState::SCHEDULED));
            this->exFunc (ctxt->getActive(), ctxt->userHandle);
          }

        },
        std::make_tuple (
          galois::loopname ("executeSources"),
          galois::chunk_size<ExFunc::CHUNK_SIZE>()));

    }
  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE void applyOperator (void) {

    galois::do_all_choice (makeLocalRange (this->getCurrWL()),
        [this] (Ctxt* c) {

          if (this->NEEDS_CUSTOM_LOCKING || c->isSrc()) {
            typename Base::UserCtxt& uhand = c->userHandle;

            bool commit = true;

            if (this->NEEDS_CUSTOM_LOCKING) {
              c->enableSrc ();
              runCatching (this->opFunc, c, uhand);
              commit = c->isSrc(); // in case opFunc signalled abort

            } else {
              this->opFunc (c->getActive(), uhand);
              commit = true;
            }

            if (!commit) {
              bool b = c->casState (ContextState::SCHEDULED, ContextState::ABORTING);
              assert (b);

              c->doAbort();

            } else {
              bool b = c->casState (ContextState::SCHEDULED, ContextState::READY_TO_COMMIT);
              assert (b);

              dbg::print (c, " completed operator, with active: ", c->getActive());

              this->commitQ.get().push_back (c);
              this->roundCommits += 1;

              if (this->NEEDS_PUSH) {
                auto& uhand = c->userHandle;
                for (auto i = uhand.getPushBuffer().begin()
                    , end_i = uhand.getPushBuffer().end(); i != end_i; ++i) {

                  updateCurrMinPending(*i);
                }
              }

              if (this->ENABLE_PARAMETER) {
                c->markExecRound (this->rounds);
              }
            }

          } else {

            if (c->casState (ContextState::SCHEDULED, ContextState::ABORTING)) {
              c->doAbort();

            } else {
              assert (c->hasState (ContextState::ABORTING) || c->hasState (ContextState::ABORT_DONE));
            }
          }
        },
        std::make_tuple (
          galois::loopname ("applyOperator"),
          galois::chunk_size<OpFunc::CHUNK_SIZE>()));

  }

  void tryCommit(Ctxt* c) {
    assert (c);
    assert (c->hasState(ContextState::READY_TO_COMMIT));

    if (c->casState(ContextState::READY_TO_COMMIT, ContextState::COMMITTING)) {
      if (this->NEEDS_PUSH) {
        auto& uhand = c->userHandle;
        for (auto i = uhand.getPushBuffer().begin()
            , end_i = uhand.getPushBuffer().end(); i != end_i; ++i) {

          dbg::print (c, " creating child ", *i);
          push_commit (*i);

        }
        uhand.getPushBuffer().clear();
      }

      c->doCommit();
      this->totalRetires += 1;

      if (this->ENABLE_PARAMETER) {
        assert (c->getExecRound() < this->execRcrds.size());
        this->execRcrds[c->getExecRound()].parallelism += 1;
      }

      this->freeCtxt(c);

    } // end if committing
  }

  void performCommits (void) {

    galois::optional<T> gvt = getMinPending();

    if (gvt) {
      dbg::print ("gvt computed as: ", *gvt);
    }
    // partition criteria
    auto ptest = [&gvt, this] (Ctxt* c) -> bool {
      assert (c);

      bool ret = false;
      if (c->hasState(ContextState::READY_TO_COMMIT)) {
        ret = true; // move to left 

        if (!gvt || this->getItemCmp()(c->getActive(), *gvt)) {
          ret = false; // can commit so move to right and deallocate

          dbg::print (c, " trying to commit with item ", c->getActive());
          
          tryCommit (c);
        } // end if gvt


      } else {
        ret = false; // an aborted or already committed task, so move to right
      }

      return ret;
    };


    galois::runtime::on_each_impl (
        [this, &ptest] (const unsigned tid, const unsigned numT) {
          auto& localQ = this->commitQ.get();

          auto new_end = std::partition (localQ.begin(), localQ.end(), ptest); 
          localQ.erase (new_end, localQ.end());
        });


  }

  // void performCommits (void) {
// 
    // auto revCtxtCmp = [this] (const Ctxt* a, const Ctxt* b) { return Base::ctxtCmp (b, a); };
// 
    // galois::runtime::on_each_impl (
        // [this, &revCtxtCmp] (const unsigned tid, const unsigned numT) {
          // auto& localQ = Base::commitQ.get();
          // auto new_end = std::partition (localQ.begin(), 
            // localQ.end(), 
            // [] (Ctxt* c) {
              // assert (c);
              // return c->hasState (ContextState::READY_TO_COMMIT);
            // });
// 
          // localQ.erase (new_end, localQ.end());
// 
          // std::sort (localQ.begin(), localQ.end(), revCtxtCmp);
        // });
// 
    // using C = typename Base::CommitQ::container_type;
// 
    // // assumes that per thread commit queues are sorted in reverse order
    // auto qcmp = [this] (const C* q1, const C* q2) -> bool {
// 
      // assert (q1 && !q1->empty());
      // assert (q2 && !q2->empty());
// 
      // return Base::ctxtCmp (q1->back(), q2->back());
    // };
     // 
// 
    // using PQ = galois::MinHeap<C*, typename std::remove_reference<decltype (qcmp)>::type>;
    // PQ commitMetaPQ (qcmp);
// 
    // for (unsigned i = 0; i < galois::getActiveThreads(); ++i) {
// 
      // if (!Base::commitQ.get (i).empty()) {
        // commitMetaPQ.push (&(Base::commitQ.get (i))); 
      // }
    // }
// 
    // Ctxt* minWinWL = Base::resetMinWinWL();
    // Ctxt* minPending = Base::getMinPending();
// 
// 
// 
    // while (!commitMetaPQ.empty()) { 
// 
      // C* q = commitMetaPQ.pop();
      // assert (!q->empty());
// 
      // bool e = commitMetaPQ.empty();
// 
      // bool exit = false;
// 
      // do {
        // Ctxt* c = q->back();
// 
        // if (!minPending || !Base::ctxtCmp (minPending, c)) { // minPending >= c
          // //do commit
          // q->pop_back();
// 
          // assert (c->hasState (ContextState::READY_TO_COMMIT) || c->hasState (ContextState::COMMIT_DONE));
// 
          // if (c->casState (ContextState::READY_TO_COMMIT, ContextState::COMMITTING)) {
// 
            // if (Base::NEEDS_PUSH) {
              // auto& uhand = c->userHandle;
              // for (auto i = uhand.getPushBuffer().begin()
                  // , end_i = uhand.getPushBuffer().end(); i != end_i; ++i) {
// 
                // Ctxt* child = Base::push_commit (*i, minWinWL);
// 
                // if (!minPending || Base::ctxtCmp (child, minPending)) { // update minPending
                  // minPending = child;
                // }
              // }
              // uhand.getPushBuffer().clear();
            // }
// 
            // c->doCommit();
            // Base::totalRetires += 1;
// 
            // if (Base::ENABLE_PARAMETER) {
              // assert (c->getExecRound() < Base::execRcrds.size());
              // Base::execRcrds[c->getExecRound()].parallelism += 1;
            // }
// 
// 
            // freeWL.push (c);
          // }
            // 
// 
        // } else {
          // exit = true;
          // break; // exit
        // }
      // } while (!q->empty() && !e && qcmp (q, commitMetaPQ.top()));
// 
      // if (exit) {
        // break;
      // }
// 
      // if (!q->empty()) {
        // commitMetaPQ.push (q);
      // }
    // } // end outer while
// 
// 
    // // memory is returned to owner thread, thus thread 0 doesn't accumulate all 
    // // the feed blocks
    // on_each_impl (
        // [this] (const unsigned tid, const unsigned numT) {
          // for (auto i = freeWL.get().begin()
              // , end_i = freeWL.get().end(); i != end_i; ++i) {
// 
            // Base::freeCtxt (*i);
          // }
          // freeWL.get().clear();
        // });
          // 
  // } // end performCommits

};


template <template <typename, typename, typename, typename, typename, typename> class Executor, typename R, typename Cmp, typename NhFunc, typename ExFunc, typename OpFunc, typename _ArgsTuple>
void for_each_ordered_spec_impl (const R& range, const Cmp& cmp, const NhFunc& nhFunc, const ExFunc& exFunc, const OpFunc& opFunc, const _ArgsTuple& argsTuple) {


  auto argsT = std::tuple_cat (argsTuple, 
      get_default_trait_values (argsTuple,
        std::make_tuple (loopname_tag {}, enable_parameter_tag {}),
        std::make_tuple (default_loopname {}, enable_parameter<false> {})));
  using ArgsT = decltype (argsT);

  using T = typename R::value_type;
  

  using Exec = Executor<T, Cmp, NhFunc, ExFunc, OpFunc, ArgsT>;
  
  Exec e (cmp, nhFunc, exFunc, opFunc, argsT);

  substrate::getThreadPool().burnPower (galois::getActiveThreads());

  e.push_initial (range);
  e.execute();

  substrate::getThreadPool().beKind();
}

template <typename R, typename Cmp, typename NhFunc, typename ExFunc, typename OpFunc, typename _ArgsTuple>
void for_each_ordered_optim (const R& range, const Cmp& cmp, const NhFunc& nhFunc, const ExFunc& exFunc, const OpFunc& opFunc, const _ArgsTuple& argsTuple) {

  for_each_ordered_spec_impl<OptimOrdExecutor> (range, cmp, nhFunc, exFunc, opFunc, argsTuple);
}

template <typename R, typename Cmp, typename NhFunc, typename OpFunc, typename _ArgsTuple>
void for_each_ordered_optim (const R& range, const Cmp& cmp, const NhFunc& nhFunc, const OpFunc& opFunc, const _ArgsTuple& argsTuple) {


  for_each_ordered_optim (range, cmp, nhFunc, internal::DummyExecFunc(), opFunc, argsTuple);
}

template <typename R, typename Cmp, typename NhFunc, typename ExFunc, typename OpFunc, typename _ArgsTuple>
void for_each_ordered_pessim (const R& range, const Cmp& cmp, const NhFunc& nhFunc, const ExFunc& exFunc, const OpFunc& opFunc, const _ArgsTuple& argsTuple) {

  for_each_ordered_spec_impl<PessimOrdExecutor> (range, cmp, nhFunc, exFunc, opFunc, argsTuple);
}

template <typename R, typename Cmp, typename NhFunc, typename OpFunc, typename _ArgsTuple>
void for_each_ordered_pessim (const R& range, const Cmp& cmp, const NhFunc& nhFunc, const OpFunc& opFunc, const _ArgsTuple& argsTuple) {

  for_each_ordered_pessim (range, cmp, nhFunc, internal::DummyExecFunc(), opFunc, argsTuple);
}


template <typename R, typename Cmp, typename NhFunc, typename ExFunc, typename OpFunc, typename _ArgsTuple>
void for_each_ordered_spec (const R& range, const Cmp& cmp, const NhFunc& nhFunc, const ExFunc& exFunc, const OpFunc& opFunc, const _ArgsTuple& argsTuple) {

  auto tplParam = std::tuple_cat (argsTuple, std::make_tuple (enable_parameter<true>()));
  auto tplNoParam = std::tuple_cat (argsTuple, std::make_tuple (enable_parameter<false>()));

  switch (specMode) {
    case SpecMode::OPTIM: {
      if (useParaMeterOpt) {
        for_each_ordered_spec_impl<OptimOrdExecutor> (range, cmp, nhFunc, exFunc, opFunc, tplParam);
      } else {
        for_each_ordered_spec_impl<OptimOrdExecutor> (range, cmp, nhFunc, exFunc, opFunc, tplNoParam);
      }
      break;
    }

    case SpecMode::PESSIM: {
      if (useParaMeterOpt) {
        for_each_ordered_spec_impl<PessimOrdExecutor> (range, cmp, nhFunc, exFunc, opFunc, tplParam);
      } else {
        for_each_ordered_spec_impl<PessimOrdExecutor> (range, cmp, nhFunc, exFunc, opFunc, tplNoParam);
      }
      break;
    }
    default:
      std::abort();
  }
}

template <typename R, typename Cmp, typename NhFunc, typename OpFunc, typename _ArgsTuple>
void for_each_ordered_spec (const R& range, const Cmp& cmp, const NhFunc& nhFunc, const OpFunc& opFunc, const _ArgsTuple& argsTuple) {

  for_each_ordered_spec (range, cmp, nhFunc, internal::DummyExecFunc(), opFunc, argsTuple);
}



} // end namespace runtime
} // end namespace galois


#endif // GALOIS_RUNTIME_ORDERED_SPECULATION_H
