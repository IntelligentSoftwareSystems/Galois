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
#include "Galois/Runtime/KDGtwoPhaseSupport.h"
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
    onWL (false)
  {}


  bool hasState (const ContextState& s) const { return state == s; } 

  void setState (const ContextState& s) { state = s; } 

  bool casState (ContextState s_old, const ContextState& s_new) { 
    // return state.cas (s_old, s_new);
    return state.compare_exchange_strong (s_old, s_new);
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

    userHandle.commit ();

    // for (auto i = userHandle.getPushBuffer ().begin (), 
        // end_i = userHandle.getPushBuffer ().end (); i != end_i; ++i) {
// 
      // OptimContext* child = exec.push (*i);
      // dbg::debug (this, " creating child ", child);
// 
      // children.push_back (child);
    // }
  }

  void addChild (OptimContext* child) {

    assert (std::find (children.begin (), children.end (), child) == children.end ());

    dbg::debug (this, " creating child ", child);

    children.push_back (child);

  }

  void doCommit () {

    assert (hasState (ContextState::COMMITTING));

    dbg::debug (this, " committing with item ", this->getActive ());
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

    dbg::debug (this, " aborting with item ", this->getActive ());

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
  void findCommitSrc (WL& wl) const {

    for (const NItem* ni: nhood) {

      OptimContext* c = ni->getHistHead ();
      assert (c != this);

      if (c && c->isCommitSrc () && c->onWL.cas (false, true)) {
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
      ni->resetMin (this);
    }
  }

  void addToHistory (void) {

    for (NItem* ni: nhood) {
      ni->addToHistory (this);
    }
  }

};

template <typename T, typename Cmp, typename NhFunc, typename ExFunc, typename  OpFunc, bool HAS_EXEC_FUNC> 
class OptimOrdExecutor: private boost::noncopyable {

  friend struct OptimContext<T, Cmp, OptimOrdExecutor>;
  using Ctxt = OptimContext<T, Cmp, OptimOrdExecutor>;
  using NhoodMgr = typename Ctxt::NhoodMgr;
  using CtxtCmp = typename Ctxt::CtxtCmp;
  using NItemFactory = typename Ctxt::NItem::Factory;

  static const bool ADDS = DEPRECATED::ForEachTraits<OpFunc>::NeedsPush;
  using WindowWL = typename std::conditional<ADDS, PQbasedWindowWL<Ctxt*, CtxtCmp>, SortedRangeWindowWL<Ctxt*, CtxtCmp> >::type;

  using CommitQ = Galois::PerThreadVector<Ctxt*>;

  using CtxtWL = PerThreadBag<Ctxt*>;
  using CtxtAlloc = Runtime::FixedSizeAllocator<Ctxt>;

  using UserCtxt = UserContextAccess<T>;
  using PerThreadUserCtxt = Substrate::PerThreadStorage<UserCtxt>;

  static const unsigned DEFAULT_CHUNK_SIZE = 8;

  struct CtxtMaker {
    OptimOrdExecutor& outer;

    Ctxt* operator () (const T& x) {

      Ctxt* ctxt = outer.ctxtAlloc.allocate (1);
      assert (ctxt);
      outer.ctxtAlloc.construct (ctxt, x, ContextState::UNSCHEDULED, outer);

      return ctxt;
    }
  };


  Cmp itemCmp;
  NhFunc nhFunc;
  ExFunc execFunc;
  OpFunc opFunc;
  const char* loopname;

  CtxtCmp ctxtCmp;
  NItemFactory nitemFactory;
  NhoodMgr nhmgr;
  WindowWL winWL;
  std::unique_ptr<CtxtWL> currWL;
  std::unique_ptr<CtxtWL> nextWL;
  CtxtMaker ctxtMaker;


  CtxtAlloc ctxtAlloc;

  size_t windowSize;
  size_t rounds;
  size_t prevSources;
  double targetCommitRatio;

  CommitQ commitQ;
  // PerThreadUserCtxt userHandles;
  GAccumulator<size_t> numSources;
  GAccumulator<size_t> numCommitted;
  GAccumulator<size_t> total;

public:
  OptimOrdExecutor (const Cmp& cmp, const NhFunc& nhFunc, const ExFunc& execFunc, const OpFunc& opFunc, const char* loopname)
    : 
      itemCmp (cmp), 
      nhFunc (nhFunc), 
      execFunc (execFunc),
      opFunc (opFunc), 
      loopname (loopname),
      ctxtCmp (itemCmp),
      nitemFactory (ctxtCmp),
      nhmgr (nitemFactory),
      winWL (ctxtCmp),
      currWL (new CtxtWL),
      nextWL (new CtxtWL),
      ctxtMaker {*this},
      windowSize (0),
      rounds (0),
      prevSources (0),
      targetCommitRatio (commitRatioArg)
  {
    if (!loopname) { loopname = "NULL"; }

    if (targetCommitRatio < 0.0) {
      targetCommitRatio = 0.0;
    }
    if (targetCommitRatio > 1.0) {
      targetCommitRatio = 1.0;
    }

  }

  ~OptimOrdExecutor (void) {}

  const Cmp& getItemCmp () const { return itemCmp; }

  const CtxtCmp& getCtxtCmp () const { return ctxtCmp; }

  // on each thread
  template <typename R>
  void push_initial (const R& range) {

    Galois::do_all_choice (range,
        [this] (const T& x) {

          Ctxt* c = ctxtMaker (x);
          nextWL->push (c);

        }, 
        "init-fill",
        chunk_size<DEFAULT_CHUNK_SIZE> ());

    if (targetCommitRatio != 0.0) {

      winWL.initfill (makeLocalRange (*nextWL));
      nextWL->clear_all_parallel ();
    }

  }

  void operator () (void) {
    execute ();
  }
  
  void execute () {

    rounds = 0;

    while (true) {

      prepareRound ();

      if (currWL->empty_all ()) {
        break;
      }

      // std::printf ("Round: %d, currWL size = %zd, commitQ size = %zd\n",
          // rounds, currWL->size_all (), commitQ.size_all ());

      expandNhood ();

      CtxtWL sources;

      serviceAborts (sources);

      executeSources (sources);

      applyOperator (sources);

      performCommits ();

      reclaimMemory (sources);

    }

    printStats ();
  }

private:

  void printStats (void) {
    std::printf ("OptimOrdExecutor, rounds: %zu\n", rounds);
    std::printf ("OptimOrdExecutor, commits: %zu\n", numCommitted.reduce ());
    std::printf ("OptimOrdExecutor, total: %zu\n", total.reduce ());
    std::printf ("OptimOrdExecutor, efficiency: %g\n", double (numCommitted.reduce ()) / total.reduce ());
    std::printf ("OptimOrdExecutor, avg. parallelism: %g\n", double (numCommitted.reduce ()) / rounds);
  }


  Ctxt* push (const T& x) {
    
    Ctxt* ctxt = ctxtMaker (x);

    if (targetCommitRatio == 0.0) {

      nextWL->push (ctxt);

    } else {
      winWL.push (ctxt);
          
    }

    return ctxt;
  }

  void push_abort (Ctxt* ctxt) {
    assert (ctxt);
    assert (ctxt->hasState (ContextState::ABORT_DONE));

    ctxt->setState (ContextState::UNSCHEDULED);
    nextWL->push (ctxt);
  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE void spillAll (CtxtWL& wl) {
    assert (targetCommitRatio != 0.0);
    on_each(
        [this, &wl] (const unsigned tid, const unsigned numT) {
          while (!wl[tid].empty ()) {
            Ctxt* c = wl[tid].back ();
            wl[tid].pop_back ();

            winWL.push (c);
          }
        });

    assert (wl.empty_all ());
    assert (!winWL.empty ());
  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE void refill (CtxtWL& wl, size_t currCommits, size_t prevWindowSize) {

    assert (targetCommitRatio != 0.0);

    const size_t INIT_MAX_ROUNDS = 500;
    const size_t THREAD_MULT_FACTOR = 16;
    const double TARGET_COMMIT_RATIO = targetCommitRatio;
    const size_t MIN_WIN_SIZE = OpFunc::CHUNK_SIZE * getActiveThreads ();
    // const size_t MIN_WIN_SIZE = 2000000; // OpFunc::CHUNK_SIZE * getActiveThreads ();
    const size_t WIN_OVER_SIZE_FACTOR = 8;

    if (prevWindowSize == 0) {
      assert (currCommits == 0);

      // initial settings
      if (DEPRECATED::ForEachTraits<OpFunc>::NeedsPush) {
        windowSize = std::max (
            (winWL.initSize ()),
            (THREAD_MULT_FACTOR * MIN_WIN_SIZE));

      } else {
        windowSize = std::min (
            (winWL.initSize () / INIT_MAX_ROUNDS),
            (THREAD_MULT_FACTOR * MIN_WIN_SIZE));
      }
    } else {

      assert (windowSize > 0);

      double commitRatio = double (currCommits) / double (prevWindowSize);

      if (commitRatio >= TARGET_COMMIT_RATIO) {
        windowSize *= 2;
        // windowSize = int (windowSize * commitRatio/TARGET_COMMIT_RATIO); 
        // windowSize = windowSize + windowSize / 2;

      } else {
        windowSize = int (windowSize * commitRatio/TARGET_COMMIT_RATIO); 

        // if (commitRatio / TARGET_COMMIT_RATIO < 0.90) {
          // windowSize = windowSize - (windowSize / 10);
// 
        // } else {
          // windowSize = int (windowSize * commitRatio/TARGET_COMMIT_RATIO); 
        // }
      }
    }

    if (windowSize < MIN_WIN_SIZE) { 
      windowSize = MIN_WIN_SIZE;
    }

    assert (windowSize > 0);


    if (DEPRECATED::ForEachTraits<OpFunc>::NeedsPush) {
      if (winWL.empty () && (wl.size_all () > windowSize)) {
        // a case where winWL is empty and all the new elements were going into 
        // nextWL. When nextWL becomes bigger than windowSize, we must do something
        // to control efficiency. One solution is to spill all elements into winWL
        // and refill
        //

        spillAll (wl);

      } else if (wl.size_all () > (WIN_OVER_SIZE_FACTOR * windowSize)) {
        // too many adds. spill to control efficiency
        spillAll (wl);
      }
    }

    auto noop = [] (Ctxt* const c) { return c; };

    winWL.poll (wl, windowSize, wl.size_all (), noop);
    // std::cout << "Calculated Window size: " << windowSize << ", Actual: " << wl->size_all () << std::endl;
  }

  // TODO: refactor prepareRound, refill, spillAll into a 
  // common code-base 

  GALOIS_ATTRIBUTE_PROF_NOINLINE void prepareRound (void) {
    ++rounds;
    std::swap (currWL, nextWL);

    if (targetCommitRatio != 0.0) {
      size_t currCommits = numSources.reduce () - prevSources;
      prevSources += currCommits;

      size_t prevWindowSize = nextWL->size_all ();
      refill (*currWL, currCommits, prevWindowSize);
    }

    nextWL->clear_all_parallel ();
  }



  GALOIS_ATTRIBUTE_PROF_NOINLINE void expandNhood (void) {
    Galois::do_all_choice (makeLocalRange (*currWL),
        [this] (Ctxt* c) {

          if (!c->hasState (ContextState::ABORTED_CHILD)) {

            assert (!c->hasState (ContextState::RECLAIM));
            c->schedule ();

            dbg::debug ("scheduling: ", c, " with item: ", c->getActive ());

            UserCtxt& uhand = c->userHandle;

            // nhFunc (c, uhand);
            runCatching (nhFunc, c, uhand);

            total += 1;
          }
        },
        "expandNhood",
        chunk_size<NhFunc::CHUNK_SIZE> ());

  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE void executeSources (CtxtWL& sources) {

    if (HAS_EXEC_FUNC) {

      Galois::do_all_choice (makeLocalRange (sources),
        [this] (Ctxt* ctxt) {
          assert (ctxt->isSrc ());
          assert (!ctxt->hasState (ContextState::RECLAIM));
          assert (!ctxt->hasState (ContextState::ABORTED_CHILD));

          execFunc (ctxt->getActive ());
        },
        "exec-func",
        Galois::chunk_size<ExFunc::CHUNK_SIZE> ());

    }
  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE void applyOperator (CtxtWL& sources) {
    Ctxt* minElem = nullptr;

    if (DEPRECATED::ForEachTraits<OpFunc>::NeedsPush) {
      if (targetCommitRatio != 0.0 && !winWL.empty ()) {
        minElem = *winWL.getMin();
      }
    }

    Galois::do_all_choice (makeLocalRange (sources),
        [this, minElem] (Ctxt* c) {

          UserCtxt& uhand = c->userHandle;

          assert (c->isSrc ());
          assert (!c->hasState (ContextState::RECLAIM));
          assert (!c->hasState (ContextState::ABORTED_CHILD));

          runCatching (opFunc, c, uhand);
          bool commit = c->isSrc (); // in case opFunc signalled abort

          if (commit) {


            if (DEPRECATED::ForEachTraits<OpFunc>::NeedsPush) { 

              for (auto i = uhand.getPushBuffer ().begin ()
                  , endi = uhand.getPushBuffer ().end (); i != endi; ++i) {

                Ctxt* child = ctxtMaker (*i);
                c->addChild (child);

                if (!minElem || !ctxtCmp (minElem, child)) {
                  // if *i >= *minElem
                  nextWL->push_back (child);
                } else {
                  winWL.push (child);
                } 
              }
            } else {

              assert (uhand.getPushBuffer ().begin () == uhand.getPushBuffer ().end ());
            }

            bool b = c->casState (ContextState::SCHEDULED, ContextState::READY_TO_COMMIT);

            assert (b && "CAS shouldn't have failed");
            numSources += 1;

            c->publishChanges ();
            c->addToHistory ();
            commitQ.get ().push_back (c);

          } else {

            if (c->casState (ContextState::SCHEDULED, ContextState::ABORTING)) {
              c->doAbort ();

            } else {
              assert (c->hasState (ContextState::ABORTING) || c->hasState (ContextState::ABORT_DONE));
            }
          }
        },
        "applyOperator",
        chunk_size<OpFunc::CHUNK_SIZE> ());

  }

  Ctxt* computeGVT (void) {

    Substrate::PerThreadStorage<Ctxt*> perThrdMin;

    on_each_impl ([this, &perThrdMin] (const unsigned tid, const unsigned numT) {
          
          for (auto i = nextWL->local_begin ()
            , end_i = nextWL->local_end (); i != end_i; ++i) {

            Ctxt*& lm = *(perThrdMin.getLocal ());

            if (!lm || ctxtCmp (*i, lm)) {
              lm = *i;
            }
          }

          
        });

    Ctxt* ret = nullptr;

    for (unsigned i = 0; i < perThrdMin.size (); ++i) {

      Ctxt* lm = *(perThrdMin.getRemote (i));

      if (lm) {
        if (!ret || ctxtCmp (lm, ret)) {
          ret = lm;
        }
      }
    }

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
    
    CtxtWL abortWL;

    Galois::do_all_choice (makeLocalRange (*currWL),
        [this, &abortWL] (Ctxt* c) {

          if (c->isSrc ()) {

            if (c->findAborts (abortWL)) {
              // XXX: c does not need to abort if it's neighborhood
              // isn't dependent on values computed by other tasks

              c->disableSrc ();
            }

          } 
        },
        "mark-aborts",
        Galois::chunk_size<DEFAULT_CHUNK_SIZE> ());


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
    

    Galois::do_all_choice (makeLocalRange (*currWL),

        [this, &sources] (Ctxt* c) {
          if (c->isSrc () && !c->hasState (ContextState::ABORTED_CHILD)) {
            assert (c->hasState (ContextState::SCHEDULED));

            c->resetMarks ();
            sources.push (c);

          } else if (c->hasState (ContextState::ABORTED_CHILD)) {
            commitQ.get ().push_back (c); // for reclaiming memory 

          } else {
            assert (!c->hasState (ContextState::ABORTED_CHILD));
            quickAbort (c);
          }
        },

        "collect-sources",
        Galois::chunk_size<DEFAULT_CHUNK_SIZE> ());


  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE void performCommits () {

    CtxtWL commitSources;

    Ctxt* gvt = computeGVT ();

    Galois::do_all_choice (makeLocalRange (commitQ),
        [this, gvt, &commitSources] (Ctxt* c) {

          assert (c);

          if (c->hasState (ContextState::READY_TO_COMMIT) 
              && (!gvt || ctxtCmp (c, gvt))
              && c->isCommitSrc ()) {

            commitSources.push (c);
          }
        },
        "find-commit-srcs",
        Galois::chunk_size<DEFAULT_CHUNK_SIZE> ());
        

    Galois::Runtime::for_each_gen (
        makeLocalRange (commitSources),
        [this] (Ctxt* c, UserContext<Ctxt*>& wlHandle) {

          bool b = c->casState (ContextState::READY_TO_COMMIT, ContextState::COMMITTING);

          if (b) {

            assert (c->isCommitSrc ());
            c->doCommit ();
            c->findCommitSrc (wlHandle);
            numCommitted += 1;

          } else {
            assert (c->hasState (ContextState::COMMIT_DONE));
          }
        },
        std::make_tuple (
          Galois::loopname ("retire"),
          Galois::does_not_need_aborts_tag (),
          Galois::wl<Galois::WorkList::dChunkedFIFO<DEFAULT_CHUNK_SIZE> > ()));



  }

  void freeCtxt (Ctxt* ctxt) {
    ctxtAlloc.destroy (ctxt);
    ctxtAlloc.deallocate (ctxt, 1);
  }

  void reclaimMemory (CtxtWL& sources) {

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
              dbg::debug ("Ctxt destroyed from commitQ: ", *i);
              freeCtxt (*i);
            }
          }

          localQ.erase (new_end, localQ.end ());
        });


  }




};


namespace HIDDEN {
  
  struct DummyExecFunc {
    static const unsigned CHUNK_SIZE = 1;
    template <typename T>
    void operator () (const T&) const {}
  };
}

template <typename R, typename Cmp, typename NhFunc, typename ExFunc, typename OpFunc>
void for_each_ordered_optim (const R& range, const Cmp& cmp, const NhFunc& nhFunc, const ExFunc& execFunc, const OpFunc& opFunc, const char* loopname=0) {

  using T = typename R::value_type;



  const bool HAS_EXEC_FUNC = std::is_same<ExFunc, HIDDEN::DummyExecFunc>::value;

  using Exec = OptimOrdExecutor<T, Cmp, NhFunc, ExFunc, OpFunc, HAS_EXEC_FUNC>;
  
  Exec e (cmp, nhFunc, execFunc, opFunc, loopname);

  Substrate::getThreadPool().burnPower (Galois::getActiveThreads ());

  e.push_initial (range);
  e.execute ();

  Substrate::getThreadPool().beKind();
}

template <typename R, typename Cmp, typename NhFunc, typename OpFunc>
void for_each_ordered_optim (const R& range, const Cmp& cmp, const NhFunc& nhFunc, const OpFunc& opFunc, const char* loopname=0) {


  for_each_ordered_optim (range, cmp, nhFunc, HIDDEN::DummyExecFunc (), opFunc, loopname);
}


namespace ParaMeter {

template <typename Ctxt, typename CtxtCmp>
struct OptimParamNhoodItem: public OrdLocBase<OptimParamNhoodItem<Ctxt, CtxtCmp>, Ctxt, CtxtCmp> {

  using Base = OrdLocBase<OptimParamNhoodItem, Ctxt, CtxtCmp>;
  using Factory = OrdLocFactoryBase<OptimParamNhoodItem, Ctxt, CtxtCmp>;

  using Sharers = std::list<Ctxt*, Galois::Runtime::FixedSizeAllocator<Ctxt*> >;


  Sharers sharers;
  const CtxtCmp& ctxtCmp;


  OptimParamNhoodItem (Lockable* l, const CtxtCmp& ctxtCmp): Base (l), ctxtCmp (ctxtCmp) {}

  bool add (Ctxt* ctxt) {

    if (sharers.empty ()) { // empty 
      sharers.push_back (ctxt);
      return true;

    } else {

      while (!sharers.empty ()) {
        Ctxt* tail = sharers.back ();
        assert (tail);


        if (ctxtCmp (ctxt, tail)) { // ctxt < tail
          // tail should not be running
          assert (tail->hasState (ContextState::READY_TO_COMMIT));
          dbg::debug (ctxt, " aborting lower priority sharer ", tail);
          tail->doAbort ();

        } else { // ctxt >= tail

          assert (ctxt->step >= tail->step);
          if (ctxt->step == tail->step) {
            assert (!ctxtCmp (ctxt, tail)); // ctxt >= tail
            dbg::debug (ctxt, " lost conflict to  ", tail);
            return false;
          }
          else {
            break;
          }
        }
      }

      if (!sharers.empty ()) { 
        assert (ctxt->step >= sharers.back ()->step);
        assert (!ctxtCmp (ctxt, sharers.back ())); // ctxt >= tail
      }

      sharers.push_back (ctxt);
      return true;
    } // end else
  }

  void removeAbort (Ctxt* ctxt) {
    assert (std::find (sharers.begin (), sharers.end (), ctxt) != sharers.end ());

    bool found = false;

    while (!sharers.empty ()) {
      Ctxt* tail = sharers.back ();
      assert (tail);

      if (ctxt == tail) {
        sharers.pop_back (); 
        found = true;
        break;
      }

      if (ctxtCmp (ctxt, tail)) { // ctxt < tail
        dbg::debug (ctxt, " removing self & aborting lower priority sharer ", tail);
        tail->doAbort ();
      } else {
        assert (!found);
        GALOIS_DIE ("shouldn't reach here");
      }
    }

    assert (found);
    assert (std::find (sharers.begin (), sharers.end (), ctxt) == sharers.end ());
  }

  void removeCommit (Ctxt* ctxt) {
    assert (std::find (sharers.begin (), sharers.end (), ctxt) != sharers.end ());
    assert (!sharers.empty ());
    assert (sharers.front () == ctxt);
    sharers.pop_front ();

    assert (std::find (sharers.begin (), sharers.end (), ctxt) == sharers.end ());
  }

};

template <typename T, typename Cmp, typename Exec>
struct OptimParamContext: public OrderedContextBase<T> {

  using Base = OrderedContextBase<T>;

  using CtxtCmp = ContextComparator<OptimParamContext, Cmp>;
  using NItem = OptimParamNhoodItem<OptimParamContext, CtxtCmp>;
  using NhoodMgr = PtrBasedNhoodMgr<NItem>;
  using NhoodList = typename gstl::Vector<NItem*>;
  using ChildList = typename gstl::Vector<OptimParamContext*>;

  ContextState state;
  size_t step;
  Exec& exec;
  NhoodList nhood;
  bool abortSelf = false;

  // TODO: avoid using UserContextAccess and per-iteration allocator
  // use Pow of 2 block allocator instead. 
  UserContextAccess<T> userHandle;
  ChildList children;


  explicit OptimParamContext (const T& x, const ContextState& s, size_t step, Exec& exec)
  :
    Base (x), state (s), step (step), exec (exec) 
  {}

  GALOIS_ATTRIBUTE_PROF_NOINLINE
  virtual void subAcquire (Lockable* l, Galois::MethodFlag m) {

    NItem& nitem = exec.nhmgr.getNhoodItem (l);
    assert (NItem::getOwner (l) == &nitem);

    if (!abortSelf && std::find (nhood.begin (), nhood.end (), &nitem) == nhood.end ()) {

      bool succ = nitem.add (this);
      if (succ) {
        nhood.push_back (&nitem);

      } else {
        abortSelf = true;
        setState (ContextState::ABORT_SELF);
      }
    }
  }

  void schedule (const size_t step) {
    setState (ContextState::SCHEDULED); 
    this->step = step;
    nhood.clear ();
    userHandle.reset ();
    children.clear ();
    abortSelf = false;
  }

  void publishChanges (void) {

    userHandle.commit ();
    for (auto i = userHandle.getPushBuffer ().begin (), 
        end_i = userHandle.getPushBuffer ().end (); i != end_i; ++i) {

      OptimParamContext* child = exec.push (*i);
      dbg::debug (this, " creating child ", child);

      children.push_back (child);
    }
  }

  void doCommit () {
    assert (state == ContextState::READY_TO_COMMIT);

    dbg::debug (this, " committing with item ", this->getActive ());
    for (NItem* n: nhood) {
      n->removeCommit (this);
    }

    setState (ContextState::COMMIT_DONE);
  }

  bool hasState (const ContextState& s) const { return state == s; } 

  void setState (const ContextState& s) { state = s; } 

  void doAbort (bool addBack=true) {
    // this can be in states READY_TO_COMMIT, ABORT_SELF
    // children can be in UNSCHEDULED, READY_TO_COMMIT, ABORT_DONE

    // first abort all the children recursively
    // then abort self.
    //
    bool b = hasState (ContextState::ABORT_SELF) || hasState (ContextState::READY_TO_COMMIT);
    assert (b);

    for (OptimParamContext* child: children) {

      assert (!child->hasState (ContextState::SCHEDULED));
      assert (!child->hasState (ContextState::ABORTED_CHILD));

      bool c = child->hasState (ContextState::UNSCHEDULED)
        || child->hasState (ContextState::READY_TO_COMMIT)
        || child->hasState (ContextState::ABORT_DONE);
      assert (c);

      if (child->hasState (ContextState::READY_TO_COMMIT)) {
        dbg::debug (this, " aborting child ", child);
        child->doAbort (false);
        // exec.freeCtxt (child); // TODO: free memory at the right point
      } else {
        child->setState (ContextState::ABORTED_CHILD);
      }
    }

    dbg::debug (this, " aborting with item ", this->getActive ());

    userHandle.rollback ();

    for (NItem* n: nhood) {
      n->removeAbort (this);
    }

    if (addBack) {
      setState (ContextState::ABORT_DONE);
      exec.push_abort (this);
    }

  }

};


template <typename T, typename Cmp, typename NhFunc, typename OpFunc> 
class OptimParaMeterExecutor: private boost::noncopyable {

  friend class OptimParamContext<T, Cmp, OptimParaMeterExecutor>;
  using Ctxt = OptimParamContext<T, Cmp, OptimParaMeterExecutor>;
  using NhoodMgr = typename Ctxt::NhoodMgr;
  using CtxtCmp = typename Ctxt::CtxtCmp;
  using NItemFactory = typename Ctxt::NItem::Factory;

  using CommitQ = std::vector<Ctxt*>;
  using PendingQ = Galois::MinHeap<Ctxt*, CtxtCmp>;

  using CtxtAlloc = Runtime::FixedSizeAllocator<Ctxt>;
  using ExecutionRecords = std::vector<StepStats>;

  Cmp itemCmp;
  NhFunc nhFunc;
  OpFunc opFunc;
  const char* loopname;

  CtxtCmp ctxtCmp;
  NItemFactory nitemFactory;
  NhoodMgr nhmgr;
  CommitQ rob;

  PendingQ* currPending;
  PendingQ* nextPending;
  
  CtxtAlloc ctxtAlloc;
  ExecutionRecords execRcrds;
  size_t steps = 0;


public:
  OptimParaMeterExecutor (const Cmp& cmp, const NhFunc& nhFunc, const OpFunc& opFunc, const char* loopname)
    : 
      itemCmp (cmp), 
      nhFunc (nhFunc), 
      opFunc (opFunc), 
      loopname (loopname),
      ctxtCmp (itemCmp),
      nitemFactory (ctxtCmp),
      nhmgr (nitemFactory)
  {
    currPending = new PendingQ (ctxtCmp);
    nextPending = new PendingQ (ctxtCmp);
    steps = 0;

    if (!loopname) { loopname = "NULL"; }
  }

  ~OptimParaMeterExecutor (void) {
    currPending->clear ();
    nextPending->clear ();
    delete currPending; currPending = nullptr;
    delete nextPending; nextPending = nullptr;
  }

  const Cmp& getItemCmp () const { return itemCmp; }

  const CtxtCmp& getCtxtCmp () const { return ctxtCmp; }

  template <typename R>
  void push_initial (const R& range) {
    push (range.begin (), range.end ());
  }

  template <typename Iter>
  void push (Iter beg, Iter end) {
    for (Iter i = beg; i != end; ++i) {
      push (*i);
    }
  }

  Ctxt*  push (const T& x) {
    Ctxt* ctxt = ctxtAlloc.allocate (1);
    assert (ctxt);
    assert (steps >= 0);
    ctxtAlloc.construct (ctxt, x, ContextState::UNSCHEDULED, steps, *this);
    nextPending->push (ctxt);

    return ctxt;
  }

  void push_abort (Ctxt* ctxt) {
    assert (ctxt);
    assert (ctxt->hasState (ContextState::ABORT_DONE));

    ctxt->setState (ContextState::UNSCHEDULED);
    nextPending->push (ctxt);
  }

  void execute () {

    size_t totalIter = 0;
    size_t totalCommits = 0;

    while (!nextPending->empty () || !rob.empty ()) {

      std::swap (currPending, nextPending);
      nextPending->clear ();
      execRcrds.emplace_back (steps, currPending->size ()); // create record entry for current step;
      ++steps;
      assert (execRcrds.size () == steps);

      while (!currPending->empty ()) {
        Ctxt* ctxt = schedule ();

        if (!ctxt) {
          assert (currPending->empty ());
          break;
        }

        dbg::debug (ctxt, " scheduled with item ", ctxt->getActive ());

        ++totalIter;

        Galois::Runtime::setThreadContext (ctxt);
        nhFunc (ctxt->getActive (), ctxt->userHandle);

        if (ctxt->hasState (ContextState::SCHEDULED)) {
          opFunc (ctxt->getActive (), ctxt->userHandle);
        }
        Galois::Runtime::setThreadContext (nullptr);

        if (ctxt->hasState (ContextState::SCHEDULED)) {
          ctxt->setState (ContextState::READY_TO_COMMIT);

          // publish remaining changes
          ctxt->publishChanges ();

          dbg::debug (ctxt, " adding self to rob");
          rob.push_back (ctxt);
          assert (std::find (rob.begin (), rob.end (), ctxt) != rob.end ());

        } else {
          assert (ctxt->hasState (ContextState::ABORT_SELF));
          dbg::debug (ctxt, " aborting self");
          ctxt->doAbort ();
        }
      }

      size_t numCommitted = clearROB ();
      totalCommits += numCommitted;

      if (numCommitted == 0) {
        dbg::debug ("head of rob: ", rob.back (),  "  with item: ", rob.back ()->getActive ());

        dbg::debug ("head of nextPending: ", nextPending->top (),  "  with item: ", nextPending->top ()->getActive ());
      }
      assert (numCommitted > 0);


    }

    std::printf ("OptimParaMeterExecutor: steps=%zd, totalIter=%zd, totalCommits=%zd, avg=%f\n", steps, totalIter, totalCommits, float(totalCommits)/float(steps));

    finish ();
  }
  
private:

  void freeCtxt (Ctxt* ctxt) {
    ctxtAlloc.destroy (ctxt);
    ctxtAlloc.deallocate (ctxt, 1);
  }

  void finish (void) {
    for (const StepStats& s: execRcrds) {
      //FIXME:      s.dump (getStatsFile (), loopname);
    }

    //FIXME: closeStatsFile ();
  }

  Ctxt* schedule () {

    assert (!currPending->empty ());

    while (!currPending->empty ()) {
      Ctxt* ctxt = currPending->pop ();

      bool b = ctxt->hasState (ContextState::UNSCHEDULED) 
        || ctxt->hasState (ContextState::ABORTED_CHILD);
      assert (b);

      if (ctxt->hasState (ContextState::UNSCHEDULED)) {
        assert (steps > 0);
        ctxt->schedule ((steps - 1));
        return ctxt;

      } else {
        assert (ctxt->hasState (ContextState::ABORTED_CHILD));
        dbg::debug ("deleting aborted child: ", ctxt, " with item ", ctxt->getActive ());
        freeCtxt (ctxt);

      }
    }

    return nullptr;
  }

  size_t clearROB (void) {
    // first remove all tasks that are not in READY_TO_COMMIT
    auto new_end = std::partition (rob.begin (), rob.end (), 
        [] (Ctxt* c) { 
          assert (c);
          return c->hasState (ContextState::READY_TO_COMMIT);
        });


    rob.erase (new_end, rob.end ());

    // now sort in reverse order
    auto revCmp = [this] (Ctxt* a, Ctxt* b) { 
      assert (a);
      assert (b);
      return !ctxtCmp (a, b);
    };

    std::sort (rob.begin (), rob.end (), revCmp);

    // for debugging only, there should be no duplicates
    auto uniq_end = std::unique (rob.begin (), rob.end ());
    assert (uniq_end == rob.end ());

    size_t numCommitted = 0;

    while (!rob.empty ()) {
      Ctxt* head = rob.back ();

      assert (head->hasState (ContextState::READY_TO_COMMIT));

      dbg::debug ("head of rob ready to commit : ", head);
      bool earliest = false;
      if (!nextPending->empty ()) {
        earliest = !ctxtCmp (nextPending->top (), head);

      } else {
        earliest = true;
      }

      if (earliest) {

        head->doCommit ();
        rob.pop_back ();

        const size_t s = head->step;
        assert (s < execRcrds.size ());
        ++execRcrds[s].parallelism;
        numCommitted += 1;
        freeCtxt (head);


      } else {
        dbg::debug ("head of rob could not commit : ", head);
        break;
      }

    }

    return numCommitted;

  }

};



} // end namespace ParaMeter




template <typename R, typename Cmp, typename NhFunc, typename OpFunc>
void for_each_ordered_optim_param (const R& range, Cmp cmp, NhFunc nhFunc, OpFunc opFunc, const char* loopname=0) {

  using T = typename R::value_type;

  ParaMeter::OptimParaMeterExecutor<T, Cmp, NhFunc, OpFunc> exec (cmp, nhFunc, opFunc, loopname);

  exec.push_initial (range);
  exec.execute ();
}




} // end namespace Runtime
} // end namespace Galois


#endif // GALOIS_RUNTIME_ORDERED_SPECULATION_H
