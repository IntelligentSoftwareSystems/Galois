/** ?? -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a gramework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
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

#include "Galois/Runtime/Context.h"
#include "Galois/Runtime/OrderedLockable.h"

namespace Galois {
namespace Runtime {

enum class ContextState: int {
    UNSCHEDULED,
    SCHEDULED,
    READY_TO_COMMIT,
    ABORT_SELF,
    ABORT_HELP,
    ABORTING,
    COMMITTING, 
    COMMIT_DONE,
    ABORT_DONE,
    ABORTED_CHILD,
};


template <typename Ctxt, typename CtxtCmp>
struct OptimNhoodItem: public OrdLocBase<OptimNhoodItem<Ctxt, CtxtCmp>, Ctxt, CtxtCmp> {

  using Base = OrdLocBase<OptimNhoodItem, Ctxt, CtxtCmp>;
  using Factory = OrdLocFactoryBase<OptimNhoodItem, Ctxt, CtxtCmp>;

  using Sharers = Galois::gstl::List<Ctxt*>;
  using Lock_ty = Galois::Substrate::SimpleLock;


  const CtxtCmp& ctxtCmp;
  Sharers sharers;
  Lock_ty mutex;


  OptimNhoodItem (Lockable* l, const CtxtCmp& ctxtCmp): 
    Base (l), 
    ctxtCmp (ctxtCmp) {}

  bool add (Ctxt* ctxt) {

    mutex.lock ();
    if (sharers.empty ()) {
      sharers.push_back (ctxt);
      mutex.unlock ();
      return true;

    } else {

      while (!sharers.empty ()) {
        Ctxt* tail = sharers.back ();
        assert (tail);
        
        if (ctxtCmp (ctxt, tail)) { // ctxt < tail
          assert (!tail->hasState (ContextState::COMMIT_DONE));
          assert (!tail->hasState (ContextState::COMMITTING));

          if (tail->isRunning ()) {
            tail->abortSelf = true;

            mutex.unlock ();
            ctxt->waitFor (tail);
            mutex.lock ();
            continue;

          } else {

            assert (!tail->hasState (ContextState::SCHEDULED));
            assert (!tail->hasState (ContextState::ABORT_SELF));
            assert (!tail->hasState (ContextState::ABORTING));

            // TODO: abort tail, by only one thread, only once
            continue;

          }

        } else { // ctxt >= tail

          assert (ctxt != tail);

          if (tail->isRunning ()) {
            mutex.unlock ();
            ctxt->waitFor (tail);
            mutex.lock ();
            continue;

          } else {
            assert (!tail->isRunning ());
            break;
          }


        }
      }

      sharers.push_back (ctxt);
      return true;


    }

  }

  void removeAbort (Ctxt* ctxt) {

    mutex.lock ();

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

        mutex.unlock ();
        tail->doAbort ();
        mutex.lock ();
      } else {
        assert (!found);
        GALOIS_DIE ("shouldn't reach here");
      }
    }

    assert (found);
    assert (std::find (sharers.begin (), sharers.end (), ctxt) == sharers.end ());

    mutex.unlock ();
  }

  void removeCommit (Ctxt* ctxt) {

    mutex.lock ();

    assert (std::find (sharers.begin (), sharers.end (), ctxt) != sharers.end ());
    assert (!sharers.empty ());
    assert (sharers.front () == ctxt);
    sharers.pop_front ();

    assert (std::find (sharers.begin (), sharers.end (), ctxt) == sharers.end ());

    mutex.unlock ();
  }

};


template <typename T, typename Cmp, typename Exec>
struct OptimContext: public SimpleRuntimeContext {

  using Base = SimpleRuntimeContext;

  using CtxtCmp = ContextComparator<OptimContext, Cmp>;
  using NItem = OptimNhoodItem<OptimContext, CtxtCmp>;
  using NhoodMgr = PtrBasedNhoodMgr<NItem>;
  using NhoodList = typename gstl::Vector<NItem*>;
  using ChildList = typename gstl::Vector<OptimContext*>;
  using Lock_ty = Galois::Substrate::SimpleLock;

  T active;
  std::atomic<ContextState> state;
  Exec& exec;
  NhoodList nhood;
  bool abortSelf = false;
  Lock_ty mutex;
 
  bool finished = false;

  // TODO: avoid using UserContextAccess and per-iteration allocator
  // use Pow of 2 block allocator instead. 
  UserContextAccess<T> userHandle;
  ChildList children;


  explicit OptimContext (const T& x, const ContextState& s, Exec& exec)
  :
    Base (true), active (x), state (s), exec (exec) 
  {}

  const T& getActive () const { return active; }

  bool hasState (const ContextState& s) const { return state == s; } 

  void setState (const ContextState& s) { state = s; } 

  bool casState (ContextState s_old, const ContextState& s_new) { 
    // return state.cas (s_old, s_new);
    return state.compare_exchange_strong (s_old, s_new);
  }

  bool isRunning (void) const {
    return hasState (ContextState::SCHEDULED) || hasState (ContextState::ABORT_SELF);
  }

  void waitFor (OptimContext* that) const {
    assert (that);

    while (that->isRunning ()) {
      Substrate::asmPause ();
    }
  }

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

  void schedule (void) {
    setState (ContextState::SCHEDULED); 
    nhood.clear ();
    userHandle.reset ();
    children.clear ();
    abortSelf = false;
    finished = false;
  }

  void publishChanges (void) {

    userHandle.commit ();
    for (auto i = userHandle.getPushBuffer ().begin (), 
        end_i = userHandle.getPushBuffer ().end (); i != end_i; ++i) {

      OptimContext* child = exec.push (*i);
      dbg::debug (this, " creating child ", child);

      children.push_back (child);
    }
  }

  void doCommit () {

    if (hasState (ContextState::COMMIT_DONE) || hasState (ContextState::COMMITTING)) {

      return;

    } else {

      mutex.lock ();
      
      if (hasState (ContextState::COMMIT_DONE) || hasState (ContextState::COMMITTING)) {
        mutex.unlock ();
        return;
      }

      assert (hasState (ContextState::READY_TO_COMMIT));
      
      setState (ContextState::COMMITTING);

      dbg::debug (this, " committing with item ", this->active);
      for (NItem* n: nhood) {
        n->removeCommit (this);
      }

      setState (ContextState::COMMIT_DONE);

      mutex.unlock ();

    }

  }

  void doAbort (bool addBack=true) {
    // this can be in states READY_TO_COMMIT, ABORT_SELF
    // children can be in UNSCHEDULED, READY_TO_COMMIT, ABORT_DONE

    // first abort all the children recursively
    // then abort self.
    //
    if (hasState (ContextState::ABORT_DONE) || hasState (ContextState::ABORTING)) {
      return;
    }

    mutex.lock ();

    if (hasState (ContextState::ABORT_DONE) || hasState (ContextState::ABORTING)) { 
      mutex.unlock ();
      return;
    }

    setState (ContextState::ABORTING);


    for (OptimContext* child: children) {

      assert (!child->hasState (ContextState::SCHEDULED));
      assert (!child->hasState (ContextState::ABORTED_CHILD));

      bool c = child->hasState (ContextState::UNSCHEDULED)
        || child->hasState (ContextState::READY_TO_COMMIT)
        || child->hasState (ContextState::ABORT_DONE);
      assert (c);

      if (child->hasState (ContextState::READY_TO_COMMIT)) {
        dbg::debug (this, " aborting child ", child);
        child->doAbort (false);
        exec.freeCtxt (child);
      } else {
        child->setState (ContextState::ABORTED_CHILD);
      }
    }

    dbg::debug (this, " aborting with item ", this->active);

    userHandle.rollback ();

    for (NItem* n: nhood) {
      n->removeAbort (this);
    }

    if (addBack) {
      setState (ContextState::ABORT_DONE);
      exec.push_abort (this);
    }

    mutex.unlock ();

  }

};

template <typename T, typename Cmp, typename NhFunc, typename OpFunc> 
class OptimOrdExecutor: private boost::noncopyable {

  friend struct OptimContext<T, Cmp, OptimOrdExecutor>;
  using Ctxt = OptimContext<T, Cmp, OptimOrdExecutor>;
  using NhoodMgr = typename Ctxt::NhoodMgr;
  using CtxtCmp = typename Ctxt::CtxtCmp;
  using NItemFactory = typename Ctxt::NItem::Factory;

  using CommitQ = Galois::PerThreadList<Ctxt*>;
  using PendingQ = Galois::PerThreadMinHeap<Ctxt*, CtxtCmp>;

  using CtxtAlloc = Runtime::FixedSizeAllocator<Ctxt>;


  Cmp itemCmp;
  NhFunc nhFunc;
  OpFunc opFunc;
  Substrate::Barrier& gvtBarrier;
  const char* loopname;

  CtxtCmp ctxtCmp;
  PendingQ pendingQ; 
  CommitQ commitQ;
  NItemFactory nitemFactory;
  NhoodMgr nhmgr;
  Substrate::TerminationDetection& term;


  GAtomic<bool> startGVT;
  Ctxt* globalFront;


  CtxtAlloc ctxtAlloc;
  Substrate::PerThreadStorage<Ctxt*> perThrdLocalFront; 

public:
  OptimOrdExecutor (const Cmp& cmp, const NhFunc& nhFunc, const OpFunc& opFunc, Substrate::Barrier& barrier, const char* loopname)
    : 
      itemCmp (cmp), 
      nhFunc (nhFunc), 
      opFunc (opFunc), 
      gvtBarrier (barrier),
      loopname (loopname),
      ctxtCmp (itemCmp),
      pendingQ (ctxtCmp),
      commitQ (),
      nitemFactory (ctxtCmp),
      nhmgr (nitemFactory),
      term (Substrate::getSystemTermination (activeThreads))
  {
    if (!loopname) { loopname = "NULL"; }
  }

  ~OptimOrdExecutor (void) {
  }

  const Cmp& getItemCmp () const { return itemCmp; }

  const CtxtCmp& getCtxtCmp () const { return ctxtCmp; }

  // on each thread
  template <typename R>
  void push_initial (const R& range) {

    push (range.local_begin (), range.local_end ());

    term.initializeThread ();
  }

  template <typename Iter>
  void push (Iter beg, Iter end) {
    for (Iter i = beg; i != end; ++i) {
      push (*i);
    }
  }

  Ctxt* push (const T& x) {
    Ctxt* ctxt = ctxtAlloc.allocate (1);
    assert (ctxt);
    ctxtAlloc.construct (ctxt, x, ContextState::UNSCHEDULED, *this);
    pendingQ.get ().push (ctxt); 

    return ctxt;
  }

  void push_abort (Ctxt* ctxt) {
    assert (ctxt);
    assert (ctxt->hasState (ContextState::ABORT_DONE));

    ctxt->setState (ContextState::UNSCHEDULED);
    pendingQ.get ().push (ctxt);
  }

  void execute () {

    static const unsigned commit_interval = 10;

    size_t totalIter = 0;
    size_t totalCommits = 0;

    do {
      bool didWork = false;


      unsigned num_scheduled = 0;

      for (; (num_scheduled < commit_interval) && !pendingQ.get ().empty (); ++num_scheduled, ++totalIter) {

        Ctxt* ctxt = schedule ();

        if (ctxt) {
          didWork = true;

          applyOperator (ctxt);

          if (!ctxt->casState (ContextState::SCHEDULED, ContextState::READY_TO_COMMIT)) {

            if (ctxt->casState (ContextState::ABORT_SELF, ContextState::ABORTING)) {

              dbg::debug (ctxt, " Shouldn't reach here with ABORT_SELF");
              ctxt->doAbort ();

            }
          } else {


            dbg::debug (ctxt, " adding self to commit queue");
            commitQ.get ().push_back (ctxt);
            // publish remaining changes
            ctxt->publishChanges ();

          }

        }

      }

      // TODO: do this after processing some tasks
      unsigned numCommitted = localCommit ();
      totalCommits += numCommitted;

      if (!computeGVT ()) {
        reportGVT ();
      }

      if (!commitQ.get ().empty () || !pendingQ.get ().empty ()) {
        continue;
      }


      didWork = didWork || (numCommitted > 0);


      // TODO: insert stealing here


      term.localTermination (didWork);

    } while (!term.globalTermination ());



    std::printf ("OptimOrdExecutor: totalIter=%zd, totalCommits=%zd\n", totalIter, totalCommits);
  }

  void operator () (void) {
    execute ();
  }
  
private:

  GALOIS_ATTRIBUTE_PROF_NOINLINE void applyOperator (Ctxt* ctx) {

    assert (ctx);

    Galois::Runtime::setThreadContext (ctx);
    nhFunc (ctx->active, ctx->userHandle);

    if (ctx->hasState (ContextState::SCHEDULED)) {
      opFunc (ctx->active, ctx->userHandle);
    }
    Galois::Runtime::setThreadContext (nullptr);

  }


  void freeCtxt (Ctxt* ctxt) {
    ctxtAlloc.destroy (ctxt);
    ctxtAlloc.deallocate (ctxt, 1);
  }

  Ctxt* schedule (void) {

    while (!pendingQ.get ().empty ()) {

      Ctxt* ctxt = pendingQ.get ().pop ();

      bool b = ctxt->hasState (ContextState::UNSCHEDULED) 
        || ctxt->hasState (ContextState::ABORTED_CHILD);
      assert (b);

      if (ctxt->hasState (ContextState::UNSCHEDULED)) {
        ctxt->schedule ();
        return ctxt;

      } else {
        assert (ctxt->hasState (ContextState::ABORTED_CHILD));
        dbg::debug ("deleting aborted child: ", ctxt, " with item ", ctxt->active);
        freeCtxt (ctxt);

      }
    } // end while

    return nullptr;
  }

  void computeLocalFront (void) {
    if (!pendingQ.get ().empty ()) { 
      *(perThrdLocalFront.getLocal ()) = pendingQ.get ().top ();

    } else {
      *(perThrdLocalFront.getLocal ()) = nullptr;
    }
  }

  void computeGlobalFront (void) {
    assert (bool (startGVT));


    Ctxt* minp = nullptr;

    for (unsigned i = 0; i < Galois::getActiveThreads (); ++i) {
      Ctxt* lf = *(perThrdLocalFront.getRemote (i));

      if (lf && (!minp || ctxtCmp (lf, minp))) {
        minp = lf;
      }
    }

    globalFront = minp;
  }


  bool computeGVT (void) {
    if (startGVT.cas (false, true)) {

      computeLocalFront ();

      gvtBarrier ();

      computeGlobalFront ();

      startGVT = false;

      gvtBarrier ();

      return true;

    } else {
      return false;
    }
  }

  void reportGVT (void) {

    if (bool (startGVT)) {
      computeLocalFront ();

      gvtBarrier ();

      gvtBarrier ();
      assert (!bool (startGVT));
    }
  }


  size_t localCommit (void) {

    auto uniq_end = std::unique (commitQ.get ().begin (), commitQ.get ().end ());

    commitQ.get ().erase (uniq_end, commitQ.get ().end ());
    
    Ctxt* gfront = globalFront;

    size_t numCommitted = 0;

    if (gfront) {

      // all tasks lesser than globalMin
      auto c_end = std::partition (commitQ.get ().begin (), commitQ.get ().end (), 
          [&] (Ctxt* c) { 
            assert (c);
            return ctxtCmp (c, gfront);
          });



      for (auto i = commitQ.get ().begin (); i != c_end; ++i) {

        assert (ctxtCmp (*i, gfront));
        assert ((*i)->hasState (ContextState::READY_TO_COMMIT));

        if ((*i)->hasState (ContextState::READY_TO_COMMIT)) {
          (*i)->doCommit ();
          ++numCommitted;
        }
      }

      commitQ.get ().erase (commitQ.get ().begin (), c_end);

      if (!commitQ.get ().empty ()) {
        // assert ( TODO
      }

    }

    return numCommitted;

  }

};


template <typename R, typename Cmp, typename NhFunc, typename OpFunc>
void for_each_ordered_optim (const R& range, Cmp cmp, NhFunc nhFunc, OpFunc opFunc, const char* loopname=0) {

  using T = typename R::value_type;

  unsigned numT = Galois::getActiveThreads ();
  auto& barrier = getBarrier (numT);

  OptimOrdExecutor<T, Cmp, NhFunc, OpFunc> exec (cmp, nhFunc, opFunc, barrier, loopname);

  Substrate::getSystemThreadPool ().run (numT,
      [&exec, &range] (void) { exec.push_initial (range); },
      std::ref (barrier),
      std::ref (exec));

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
struct OptimParamContext: public SimpleRuntimeContext {

  using Base = SimpleRuntimeContext;

  using CtxtCmp = ContextComparator<OptimParamContext, Cmp>;
  using NItem = OptimParamNhoodItem<OptimParamContext, CtxtCmp>;
  using NhoodMgr = PtrBasedNhoodMgr<NItem>;
  using NhoodList = typename gstl::Vector<NItem*>;
  using ChildList = typename gstl::Vector<OptimParamContext*>;

  T active;
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
    Base (true), active (x), state (s), step (step), exec (exec) 
  {}

  const T& getActive () const { return active; }

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

    dbg::debug (this, " committing with item ", this->active);
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
        exec.freeCtxt (child);
      } else {
        child->setState (ContextState::ABORTED_CHILD);
      }
    }

    dbg::debug (this, " aborting with item ", this->active);

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

        dbg::debug (ctxt, " scheduled with item ", ctxt->active);

        ++totalIter;

        Galois::Runtime::setThreadContext (ctxt);
        nhFunc (ctxt->active, ctxt->userHandle);

        if (ctxt->hasState (ContextState::SCHEDULED)) {
          opFunc (ctxt->active, ctxt->userHandle);
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
        dbg::debug ("head of rob: ", rob.back (),  "  with item: ", rob.back ()->active);

        dbg::debug ("head of nextPending: ", nextPending->top (),  "  with item: ", nextPending->top ()->active);
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
        dbg::debug ("deleting aborted child: ", ctxt, " with item ", ctxt->active);
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
