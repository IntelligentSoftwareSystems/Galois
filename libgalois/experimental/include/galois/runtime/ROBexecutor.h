#ifndef GALOIS_RUNTIME_ROB_EXECUTOR_H
#define GALOIS_RUNTIME_ROB_EXECUTOR_H

#include "galois/Reduction.h"
#include "galois/Atomic.h"
#include "galois/BoundedVector.h"
#include "galois/Galois.h"
#include "galois/gdeque.h"
#include "galois/PriorityQueue.h"
#include "galois/Timer.h"
#include "galois/PerThreadContainer.h"

#include "galois/substrate/Barrier.h"
#include "galois/runtime/Context.h"
#include "galois/runtime/Executor_DoAll.h"
#include "galois/runtime/Executor_ParaMeter.h"
#include "galois/runtime/ForEachTraits.h"
#include "galois/runtime/Range.h"
#include "galois/runtime/Profile.h"
#include "galois/runtime/Support.h"
#include "galois/substrate/Termination.h"
#include "galois/substrate/ThreadPool.h"
#include "galois/runtime/UserContextAccess.h"
#include "galois/gIO.h"
#include "galois/substrate/CompilerSpecific.h"
#include "galois/runtime/Mem.h"
#include "galois/runtime/OrderedSpeculation.h"

#include <atomic>

namespace galois {
namespace runtime {

  // race conditions
  // 1. two iterations trying to abort the same iteration
  //  a. two iterations trying to abort an iteration that has already executed
  //  b. an iteration trying to abort self, while other aborting it when clearing
  //  rob
  // 2. The iteration itself trying to go into RTC, while other trying to abort it
  // 3. Two threads trying to schedule item from pending
  // 4. One thread trying to abort or add an item after commit, while other trying to
  // schedule an item from pending
  // 5.

  // multiple attempts to abort an iteration
  // soln1: use a mutex per iteration and use state to indicate if someone else
  // already aborted the iteration
  // soln2: for an iteration that has executed, the threads competing to abort it
  // use a cas (on state) to find the winner who goes on to abort the iteration
  // for an iteration that has not completed execution yet, the thread signals the
  // iteration to abort itself. each iteration keeps track of its owner thread and
  // only the owner thread aborts the iteration.
  //

  // TODO: throw abort exception vs use a flag
  // on self aborts
  // TODO: memory management: other threads may refer to an iteration context that has
  // been deallocated after commit or abort
template <typename T, typename Cmp, typename Exec>
class ROBcontext: public OrderedContextBase<T> {

  using Base = OrderedContextBase<T>;
  using NhoodList =  galois::gdeque<Lockable*, 4>; // 2nd arg is chunk size

public:

  enum class State: int {
    UNSCHEDULED,
    SCHEDULED,
    READY_TO_COMMIT,
    ABORT_SELF,
    ABORT_HELP,
    COMMITTING,
    ABORTING,
    COMMIT_DONE,
    ABORT_DONE,
  };

  // TODO: privatize
public:
  // GALOIS_ATTRIBUTE_ALIGN_CACHE_LINE galois::GAtomic<State> state;
  GALOIS_ATTRIBUTE_ALIGN_CACHE_LINE std::atomic<State> state;
  Exec& executor;

  bool lostConflict;
  volatile bool executed;

  unsigned owner;

  NhoodList nhood;
  UserContextAccess<T> userHandle;



private:
  ROBcontext (const ROBcontext& that) {} // TODO: remove since SimpleRuntimeContext inherits from boost::noncopyable

public:

  explicit ROBcontext (const T& x, Exec& e)
    :
      Base (x),
      state (State::UNSCHEDULED),
      executor (e),
      lostConflict (false),
      executed (false),
      owner (substrate::ThreadPool::getTID ())

  {}

  bool hasExecuted () const { return executed; }

  bool hasState (State s) const { return ((State) state) == s; }

  void setState (State s) {
    state = s;
  }

  bool casState (State s_old, State s_new) {
    // return state.cas (s_old, s_new);
    return state.compare_exchange_strong (s_old, s_new);
  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE
  virtual void subAcquire (Lockable* l, galois::MethodFlag) {

    for (bool succ = false; !succ; ) {
      typename Base::AcquireStatus acq = Base::tryAcquire (l);

      switch (acq) {
        case Base::FAIL: {
          ROBcontext* volatile that = static_cast<ROBcontext*> (Base::getOwner (l));
          if (that != nullptr) {
            // assert (dynamic_cast<ROBcontext*> (Base::getOwner (l)) != nullptr);
            bool abortSelf = resolveConflict (that, l);
            succ = abortSelf;
            lostConflict = true;

          } else {
            dbg::debug ("owner found to be null, current value: ", Base::getOwner (l)
                , " for lock: ", l);
          }
          break;
        }

        case Base::NEW_OWNER: {
          nhood.push_back (l);
          succ = true;
          break;
        }

        case Base::ALREADY_OWNER: {
          assert (std::find (nhood.begin (), nhood.end (), l) != nhood.end ());
          succ = true;
          break;
        }

        default: {
          GALOIS_DIE ("invalid acquire status");
          break;
        }
      }
    }
  }



  GALOIS_ATTRIBUTE_PROF_NOINLINE void doCommit () {
    assert (hasState (State::COMMITTING));
    // release locks
    // add new elements to worklist

    // TODO: check for typetraits 'noadd'

    userHandle.commit ();
    releaseLocks ();
    executor.push (userHandle.getPushBuffer ().begin (), userHandle.getPushBuffer ().end ());
    userHandle.reset ();

    substrate::compilerBarrier ();

    setState (State::COMMIT_DONE);
  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE void doAbort () {
    assert (hasState (State::ABORTING));
    // perform undo actions in reverse order
    // release locks
    // add active element to worklist

    userHandle.rollback ();
    releaseLocks ();
    executor.push_abort (Base::getActive (), owner);
    userHandle.reset ();

    substrate::compilerBarrier ();

    setState (State::ABORT_DONE);

  }

private:

  void releaseLocks () {
    for (Lockable* l: nhood) {
      assert (l != nullptr);
      if (Base::getOwner (l) == this) {
        dbg::debug (this, " releasing lock ", l);
        Base::release (l);
      }
    }

  }

private:

  GALOIS_ATTRIBUTE_PROF_NOINLINE bool resolveConflict (ROBcontext* volatile that, const Lockable* const l) {
    // precond: this could not acquire lock. lock owned by that
    // this can only be in state SCHEDULED or ABORT_SELF
    // that can be in SCHEDULED, ABORT_SELF, ABORT_HELP, READY_TO_COMMIT, ABORT_DONE
    // returns true if this loses the conflict and signals itself to abort

    bool ret = false;

    if (executor.getCtxtCmp () (this, that)) {

      assert (!that->hasState (State::COMMIT_DONE) && !that->hasState (State::COMMITTING));
      // abort that
      if (that->hasState (State::ABORT_DONE)) {
        // do nothing

      } else if (that->casState (State::SCHEDULED, State::ABORT_SELF) || that->hasState (State::ABORT_SELF)) {
        // signalled successfully
        // now wait for it to abort or abort yourself if 'that' missed the signal
        // and completed execution
        dbg::debug ( this, " signalled ", that, " to ABORT_SELF on lock ", l);
        while (true) {

          if (that->hasState (State::ABORT_DONE)) {
            break;
          }

          if (that->hasExecuted ()) {
            if (that->casState (State::ABORT_SELF, State::ABORT_HELP)) {
              that->setState (State::ABORTING);
              that->doAbort ();
              executor.abortByOther += 1;
              dbg::debug (this, " aborting ABORT_SELF->ABORT_HELP missed signal ", that, " on lock ", l);
            }

          }

          substrate::asmPause ();
        }

      } else if (that->casState (State::READY_TO_COMMIT, State::ABORT_HELP)) {
        that->setState (State::ABORTING);
        that->doAbort ();
        executor.abortByOther += 1;
        dbg::debug (this, " aborting RTC->ABORT_HELP ", that, " on lock ", l);
      }

    } else {
      // abort self
      this->setState (State::ABORT_SELF);
      dbg::debug (this, " losing conflict with ", that, " on lock ", l);
      ret = true;
    }

    return ret;

  }



};

template <typename T, typename Cmp, typename NhFunc, typename OpFunc>
class ROBexecutor: private boost::noncopyable {

  using Ctxt = ROBcontext<T, Cmp, ROBexecutor>;
  using CtxtAlloc = FixedSizeAllocator<Ctxt>;
  using CtxtCmp = ContextComparator<Ctxt, Cmp>;
  using CtxtDeq = PerThreadDeque<Ctxt*>;
  using CtxtVec = PerThreadVector<Ctxt*>;

  using PendingQ = galois::MinHeap<T, Cmp>;
  using PerThrdPendingQ = PerThreadMinHeap<T, Cmp>;
  using ROB = galois::MinHeap<Ctxt*, CtxtCmp>;

  using Lock_ty = galois::substrate::SimpleLock;
  // using Lock_ty = galois::runtime::LL::PthreadLock<true>;


  Cmp itemCmp;
  NhFunc nhFunc;
  OpFunc opFunc;
  CtxtCmp ctxtCmp;

  PerThrdPendingQ pending;
  ROB rob;
  substrate::TerminationDetection& term;


  CtxtAlloc ctxtAlloc;
  // CtxtDeq ctxtDelQ;
  CtxtDeq freeList;

  // GALOIS_ATTRIBUTE_ALIGN_CACHE_LINE Lock_ty pendingMutex;
  substrate::PerThreadStorage<Lock_ty> pendingMutex;

  GALOIS_ATTRIBUTE_ALIGN_CACHE_LINE Lock_ty robMutex;

  GAccumulator<size_t> numTotal;
  GAccumulator<size_t> numCommitted;
  GAccumulator<size_t> numGlobalCleanups;



  static const size_t WINDOW_SIZE_PER_THREAD = 1024;

  // static const size_t DELQ_THRESHOLD_UPPER = 1100;
  // static const size_t DELQ_THRESHOLD_LOWER = 1000;

public:

  GAccumulator<size_t> abortSelfByConflict;
  GAccumulator<size_t> abortSelfBySignal;
  GAccumulator<size_t> abortByOther;

  ROBexecutor (const Cmp& cmp, const NhFunc& nhFunc, const OpFunc& opFunc)
    :
      itemCmp (cmp),
      nhFunc (nhFunc),
      opFunc (opFunc),
      ctxtCmp (itemCmp),
      pending (itemCmp),
      rob (ctxtCmp),
      term (substrate::getSystemTermination (activeThreads))
  {}

  const Cmp& getItemCmp () const { return itemCmp; }

  const CtxtCmp& getCtxtCmp () const { return ctxtCmp; }

  template <typename Iter>
  void push (Iter beg, Iter end) {
    // TODO: whether to add new elements to the owner or to the committer?

    pendingMutex.getLocal ()->lock (); {
      for (Iter i = beg; i != end; ++i) {
        pending.get ().push (*i);
      }
    }
    pendingMutex.getLocal ()->unlock ();
  }

  template <typename R>
  GALOIS_ATTRIBUTE_PROF_NOINLINE void push_initial (const R& range) {

    assert (range.begin () != range.end ());

    galois::do_all (range,
        [this] (const T& x) {
        pending.get ().push (x);
        });

    assert (!pending.empty_all ());

    const T& dummy = *pending[0].begin ();

    galois::on_each(
        [&dummy,this] (const unsigned tid, const unsigned numT) {
          for (unsigned j = 0; j < WINDOW_SIZE_PER_THREAD; ++j) {
            Ctxt* ctx = ctxtAlloc.allocate (1);
            assert (ctx != nullptr);
            ctxtAlloc.construct (ctx, dummy, *this);
            ctx->setState (Ctxt::State::SCHEDULED);

            freeList[tid].push_back (ctx);
          }
        });

    // for (unsigned i = 0; i < freeList.numRows (); ++i) {
      // for (unsigned j = 0; j < WINDOW_SIZE_PER_THREAD; ++j) {
        // Ctxt* ctx = ctxtAlloc.allocate (1);
        // assert (ctx != nullptr);
        // ctxtAlloc.construct (ctx, dummy, *this);
        // ctx->setState (Ctxt::State::SCHEDULED);
//
        // freeList[i].push_back (ctx);
      // }
    // }
    term.initializeThread ();
  }

  void push_abort (const T& x, const unsigned owner) {

    // TODO: who gets the aborted item, owner or aborter?

    // tree based serialization
    unsigned nextOwner = owner / 2;

    pendingMutex.getRemote (nextOwner)->lock (); {
      pending[nextOwner].push (x);
    } pendingMutex.getRemote (nextOwner)->unlock ();


    // pendingMutex.getLocal ()->lock (); {
      // pending.get ().push (x);
    // } pendingMutex.getLocal ()->unlock ();

  }

  void execute () {


    do {

      bool didWork = false;

      do {

        Ctxt* ctx = scheduleGlobalMinFirst ();

        if (ctx != nullptr) {

          didWork = true;

          dbg::debug (ctx, " scheduled with item ", ctx->getActive (),
              " remaining contexts: ", freeList.get ().size ());

          applyOperator (ctx);

          if (!ctx->casState (Ctxt::State::SCHEDULED, Ctxt::State::READY_TO_COMMIT)) {
            if (ctx->casState (Ctxt::State::ABORT_SELF, Ctxt::State::ABORTING)) {

              if (ctx->lostConflict) {
                abortSelfByConflict += 1;

              } else {
                abortSelfBySignal += 1;
              }

              ctx->doAbort ();
              dbg::debug (ctx, " aborting SELF after reading signal");
            }
          }

          ctx->executed = true;

          substrate::compilerBarrier ();

        }


        bool cleared = clearROB (ctx);

        didWork = didWork || cleared;

        // // TODO: termination detection
        // if (robEmpty) {
        // bool fin = false;
        //
        // // check my queue first
        // pendingMutex.lock (); {
        // fin = pending.empty ();
        // } pendingMutex.unlock ();
        //
        // abortQmutex.lock (); {
        // fin = fin && abortQ.empty ();
        // } abortQmutex.unlock ();
        //
        // if (fin) {
        // break;
        // }
        // }

        // XXX: unprotected check. may crash
      } while (!rob.empty () || !pending.empty_all ());

      term.localTermination (didWork);

    } while (!term.globalTermination ());
  }

  void operator () () {
    execute ();
  }

  void printStats () {
    // just to compile the size methods for debugging
    rob.size ();
    pending.size_all ();

    assert (rob.empty ());
    assert (pending.empty_all ());


    double ar = double (numTotal.reduce () - numCommitted.reduce ()) / double (numTotal.reduce ());
    double totalAborts = double (abortSelfByConflict.reduce () + abortSelfBySignal.reduce () + abortByOther.reduce ());
    gPrint("Total Iterations: ", numTotal.reduce(), "\n");
    gPrint("Number Committed: ", numCommitted.reduce(), "\n");
    gPrint("Abort Ratio: ", ar, "\n");
    gPrint("abortSelfByConflict: ", abortSelfByConflict.reduce(), ", ", (100.0*abortSelfByConflict.reduce())/totalAborts, "%", "\n");
    gPrint("abortSelfBySignal: ", abortSelfBySignal.reduce(), ", ", (100.0*abortSelfBySignal.reduce())/totalAborts, "%", "\n");
    gPrint("abortByOther: ", abortByOther.reduce(), ", ", (100.0*abortByOther.reduce())/totalAborts, "%", "\n");
    gPrint("Number of Global Cleanups: ", numGlobalCleanups.reduce(), "\n");
  }

private:

  GALOIS_ATTRIBUTE_PROF_NOINLINE void applyOperator (Ctxt* ctx) {

    assert (ctx != nullptr);

    galois::runtime::setThreadContext (ctx);
    nhFunc (ctx->getActive (), ctx->userHandle);

    if (ctx->hasState (Ctxt::State::SCHEDULED)) {
      opFunc (ctx->getActive (), ctx->userHandle);
    }
    galois::runtime::setThreadContext (nullptr);

  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE Ctxt* scheduleGlobalMinFirst () {
    bool notEmpty = true;
    Ctxt* ctx = nullptr;

    // XXX: unprotected check, may cause crash
    notEmpty = !freeList.get ().empty () && !pending.empty_all ();

    if (notEmpty) {
      robMutex.lock (); {

        if (!freeList.get ().empty ()) {

          unsigned minTID = 0;
          const T* minPtr = nullptr;

          for (unsigned i = 0; i < getActiveThreads (); ++i) {

            pendingMutex.getRemote (i)->lock (); {

              if (!pending[i].empty ()) {
                if (minPtr == nullptr || itemCmp (pending[i].top (), *minPtr)) {
                  // minPtr == nullptr or pending[i].top () < *minPtr
                  minPtr = &pending[i].top ();
                  minTID = i;
                }
              }

            } pendingMutex.getRemote (i)->unlock ();
          } // end for

          pendingMutex.getRemote (minTID)->lock (); {

            if (!pending[minTID].empty ()) {

              ctx = freeList.get ().back ();
              freeList.get ().pop_back ();

              ctx->~Ctxt (); // destroy here only
              new (ctx) Ctxt (pending[minTID].pop (), *this);

              ctx->setState (Ctxt::State::SCHEDULED);
              ctx->owner = substrate::ThreadPool::getTID ();
              rob.push (ctx);
              numTotal += 1;
            }

          } pendingMutex.getRemote (minTID)->unlock ();

        }

      } robMutex.unlock ();

    } // end if notEmpty

    return ctx;
  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE Ctxt* scheduleThreadLocalFirst () {
    bool notEmpty = true;
    Ctxt* ctx = nullptr;

    // XXX: unprotected check, may cause crash
    notEmpty = !freeList.get ().empty () && !pending.empty_all ();

    if (notEmpty) {
      robMutex.lock (); {

        if (!freeList.get ().empty ()) {
          unsigned beg = substrate::ThreadPool::getTID ();
          unsigned end = beg + getActiveThreads ();

          for (unsigned i = beg; i < end; ++i) {

            unsigned tid = i % getActiveThreads ();

            pendingMutex.getRemote (tid)->lock (); {

              if (!pending[tid].empty ()) {
                ctx = freeList.get ().back ();
                freeList.get ().pop_back ();

                ctx->~Ctxt (); // destroy here only
                new (ctx) Ctxt (pending[tid].pop (), *this);

                ctx->setState (Ctxt::State::SCHEDULED);
                ctx->owner = substrate::ThreadPool::getTID ();
                rob.push (ctx);
                numTotal += 1;
              }

            } pendingMutex.getRemote (tid)->unlock ();

            if (ctx != nullptr) {
              break;
            }

          } // end for
        } // end if freeList
      } robMutex.unlock ();
    }

    return ctx;
  }

  bool isEarliest (const T& x) {
    bool earliest = true;

    for (unsigned i = 0; i < getActiveThreads (); ++i) {

      pendingMutex.getRemote (i)->lock (); {

        if (!pending[i].empty ()) {
          earliest = earliest && !itemCmp (pending[i].top (), x);
        }

      } pendingMutex.getRemote (i)->unlock ();

      if (!earliest) {
        break;
      }
    }

    return earliest;
  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE bool clearROB (Ctxt* __ctxt) {
    bool didWork = false;

    robMutex.lock (); {
      while (!rob.empty ()) {

        Ctxt* head = rob.top ();

        if (head->hasState (Ctxt::State::ABORT_DONE)) {
          // ctxtDelQ.get ().push_back (rob.pop ());
          reclaim (rob.pop ());
          didWork = true;
          continue;

        }  else if (head->hasState (Ctxt::State::READY_TO_COMMIT)) {

          if (isEarliest (head->getActive ())) {

            head->setState (Ctxt::State::COMMITTING);
            head->doCommit ();

            Ctxt* t = rob.pop ();
            assert (t == head);

            // ctxtDelQ.get ().push_back (t);
            reclaim (t);
            didWork = true;
            numCommitted += 1;
            dbg::debug (__ctxt, " committed: ", head, ", with active: ", head->getActive ());

          } else {
            break;
          }

        } else {
          break;
        }
      }

      if (!rob.empty () && freeList.empty_all ()) {
        // a deadlock situation where no more contexts to schedule
        // and more work needs to be done
        reclaimGlobally ();
      }

    } robMutex.unlock ();

    return didWork;
  }

  void reclaim (Ctxt* ctx) {
    // assumes that robMutex is acquired
    unsigned owner = ctx->owner;
    // return to owner. Safe since we already hold robMutex and any scheduling
    // thread must also acquire robMutex.
    freeList[owner].push_back (ctx);
    // freeList.get ().push_back (ctx);
  }

  void reclaimGlobally () {
    // assumes that robMutex is acquired
    //

    numGlobalCleanups += 1;

    std::vector<Ctxt*> buffer;
    buffer.reserve (rob.size ());

    while (!rob.empty ()) {
      Ctxt* ctx = rob.pop ();

      if (ctx->hasState (Ctxt::State::ABORT_DONE)) {
        reclaim (ctx);

      } else {
        buffer.push_back (ctx);
      }
    }

    for (auto i = buffer.begin (), endi = buffer.end (); i != endi; ++i) {
      rob.push (*i);
    }

  }


  // void reclaimMemoryImpl (size_t upperLim, size_t lowerLim) {
    // assert (upperLim >= lowerLim);
//
    // if (ctxtDelQ.get ().size () >= upperLim) {
      // for (size_t s = ctxtDelQ.get ().size (); s > lowerLim; --s) {
        // Ctxt* c = ctxtDelQ.get ().front ();
        // ctxtDelQ.get ().pop_front ();
//
        // ctxtAlloc.destroy (c);
        // ctxtAlloc.deallocate (c, 1);
      // }
      //
    // }
  // }
//
  // void reclaimMemoryPeriodic () {
   // reclaimMemoryImpl (DELQ_THRESHOLD_UPPER, DELQ_THRESHOLD_LOWER);
  // }
//
  // void reclaimMemoryFinal () {
    // reclaimMemoryImpl (0, 0);
  // }


};


namespace ParaMeter {

template <typename T, typename Cmp, typename Exec>
class ROBparamContext: public ROBcontext<T, Cmp, Exec> {

  using Base = ROBcontext<T, Cmp, Exec>;

public:
  const size_t step;

  ROBparamContext (const T& x, Exec& e, const size_t _step): Base (x, e), step (_step) {}

  virtual void subAcquire (Lockable* l, galois::MethodFlag m) {
    for (bool succ = false; !succ; ) {
      typename Base::AcquireStatus acq = Base::tryAcquire (l);

      switch (acq) {
        case Base::FAIL: {
          ROBparamContext* that = static_cast<ROBparamContext*> (Base::getOwner (l));
          assert (that != nullptr);
          bool abortSelf = resolveConflict (that, l);
          succ = abortSelf;
          break;
        }

        case Base::NEW_OWNER: {
          Base::nhood.push_back (l);
          succ = true;
          break;
        }

        case Base::ALREADY_OWNER: {
          assert (std::find (Base::nhood.begin (), Base::nhood.end (), l) != Base::nhood.end ());
          succ = true;
          break;
        }

        default: {
          GALOIS_DIE ("invalid acquire status");
          break;
        }
      }
    }
  }


private:

  bool resolveConflict (ROBparamContext* that, const Lockable* const l) {
    // this can be in SCHEDULED or ABORT_SELF
    // that can be in READY_TO_COMMIT only
    // return true if this aborts self

    assert (this->hasState (Base::State::SCHEDULED) || this->hasState (Base::State::ABORT_SELF));
    assert (that->hasState (Base::State::READY_TO_COMMIT));

    bool ret = false;
    if (Base::executor.getCtxtCmp () (this, that)) {
      assert (that->hasState (Base::State::READY_TO_COMMIT));
      that->setState (Base::State::ABORTING);
      that->doAbort ();
      dbg::debug (this, " aborting ", that, " on lock ", l);

    } else {
      this->setState (Base::State::ABORT_SELF);
      ret = true;
    }

    return ret;
  }

};


template <typename T, typename Cmp, typename NhFunc, typename OpFunc>
class ROBparaMeter: private boost::noncopyable {

  using Ctxt = ROBparamContext<T, Cmp, ROBparaMeter>;
  using CtxtAlloc = FixedSizeAllocator<Ctxt>;
  using CtxtCmp = ContextComparator<Ctxt, Cmp>;
  using CtxtDeq = galois::PerThreadDeque<Ctxt*>;

  using PendingQ = galois::MinHeap<T, Cmp>;
  using ROB = std::vector<Ctxt*>;
  using ExecutionRecords = std::vector<StepStats>;

  Cmp itemCmp;
  NhFunc nhFunc;
  OpFunc opFunc;
  const char* loopname;

  CtxtCmp ctxtCmp;
  ROB rob;

  PendingQ* currPending;
  PendingQ* nextPending;
  CtxtAlloc ctxtAlloc;
  ExecutionRecords execRcrds;
  size_t steps;


public:
  // TODO: unused statistic added to avoid compile error. Fix ROBcontext
  size_t abortByOther = 0;

  ROBparaMeter (const Cmp& cmp, const NhFunc& nhFunc, const OpFunc& opFunc, const char* loopname)
    :
      itemCmp (cmp),
      nhFunc (nhFunc),
      opFunc (opFunc),
      loopname (loopname),
      ctxtCmp (itemCmp)
  {
    currPending = new PendingQ;
    nextPending = new PendingQ;
    steps = 0;

    if (loopname == nullptr) { loopname = "NULL"; }
  }

  ~ROBparaMeter (void) {
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
      nextPending->push (*i);
    }
  }

  void push (const T& x) {
    nextPending->push (x);
  }

  void push_abort (const T& x, const unsigned owner) {
    nextPending->push (x);
  }

  void execute () {

    size_t totalIter = 0;
    size_t totalCommits = 0;

    while (!nextPending->empty () || !rob.empty ()) {

      ++steps;
      std::swap (currPending, nextPending);
      nextPending->clear ();
      execRcrds.emplace_back (steps - 1, currPending->size ()); // create record entry for current step;
      assert (execRcrds.size () == steps);

      while (!currPending->empty ()) {
        Ctxt* ctx = schedule ();
        assert (ctx != nullptr);

        dbg::debug (ctx, " scheduled with item ", ctx->getActive ());

        ++totalIter;

        galois::runtime::setThreadContext (ctx);
        nhFunc (ctx->getActive (), ctx->userHandle);

        if (ctx->hasState (Ctxt::State::SCHEDULED)) {
          opFunc (ctx->getActive (), ctx->userHandle);
        }
        galois::runtime::setThreadContext (nullptr);

        if (ctx->hasState (Ctxt::State::SCHEDULED)) {
          ctx->setState (Ctxt::State::READY_TO_COMMIT);

        } else {
          assert (ctx->hasState (Ctxt::State::ABORT_SELF));
          ctx->setState (Ctxt::State::ABORTING);
          ctx->doAbort ();
        }

        rob.push_back (ctx);
      }

      size_t numCommitted = clearROB ();
      assert (numCommitted > 0);

      totalCommits += numCommitted;

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
      //FIXME: s.dump (getStatsFile (), loopname);
    }

    //FIXME: closeStatsFile ();
  }

  Ctxt* schedule () {

    assert (!currPending->empty ());

    Ctxt* ctx = ctxtAlloc.allocate (1);
    assert (ctx != nullptr);
    assert (steps > 0);
    ctxtAlloc.construct (ctx, currPending->pop (), *this, (steps-1));

    ctx->setState (Ctxt::State::SCHEDULED);

    return ctx;
  }

  size_t clearROB (void) {
    // first remove all tasks that are not in READY_TO_COMMIT
    auto new_end = std::partition (rob.begin (), rob.end (),
        [] (Ctxt* c) {
          assert (c != nullptr);
          return c->hasState (Ctxt::State::READY_TO_COMMIT);
        });


    for (auto i = new_end, end_i = rob.end (); i != end_i; ++i) {
      assert ((*i)->hasState (Ctxt::State::ABORT_DONE));
      freeCtxt (*i);
    }

    rob.erase (new_end, rob.end ());

    // now sort in reverse order
    auto revCmp = [this] (Ctxt* a, Ctxt* b) {
      assert (a != nullptr);
      assert (b != nullptr);
      return !ctxtCmp (a, b);
    };

    std::sort (rob.begin (), rob.end (), revCmp);

    // for debugging only, there should be no duplicates
    auto uniq_end = std::unique (rob.begin (), rob.end ());
    assert (uniq_end == rob.end ());

    size_t numCommitted = 0;

    while (!rob.empty ()) {
      Ctxt* head = rob.back ();

      assert (head->hasState (Ctxt::State::READY_TO_COMMIT));

      dbg::debug ("head of rob ready to commit : ", head);
      bool earliest = false;
      if (!nextPending->empty ()) {
        earliest = !itemCmp (nextPending->top (), head->getActive ());

      } else {
        earliest = true;
      }

      if (earliest) {

        head->setState (Ctxt::State::COMMITTING);
        head->doCommit ();
        rob.pop_back ();

        const size_t s = head->step;
        assert (s < execRcrds.size ());
        execRcrds[s].parallelism += 1;
        numCommitted += 1;
        freeCtxt (head);


      } else {
        dbg::debug ("head of rob could not commit : ", head);
        break;
      }

    }

    return numCommitted;

  }

  /*
  size_t clearROB (void) {

    size_t numCommitted = 0;

    while (!rob.empty ()) {
      Ctxt* head = rob.top ();

      if (head->hasState (Ctxt::State::ABORT_DONE)) {
        freeCtxt (rob.pop ());
        continue;

      } else if (head->hasState (Ctxt::State::READY_TO_COMMIT)) {
        assert (currPending->empty ());

        bool earliest = false;
        if (!nextPending->empty ()) {
          earliest = !itemCmp (nextPending->top (), head->getActive ());

        } else {
          earliest = true;
        }

        if (earliest) {
          head->setState (Ctxt::State::COMMITTING);
          head->doCommit ();
          Ctxt* t = rob.pop ();
          assert (t == head);

          const size_t s = head->step;
          assert (s < execRcrds.size ());
          ++execRcrds[s].parallelism;
          numCommitted += 1;
          freeCtxt (t);


        } else {
          break;
        }

      } else {
        GALOIS_DIE ("head in rob with invalid status");
      }
    }

    return numCommitted;
  }
  */



};



} // end namespace ParaMeter

template <typename R, typename Cmp, typename NhFunc, typename OpFunc>
void for_each_ordered_rob (const R& range, Cmp cmp, NhFunc nhFunc, OpFunc opFunc, const char* loopname=0) {

  using T = typename R::value_type;

  ParaMeter::ROBparaMeter<T, Cmp, NhFunc, OpFunc> exec (cmp, nhFunc, opFunc, loopname);

  exec.push_initial (range);
  exec.execute ();

//
  // ROBexecutor<T, Cmp, NhFunc, OpFunc>  exec (cmp, nhFunc, opFunc, loopname);
//
  // if (range.begin () != range.end ()) {
//
    // exec.push_initial (range);
//
    // getThreadPool ().run (activeThreads, std::ref(exec));
//
//
    // exec.printStats ();
  // }
}


} // end namespace runtime
} // end namespace galois

#endif //  GALOIS_RUNTIME_ROB_EXECUTOR_H
