#ifndef GALOIS_RUNTIME_SPECULATIVE_PARAMETER_H
#define GALOIS_RUNTIME_SPECULATIVE_PARAMETER_H

#include "Galois/PerThreadContainer.h"

#include "Galois/Runtime/Context.h"
#include "Galois/Runtime/ContextComparator.h"
#include "Galois/Runtime/CustomLockable.h"

namespace Galois {
namespace Runtime {


namespace ParaMeter {

enum class ContextState: int {
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


template <typename Ctxt, typename CtxtCmp>
struct OptimNhoodItem: public LockManagerBase {

  using Sharers = std::list<Ctxt*, Galois::Runtime::FixedSizeAllocator<Ctxt*> >;

  Sharers sharers;



  bool add (Ctxt* ctxt) {

    if (sharers.empty ()) { // empty 
      sharers.push_back (ctxt);
      return true;

    } else {

      while (!sharers.empty ()) {
        Ctxt* tail = sharers.back ();
        assert (tail != nullptr);

        assert (ctxt->step >= tail.step);
        if (ctxt->step == tail.step) {
          return false;
        }

        if (ctxtCmp (ctxt, tail)) { // ctxt < tail
          // tail should not be running
          assert (tail->hasState (State::READY_TO_COMMIT));
          tail->doAbort ();

        } else { // ctxt >= tail
          break;
        }
      }

      if (!sharers.empty ()) { 
        assert (ctxt->step >= sharers.back ()->step);
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
      assert (tail != nullptr);

      if (ctxt == tail) {
        sharers.pop_back (); 
        found = true;
        break;
      }

      if (ctxtCmp (ctxt, tail)) { // ctxt < tail
        tail->doAbort ();
      } else {
        assert (!found);
        GALOIS_DIE ("shouldn't reach here");
      }
    }

    assert (found);

  }

  void removeCommit (Ctxt* ctxt) {
    assert (std::find (sharers.begin (), sharers.end (), ctxt) != sharers.end ());
    assert (!sharers.empty ());
    assert (sharers.front () == ctxt);
    ctxt->pop_front ();
  }

};

template <typename T, typename Cmp, typename Exec>
struct OptimContext: public SimpleRuntimeContext {

  using Base = SimpleRuntimeContext;

  using CtxtCmp = ContextComparator<OptimContext, Cmp>;
  using NItem = OptimNhoodItem<OptimContext, CtxtCmp>;
  using NhoodMgr = PtrBasedNhoodMgr<NItem>;
  using NhoodList = typename ContainersWithGAlloc::Vector<NItem*>::type;
  using ChildList = typename ContainersWithGAlloc::Vector<OptimContext*>::type;

  T active;
  Exec& exec;
  ContextState state;
  NhoodList nhood;
  bool abortSelf = false;

  // TODO: avoid using UserContextAccess and per-iteration allocator
  // use Pow of 2 block allocator instead. 
  UserContextAccess<T> userHandle;
  ChildList children;


  explicit ROBcontext (const T& x, Exec& exec)
  :
    Base (true), active (x), exec (exec) 
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
      }
    }

  }

  void doCommit () {
    assert (state == ContextState::COMMITTING);

    for (NItem* n: nhood) {
      n->removeCommit (this);
    }

    state = COMMIT_DONE;
  }

  void doAbort () {
    assert (state == ContextState::ABORTING);

    // first abort all the children recursively
    // then abort self.
    //

    for (Ctxt* child: children) {
      assert (!child->hasState (ContextState::SCHEDULED));
      child->setState (ContextStat::ABORTING);
      child->doAbort ();
    }

    for (NItem* n: nhood) {
      n->removeAbort (this);
    }

    userHandle.rollback ();

    exec.push_abort (this);

    state = ABORT_DONE;
  }

};


template <typename T, typename Cmp, typename NhFunc, typename OpFunc> 
class OptimParaMeterExecutor: private boost::noncopyable {


  using Ctxt = OptimContext<T, Cmp>;
  using NhoodMgr = typename Ctxt::NhoodMgr;
  using CtxtCmp = typename Ctxt::CtxtCmp;

  using CommitQueue = Galois::MinHeap<Ctxt*, CtxtCmp>;
  using PendingQueue = Galois::MinHeap<Ctxt*, CtxtCmp>;

  using CtxtAlloc = Runtime::MM::FixedSizeAllocator<Ctxt>;
  using ExecutionRecords = std::vector<StepStats>;

  Cmp itemCmp;
  NhFunc nhFunc;
  OpFunc opFunc;
  const char* loopname;

  CtxtCmp ctxtCmp;
  CommitQueue rob;

  PendingQ* currPending;
  PendingQ* nextPending;
  
  CtxtAlloc ctxtAlloc;
  ExecutionRecords execRcrds;
  size_t steps;


public:
  ROBparaMeter (const Cmp& cmp, const NhFunc& nhFunc, const OpFunc& opFunc, const char* loopname)
    : 
      itemCmp (cmp), 
      nhFunc (nhFunc), 
      opFunc (opFunc), 
      loopname (loopname),
      ctxtCmp (itemCmp),
      rob (ctxtCmp)
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

    while (!nextPending->empty () || !rob.empty ()) {

      ++steps;
      std::swap (currPending, nextPending);
      nextPending->clear ();
      execRcrds.emplace_back (steps - 1, currPending->size ()); // create record entry for current step;
      assert (execRcrds.size () == steps);

      while (!currPending->empty ()) {
        Ctxt* ctx = schedule ();
        assert (ctx != nullptr);

        dbg::debug (ctx, " scheduled with item ", ctx->active);

        Galois::Runtime::setThreadContext (ctx);
        nhFunc (ctx->active, ctx->userHandle);

        if (ctx->hasState (Ctxt::State::SCHEDULED)) {
          opFunc (ctx->active, ctx->userHandle);
        }
        Galois::Runtime::setThreadContext (nullptr);

        if (ctx->hasState (Ctxt::State::SCHEDULED)) {
          ctx->setState (Ctxt::State::READY_TO_COMMIT);
          rob.push (ctx);

        } else {
          assert (ctx->hasState (Ctxt::State::ABORT_SELF));
          ctx->setState (Ctxt::State::ABORTING);
          ctx->doAbort ();
        }
      }

      size_t numCommitted = clearROB ();

      assert (numCommitted > 0);

    }

    finish ();
  }
  
private:

  void finish (void) {
    for (const StepStats& s: execRcrds) {
      s.dump (getStatsFile (), loopname);
    }

    closeStatsFile ();
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

    size_t numCommitted = 0;

    while (!rob.empty ()) {
      Ctxt* head = rob.top ();

      if (head->hasState (Ctxt::State::ABORT_DONE)) {
        rob.pop ();
        continue;

      } else if (head->hasState (Ctxt::State::READY_TO_COMMIT)) {
        assert (currPending->empty ());

        bool earliest = false;
        if (!nextPending->empty ()) {
          earliest = !itemCmp (nextPending->top (), head->active);

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


        } else {
          break;
        }

      } else {
        GALOIS_DIE ("head in rob with invalid status");
      }
    }

    return numCommitted;
  }



};



} // end namespace ParaMeter

template <typename R, typename Cmp, typename NhFunc, typename OpFunc>
void for_each_ordered_rob (const R& range, Cmp cmp, NhFunc nhFunc, OpFunc opFunc, const char* loopname=0) {

  using T = typename R::value_type;

  ParaMeter::ROBparaMeter<T, Cmp, NhFunc, OpFunc> exec (cmp, nhFunc, opFunc, loopname);

  exec.push_initial (range);
  exec.execute ();

  // Galois::Runtime::beginSampling ();
// 
  // ROBexecutor<T, Cmp, NhFunc, OpFunc>  exec (cmp, nhFunc, opFunc, loopname);
// 
  // if (range.begin () != range.end ()) {
// 
    // exec.push_initial (range);
// 
    // getSystemThreadPool ().run (activeThreads, std::ref(exec));
// 
    // Galois::Runtime::endSampling ();
// 
    // exec.printStats ();
  // }
}




} // end namespace Runtime
} // end namespace Galois


#endif // GALOIS_RUNTIME_SPECULATIVE_PARAMETER_H
