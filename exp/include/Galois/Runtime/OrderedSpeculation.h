#ifndef GALOIS_RUNTIME_ORDERED_SPECULATION_H
#define GALOIS_RUNTIME_ORDERED_SPECULATION_H

#include "Galois/PerThreadContainer.h"
#include "Galois/PriorityQueue.h"

#include "Galois/Runtime/Context.h"
#include "Galois/Runtime/OrderedLockable.h"

namespace Galois {
namespace Runtime {


namespace ParaMeter {

enum class ContextState: int {
    UNSCHEDULED,
    SCHEDULED,
    READY_TO_COMMIT,
    ABORT_SELF,
    ABORT_HELP,
    ABORTING,
    COMMIT_DONE,
    ABORT_DONE,
    ABORTED_CHILD,
};


template <typename Ctxt, typename CtxtCmp>
struct OptimNhoodItem: public OrdLocBase<OptimNhoodItem<Ctxt, CtxtCmp>, Ctxt, CtxtCmp> {

  using Base = OrdLocBase<OptimNhoodItem, Ctxt, CtxtCmp>;
  using Factory = OrdLocFactoryBase<OptimNhoodItem, Ctxt, CtxtCmp>;

  using Sharers = std::list<Ctxt*, Galois::Runtime::MM::FixedSizeAllocator<Ctxt*> >;


  Sharers sharers;
  const CtxtCmp& ctxtCmp;


  OptimNhoodItem (Lockable* l, const CtxtCmp& ctxtCmp): Base (l), ctxtCmp (ctxtCmp) {}

  bool add (Ctxt* ctxt) {

    if (sharers.empty ()) { // empty 
      sharers.push_back (ctxt);
      return true;

    } else {

      while (!sharers.empty ()) {
        Ctxt* tail = sharers.back ();
        assert (tail != nullptr);


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
      assert (tail != nullptr);

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
struct OptimContext: public SimpleRuntimeContext {

  using Base = SimpleRuntimeContext;

  using CtxtCmp = ContextComparator<OptimContext, Cmp>;
  using NItem = OptimNhoodItem<OptimContext, CtxtCmp>;
  using NhoodMgr = PtrBasedNhoodMgr<NItem>;
  using NhoodList = typename ContainersWithGAlloc::Vector<NItem*>::type;
  using ChildList = typename ContainersWithGAlloc::Vector<OptimContext*>::type;

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


  explicit OptimContext (const T& x, const ContextState& s, size_t step, Exec& exec)
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

      OptimContext* child = exec.push (*i);
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

  }

};


template <typename T, typename Cmp, typename NhFunc, typename OpFunc> 
class OptimParaMeterExecutor: private boost::noncopyable {

  friend class OptimContext<T, Cmp, OptimParaMeterExecutor>;
  using Ctxt = OptimContext<T, Cmp, OptimParaMeterExecutor>;
  using NhoodMgr = typename Ctxt::NhoodMgr;
  using CtxtCmp = typename Ctxt::CtxtCmp;
  using NItemFactory = typename Ctxt::NItem::Factory;

  using CommitQ = Galois::MinHeap<Ctxt*, CtxtCmp>;
  using PendingQ = Galois::MinHeap<Ctxt*, CtxtCmp>;

  using CtxtAlloc = Runtime::MM::FixedSizeAllocator<Ctxt>;
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
      nhmgr (nitemFactory),
      rob (ctxtCmp)
  {
    currPending = new PendingQ (ctxtCmp);
    nextPending = new PendingQ (ctxtCmp);
    steps = 0;

    if (loopname == nullptr) { loopname = "NULL"; }
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
    assert (ctxt != nullptr);
    assert (steps >= 0);
    ctxtAlloc.construct (ctxt, x, ContextState::UNSCHEDULED, steps, *this);
    nextPending->push (ctxt);

    return ctxt;
  }

  void push_abort (Ctxt* ctxt) {
    assert (ctxt != nullptr);
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

        // assert (ctxt != nullptr);
        if (ctxt == nullptr) {
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
          rob.push (ctxt);
          assert (std::find (rob.begin (), rob.end (), ctxt) != rob.end ());

          // for debugging
          // if (!rob.find (ctxt)) {
            // Ctxt* found = *(std::find (rob.begin (), rob.end (), ctxt));
            // dbg::debug (ctxt, " add to rob failed. found:  ", found, " with active: ", found->active);
          // }

        } else {
          assert (ctxt->hasState (ContextState::ABORT_SELF));
          dbg::debug (ctxt, " aborting self");
          ctxt->doAbort ();
        }
      }

      size_t numCommitted = clearROB ();
      totalCommits += numCommitted;

      if (numCommitted == 0) {
        dbg::debug ("head of rob: ", rob.top (),  "  with item: ", rob.top ()->active);

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
      s.dump (getStatsFile (), loopname);
    }

    closeStatsFile ();
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

    size_t numCommitted = 0;

    while (!rob.empty ()) {
      Ctxt* head = rob.top ();
      dbg::debug ("head of rob: ", head);

      if (head->hasState (ContextState::ABORTED_CHILD)
          || head->hasState (ContextState::COMMIT_DONE)
          || head->hasState (ContextState::UNSCHEDULED)) {
        dbg::debug ("head of rob removed because aborted : ", head);
        rob.pop ();
        continue;

      } else if (head->hasState (ContextState::READY_TO_COMMIT)) {
        assert (currPending->empty ());

        dbg::debug ("head of rob ready to commit : ", head);
        bool earliest = false;
        if (!nextPending->empty ()) {
          earliest = !ctxtCmp (nextPending->top (), head);

        } else {
          earliest = true;
        }

        if (earliest) {

          head->doCommit ();
          Ctxt* t = rob.pop ();
          assert (t == head);

          const size_t s = head->step;
          assert (s < execRcrds.size ());
          ++execRcrds[s].parallelism;
          numCommitted += 1;
          freeCtxt (t);


        } else {
          dbg::debug ("head of rob could not commit : ", head);
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
void for_each_ordered_optim (const R& range, Cmp cmp, NhFunc nhFunc, OpFunc opFunc, const char* loopname=0) {

  using T = typename R::value_type;

  ParaMeter::OptimParaMeterExecutor<T, Cmp, NhFunc, OpFunc> exec (cmp, nhFunc, opFunc, loopname);

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


#endif // GALOIS_RUNTIME_ORDERED_SPECULATION_H
