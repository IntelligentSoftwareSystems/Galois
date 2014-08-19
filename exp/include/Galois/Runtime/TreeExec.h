
#include "Galois/config.h"
#include "Galois/Accumulator.h"
#include "Galois/Atomic.h"
#include "Galois/gdeque.h"
#include "Galois/PriorityQueue.h"
#include "Galois/Timer.h"

#include "Galois/optional.h"
#include "Galois/Runtime/Context.h"
#include "Galois/Runtime/DoAll.h"
#include "Galois/Runtime/ForEachTraits.h"
#include "Galois/Runtime/LCordered.h"
#include "Galois/Runtime/ParallelWork.h"
#include "Galois/Runtime/PerThreadWorkList.h"
#include "Galois/Runtime/Support.h"
#include "Galois/Runtime/Termination.h"
#include "Galois/Runtime/ll/gio.h"
#include "Galois/Runtime/ll/ThreadRWlock.h"
#include "Galois/Runtime/mm/Mem.h"

#include "llvm/Support/CommandLine.h"


#include <atomic>

namespace Galois {
namespace Runtime {

template <typename T, typename DivFunc, typename ConqFunc, bool NEEDS_CHILDREN>
class TreeExecutorTwoFunc {

protected:

  class Task {
  public:
    enum Mode { DIVIDE, CONQUER };

  protected:

    GALOIS_ATTRIBUTE_ALIGN_CACHE_LINE Mode mode;
    T elem;
    Task* parent;
    std::atomic<unsigned> numChild;


    // std::atomic<unsigned> numChild;


  public:
    Task (const T& a, Task* p, const Mode& m)
      : mode (m), elem (a), parent (p), numChild (0)    
    {}

    void setNumChildren (unsigned c) {
      assert (c > 0);
      numChild = c;
    }

    bool processedLastChild (void) {
      assert (numChild > 0);
      return (--numChild == 0);
    }

    Task* getParent () { return parent; }

    const T& getElem () const { return elem; }
    T& getElem () { return elem; }

    const Mode& getMode () const { return mode; }
    bool hasMode (const Mode& m) const { return mode == m; }
    void setMode (const Mode& m) { mode = m; }

  };



  typedef MM::FixedSizeAllocator<Task> TaskAlloc;
  typedef UserContextAccess<T> UserCtx;
  typedef PerThreadStorage<UserCtx> PerThreadUserCtx;


  template <typename C>
  class CtxWrapper: boost::noncopyable {
  public:
    TaskAlloc& taskAlloc;
    C& ctx;
    Task* parent;
    size_t numChild;

    CtxWrapper (TaskAlloc& taskAlloc, C& ctx, Task* parent):
      boost::noncopyable (),
      taskAlloc (taskAlloc),
      ctx (ctx),
      parent (parent),
      numChild (0)
    {}

    void spawn (const T& elem) {
      Task* child = taskAlloc.allocate (1);
      assert (child != nullptr);
      taskAlloc.construct (child, elem, parent, Task::DIVIDE);
      ctx.push (child);
      ++numChild;
    }

    void sync (void) {}
  };


  struct ApplyOperatorSinglePhase {
    typedef int tt_does_not_need_aborts;

    TaskAlloc& taskAlloc;
    PerThreadUserCtx& userCtxts;
    DivFunc& divFunc;
    ConqFunc& conqFunc;

    template <typename C>
    void operator () (Task* t, C& ctx) {

      if (t->hasMode (Task::DIVIDE)) {
        // UserCtx& uctx = *userCtxts.getLocal ();
        // uctx.reset ();
        CtxWrapper<C> uctx {taskAlloc, ctx, t};
        divFunc (t->getElem (), uctx);

        if (uctx.numChild == 0) {
          t->setMode (Task::CONQUER);

        } else {
          t->setNumChildren (uctx.numChild);
        }

      } // end outer if

      if (t->hasMode (Task::CONQUER)) {
        conqFunc (t->getElem());

        Task* parent = t->getParent ();
        if (parent != nullptr && parent->processedLastChild()) {
          parent->setMode (Task::CONQUER);
          ctx.push (parent);
        }

        // task can be deallocated now
        taskAlloc.destroy (t);
        taskAlloc.deallocate (t, 1);
      }

    }
  };



  DivFunc divFunc;
  ConqFunc conqFunc;
  std::string loopname;
  PerThreadUserCtx userCtxts;

  static const unsigned CHUNK_SIZE = 2;
  typedef Galois::WorkList::AltChunkedLIFO<CHUNK_SIZE, Task*> WL_ty;

public:

  TreeExecutorTwoFunc (const DivFunc& divFunc, const ConqFunc& conqFunc, const char* loopname)
    : 
      divFunc (divFunc),
      conqFunc (conqFunc),
      loopname (loopname)
  {}

  void execute (const T& initItem) {
    TaskAlloc taskAlloc;

    Task* initTask = taskAlloc.allocate (1); 
    taskAlloc.construct (initTask, initItem, nullptr, Task::DIVIDE);

    Task* a[] = {initTask};

    Galois::Runtime::for_each_impl<WL_ty> (
        makeStandardRange (&a[0], &a[1]), 
        ApplyOperatorSinglePhase {taskAlloc, userCtxts, divFunc, conqFunc},
        loopname.c_str ());

    // initTask deleted in ApplyOperatorSinglePhase,
  }


};
template <typename T, typename DivFunc, typename ConqFunc>
void for_each_ordered_tree (const T& initItem, const DivFunc& divFunc, const ConqFunc& conqFunc, const char* loopname=nullptr) {

  TreeExecutorTwoFunc<T, DivFunc, ConqFunc, false> executor (divFunc, conqFunc, loopname);
  executor.execute (initItem);
}

template <typename F>
class TreeExec {

protected:

  struct Task {
    GALOIS_ATTRIBUTE_ALIGN_CACHE_LINE std::atomic<unsigned> numChild;
    Task* parent;

    Task (Task* parent): numChild (0), parent (parent)
    {}
  };

  typedef std::pair<Task*, F*> WorkItem;
  static const unsigned CHUNK_SIZE = 1;
  typedef WorkList::AltChunkedLIFO<CHUNK_SIZE, WorkItem> WL_ty;
  // typedef WorkList::AltChunkedFIFO<CHUNK_SIZE, WorkItem> WL_ty;

  struct PerThreadData {
    // most frequently accessed members first
    size_t stat_iterations;
    size_t stat_pushes;
    bool didWork;
    const char* loopname;

    PerThreadData (const char* loopname): 
      stat_iterations (0), 
      stat_pushes (0) ,
      didWork (false), 
      loopname (loopname)

    {}

    void reportStats (void) {
      reportStat(loopname, "Pushes", stat_pushes);
      reportStat(loopname, "Iterations", stat_iterations);
    }
  };

  struct CtxWrapper {
    Task* task;
    TreeExec& exec;

    void spawn (F& f) {
      ++(task->numChild);
      exec.push (WorkItem (task, &f));
    }

    void sync () {
      exec.syncLoop (*this);
    }
  };

  void push (const WorkItem& p) {
    workList.push (p);
    PerThreadData& ptd = *(perThreadData.getLocal ());
    ++(ptd.stat_pushes);
  }

  void syncLoop (CtxWrapper& ctx) {
    while (ctx.task->numChild != 0) {
      applyOperatorRecursive ();
    }
  }

  void applyOperatorRecursive () {
    Galois::optional<WorkItem> funcNparent = workList.pop ();

    if (funcNparent) {
      PerThreadData& ptd = *(perThreadData.getLocal ());
      ++(ptd.stat_iterations);

      if (!ptd.didWork) {
        ptd.didWork = true;
      }

      Task task {funcNparent->first};

      CtxWrapper ctx {&task, *this};

      (*funcNparent->second) (ctx);

      Task* parent = funcNparent->first;

      if (parent != nullptr) {
        --(parent->numChild);
      }
    }
  }

  const char* loopname;  
  PerThreadStorage<PerThreadData> perThreadData;
  TerminationDetection& term;
  WL_ty workList;

public:
  TreeExec (const char* loopname): 
    loopname (loopname), 
    perThreadData (loopname), 
    term (getSystemTermination ()) 
  {}

  void initThread (void) {
    term.initializeThread ();
  }

  void initWork (F& initTask) {
    push (WorkItem (nullptr, &initTask));
  }

  void operator () (void) {
    PerThreadData& ptd = *(perThreadData.getLocal ());
    do {
      ptd.didWork = false;

      applyOperatorRecursive ();

      term.localTermination (ptd.didWork);
      LL::asmPause (); // Take a breath, let the token propagate
    } while (!term.globalTermination ());

    ptd.reportStats ();
  }

};

template <typename F> 
void for_each_ordered_tree (F& initTask, const char* loopname=nullptr) {

  TreeExec<F> e (loopname);

  e.initWork (initTask);

  getSystemThreadPool ().run (Galois::getActiveThreads (),
      [&e] (void) { e.initThread (); },
      std::ref (e));
}

class TreeTaskBase {
public:
  virtual void operator () (void) {}
};

class TreeExecGeneric {

  struct Task {
    GALOIS_ATTRIBUTE_ALIGN_CACHE_LINE std::atomic<unsigned> numChild;
    Task* parent;

    Task (Task* parent): numChild (0), parent (parent)
    {}
  };

  typedef std::pair<Task*, TreeTaskBase*> WorkItem;
  static const unsigned CHUNK_SIZE = 2;
  typedef WorkList::AltChunkedLIFO<CHUNK_SIZE, WorkItem> WL_ty;
  // typedef WorkList::AltChunkedFIFO<CHUNK_SIZE, WorkItem> WL_ty;

  struct PerThreadData {
    // most frequently accessed members first
    Task* currTask;
    size_t stat_iterations;
    size_t stat_pushes;
    bool didWork;
    const char* loopname;

    PerThreadData (const char* loopname): 
      currTask (nullptr),
      stat_iterations (0), 
      stat_pushes (0) ,
      didWork (false), 
      loopname (loopname)

    {}

    void reportStats (void) {
      reportStat(loopname, "Pushes", stat_pushes);
      reportStat(loopname, "Iterations", stat_iterations);
    }
  };

  const char* loopname;  
  PerThreadStorage<PerThreadData> perThreadData;
  TerminationDetection& term;
  WL_ty workList;

public:
  TreeExecGeneric (const char* loopname): 
    loopname (loopname), 
    perThreadData (loopname), 
    term (getSystemTermination ()) 
  {}

  void push (TreeTaskBase& f) {
    PerThreadData& ptd = *(perThreadData.getLocal ());
    Task* t = ptd.currTask;
    if (t != nullptr) {
      ++(t->numChild);
    }
    workList.push (WorkItem (t, &f));
    ++(ptd.stat_pushes);
  }

  void syncLoop (void) {
    PerThreadData& ptd = *(perThreadData.getLocal ());
    Task* t= ptd.currTask;

    while (t->numChild != 0) {
      applyOperatorRecursive ();
    }
  }

  void applyOperatorRecursive () {
    Galois::optional<WorkItem> funcNparent = workList.pop ();

    if (funcNparent) {
      PerThreadData& ptd = *(perThreadData.getLocal ());
      ++(ptd.stat_iterations);

      if (!ptd.didWork) {
        ptd.didWork = true;
      }

      Task task {funcNparent->first};

      ptd.currTask = &task;

      funcNparent->second->operator () ();

      Task* parent = funcNparent->first;

      if (parent != nullptr) {
        --(parent->numChild);
      }

      ptd.currTask = nullptr;
    }
  }
  void initThread (void) {
    term.initializeThread ();
  }

  void operator () (void) {
    PerThreadData& ptd = *(perThreadData.getLocal ());
    do {
      ptd.didWork = false;

      applyOperatorRecursive ();

      term.localTermination (ptd.didWork);
      LL::asmPause (); // Take a breath, let the token propagate
    } while (!term.globalTermination ());

    ptd.reportStats ();
  }
};

TreeExecGeneric& getTreeExecutor (void);

void setTreeExecutor (TreeExecGeneric* t);

void spawn (TreeTaskBase& f);

void sync (void);

void for_each_ordered_tree_generic (TreeTaskBase& initTask, const char* loopname=nullptr);



} // end namespace Runtime
} // end namespace Galois
