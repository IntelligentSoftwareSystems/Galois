
#include "Galois/config.h"
#include "Galois/Accumulator.h"
#include "Galois/Atomic.h"
#include "Galois/gdeque.h"
#include "Galois/PriorityQueue.h"
#include "Galois/Timer.h"

#include "Galois/optional.h"
#include "Galois/Runtime/Context.h"
#include "Galois/Runtime/DoAll.h"
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
class DivideAndConquerExecutor {

protected:
  template <bool _NEEDS_CHILDREN, typename Task>
  struct TreeExecPolicy {
    typedef Galois::gdeque<Task*, 8> ChildList;

    static void addChild (Task* t, Task* child) {
      assert (child != nullptr);
      t->children.push_back (child);
    }

    static void invokeConqFunc (ConqFunc& conqFunc, Task* t) {

      struct GetElem: std::unary_function<Task*, const T&> {
        const T& operator () (Task* t) const {
          return t->getElem ();
        }
      };

      auto beg = boost::make_transform_iterator (t->children.begin (), GetElem ());
      auto end = boost::make_transform_iterator (t->children.end (), GetElem ());

      conqFunc (t->getElem (), beg, end);
    }

    template <typename TaskAlloc>
    static void freeTask (TaskAlloc& taskAlloc, Task* t) {
      // remove the children upon completion, but not the task itself as it 
      // will be accessed by its parent
      for (auto i = t->children.begin (), i_end = t->children.end (); i != i_end; ++i) {
        taskAlloc.destroy (*i);
        taskAlloc.deallocate (*i, 1);
        *i = nullptr;
      }
    };

  };

  template <typename Task>
  struct TreeExecPolicy<false, Task> {

    // dummy implementation
    struct ChildList {
    };

    static void addChild (Task* t, Task* child) {}

    static void invokeConqFunc (ConqFunc& conqFunc, Task* t) {
      conqFunc (t->getElem ());
    }

    template <typename TaskAlloc>
    static void freeTask (TaskAlloc& taskAlloc, Task* t) {
      // if (t->getParent () != nullptr) { // non-root task
        taskAlloc.destroy (t);
        taskAlloc.deallocate (t, 1);
      // }
    }
  };


  class Task {
  public:
    enum Mode { DIVIDE, CONQUER };

  protected:
    friend struct TreeExecPolicy<NEEDS_CHILDREN, Task>;

    GALOIS_ATTRIBUTE_ALIGN_CACHE_LINE Mode mode;
    T elem;
    Task* parent;
    std::atomic<unsigned> numChild;

    typename TreeExecPolicy<NEEDS_CHILDREN, Task>::ChildList children;


    // std::atomic<unsigned> numChild;


  public:
    Task (const T& a, Task* p, const Mode& m)
      : mode (m), elem (a), parent (p), numChild (0)    
    {}

    void setNumChildren (unsigned c) {
      assert (c > 0);
      numChild = c;
    }

    bool removedLastChild (void) {
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

  struct ApplyOperatorSinglePhase {
    typedef int tt_does_not_need_aborts;

    TaskAlloc& taskAlloc;
    PerThreadUserCtx& userCtxts;
    DivFunc& divFunc;
    ConqFunc& conqFunc;

    template <typename C>
    void operator () (Task* t, C& ctx) {

      if (t->hasMode (Task::DIVIDE)) {
        UserCtx& uctx = *userCtxts.getLocal ();
        uctx.reset ();
        divFunc (t->getElem (), uctx);

        bool hasChildren = uctx.getPushBuffer().begin () != uctx.getPushBuffer ().end ();

        if (hasChildren) {
          ptrdiff_t numChild = std::distance (uctx.getPushBuffer ().begin (), uctx.getPushBuffer ().end ());

          t->setNumChildren (numChild);

          unsigned i = 0;
          for (auto c = uctx.getPushBuffer ().begin (), c_end = uctx.getPushBuffer ().end (); 
              c != c_end; ++c, ++i) {

            Task* child = taskAlloc.allocate (1); 
            assert (child != nullptr);
            taskAlloc.construct (child, *c, t, Task::DIVIDE);
            TreeExecPolicy<NEEDS_CHILDREN, Task>::addChild (t, child);
            ctx.push (child);
          }
        } else { 
          // no children, so t is a leaf task
          t->setMode (Task::CONQUER);
        }
      } // end outer if

      if (t->hasMode (Task::CONQUER)) {
        TreeExecPolicy<NEEDS_CHILDREN, Task>::invokeConqFunc (conqFunc, t);
        // conqFunc (t->getElem());

        Task* parent = t->getParent ();
        if (parent != nullptr && parent->removedLastChild()) {
          parent->setMode (Task::CONQUER);
          ctx.push (parent);
        }

        // task can be deallocated now
        // taskAlloc.destroy (t);
        // taskAlloc.deallocate (t, 1);
        TreeExecPolicy<NEEDS_CHILDREN, Task>::freeTask (taskAlloc, t);
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

  DivideAndConquerExecutor (const DivFunc& divFunc, const ConqFunc& conqFunc, const char* loopname)
    : 
      divFunc (divFunc),
      conqFunc (conqFunc),
      loopname (loopname)
  {}

  void execute_no_children (const T& initItem) {
    TaskAlloc taskAlloc;

    Task* initTask = taskAlloc.allocate (1); 
    taskAlloc.construct (initTask, initItem, nullptr, Task::DIVIDE);

    Task* a[] = {initTask};

    Galois::Runtime::for_each_impl<WL_ty> (
        makeStandardRange (&a[0], &a[1]), 
        ApplyOperatorSinglePhase {taskAlloc, userCtxts, divFunc, conqFunc},
        loopname.c_str ());
  }

  T execute_with_children (const T& initItem) {

    // typedef Galois::WorkList::dChunkedLIFO<CHUNK_SIZE, Task*> WL_ty;

    TaskAlloc taskAlloc;

    Task* initTask = taskAlloc.allocate (1); 
    taskAlloc.construct (initTask, initItem, nullptr, Task::DIVIDE);

    Task* a[] = {initTask};

    Galois::Runtime::for_each_impl<WL_ty> (
        makeStandardRange (&a[0], &a[1]), 
        ApplyOperatorSinglePhase {taskAlloc, userCtxts, divFunc, conqFunc},
        loopname.c_str ());

    T result = initTask->getElem ();

    taskAlloc.destroy (initTask);
    taskAlloc.deallocate (initTask, 1);

    return result;
  }

};
template <typename T, typename DivFunc, typename ConqFunc>
void for_each_ordered_tree (const T& initItem, const DivFunc& divFunc, const ConqFunc& conqFunc, const char* loopname=nullptr) {

  DivideAndConquerExecutor<T, DivFunc, ConqFunc, false> executor (divFunc, conqFunc, loopname);
  executor.execute_no_children (initItem);
}

struct TreeExecNeedsChildren {};

template <typename T, typename DivFunc, typename ConqFunc>
T for_each_ordered_tree (const T& initItem, const DivFunc& divFunc, const ConqFunc& conqFunc, TreeExecNeedsChildren, const char* loopname=nullptr) {

  DivideAndConquerExecutor<T, DivFunc, ConqFunc, true> executor (divFunc, conqFunc, loopname);
  return executor.execute_with_children (initItem);

}

template <typename F>
class TreeExec {

protected:

  struct Task {
    std::atomic<unsigned> numChild;
    Task* parent;

    Task (Task* parent): numChild (0), parent (parent)
    {}
  };

  typedef std::pair<Task*, F> WorkItem;
  static const unsigned CHUNK_SIZE = 2;
  typedef WorkList::AltChunkedLIFO<CHUNK_SIZE, WorkItem> WL_ty;

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

    ~PerThreadData (void) {
      reportStat(loopname, "Pushes", stat_pushes);
      reportStat(loopname, "Iterations", stat_iterations);
    }
  };

  struct CtxWrapper {
    Task* task;
    TreeExec& exec;

    void spawn (const F& f) {
      ++(task->numChild);
      exec.push (WorkItem (task, f));
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

      funcNparent->second (ctx);

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

  void initWork (const F& initTask) {
    push (WorkItem (nullptr, initTask));
  }

  void operator () (void) {
    do {
      PerThreadData& ptd = *(perThreadData.getLocal ());
      ptd.didWork = false;

      applyOperatorRecursive ();

      term.localTermination (ptd.didWork);
      LL::asmPause (); // Take a breath, let the token propagate
    } while (!term.globalTermination ());
  }

};

template <typename F> 
void for_each_ordered_tree (const F& initTask, const char* loopname=nullptr) {

  TreeExec<F> e (loopname);

  e.initWork (initTask);

  getSystemThreadPool ().run (Galois::getActiveThreads (),
      [&e] (void) { e.initThread (); },
      std::ref (e));
}

class TreeExecGeneric {
  typedef std::function<void (void)> F;



  struct Task {
    std::atomic<unsigned> numChild;
    Task* parent;

    Task (Task* parent): numChild (0), parent (parent)
    {}
  };

  typedef std::pair<Task*, F> WorkItem;
  static const unsigned CHUNK_SIZE = 2;
  typedef WorkList::AltChunkedLIFO<CHUNK_SIZE, WorkItem> WL_ty;

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

    ~PerThreadData (void) {
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

  void push (const F& f) {
    PerThreadData& ptd = *(perThreadData.getLocal ());
    Task* t = ptd.currTask;
    if (t != nullptr) {
      ++(t->numChild);
    }
    workList.push (WorkItem (t, f));
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

      funcNparent->second ();

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
    do {
      PerThreadData& ptd = *(perThreadData.getLocal ());
      ptd.didWork = false;

      applyOperatorRecursive ();

      term.localTermination (ptd.didWork);
      LL::asmPause (); // Take a breath, let the token propagate
    } while (!term.globalTermination ());
  }
};

TreeExecGeneric& getTreeExecutor (void);

void setTreeExecutor (TreeExecGeneric* t);

void spawn (std::function<void (void)> f);

void sync (void);

void for_each_ordered_tree_generic (std::function<void (void)> initTask, const char* loopname=nullptr);



} // end namespace Runtime
} // end namespace Galois
