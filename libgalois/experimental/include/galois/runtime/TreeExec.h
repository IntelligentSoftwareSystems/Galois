/** Galois scheduler for tree shaped DAGs e.g. tree traversals and divide-and-conquer * algorithms -*- C++ -*-
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
 * @author M. Amber Hassaan <ahassaan@ices.utexas.edu>
 */
#ifndef GALOIS_RUNTIME_TREEEXEC_H
#define GALOIS_RUNTIME_TREEEXEC_H

#include "galois/GaloisForwardDecl.h"
#include "galois/optional.h"
#include "galois/Traits.h"
#include "galois/runtime/Executor_DoAll.h"
#include "galois/runtime/Support.h"
#include "galois/Substrate/Termination.h"
#include "galois/runtime/UserContextAccess.h"
#include "galois/gIO.h"
#include "galois/runtime/Mem.h"
#include "galois/worklists/AltChunked.h"
#include "galois/worklists/ExternalReference.h"

#include <atomic>

namespace galois {
namespace runtime {

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

    void incNumChild (void) { ++numChild; }

    unsigned getNumChild (void) const { return numChild; }

    Task* getParent () { return parent; }

    const T& getElem () const { return elem; }
    T& getElem () { return elem; }

    const Mode& getMode () const { return mode; }
    bool hasMode (const Mode& m) const { return mode == m; }
    void setMode (const Mode& m) { mode = m; }
  };

  static const unsigned CHUNK_SIZE = 2;
  typedef galois::worklists::AltChunkedLIFO<CHUNK_SIZE, Task*> WL_ty;
  typedef FixedSizeAllocator<Task> TaskAlloc;
  typedef UserContextAccess<T> UserCtx;
  typedef substrate::PerThreadStorage<UserCtx> PerThreadUserCtx;

  // template <typename C>
  // class CtxWrapper: boost::noncopyable {
    // TreeExecutorTwoFunc* executor;
    // C& ctx;
    // Task* parent;
    // size_t numChild;
//
  // public:
    // CtxWrapper (TreeExecutorTwoFunc* executor, C& ctx, Task* parent):
      // boost::noncopyable (),
      // executor (executor),
      // ctx (ctx),
      // parent (parent),
      // numChild (0)
    // {}
//
    // void spawn (const T& elem) {
      // Task* child = executor->spawn (elem, parent);
      // ctx.push (child);
      // ++numChild;
    // }
//
    // size_t getNumChild (void) const { return numChild; }
//
//
    // void sync (void) const {}
  // };


  class CtxWrapper: boost::noncopyable {
    TreeExecutorTwoFunc& executor;
    Task* parent;

  public:
    CtxWrapper (TreeExecutorTwoFunc& executor, Task* parent)
      : boost::noncopyable (), executor (executor), parent (parent)
    {}

    void spawn (const T& elem) {
      executor.spawn (elem, parent);
    }

  };

  struct ApplyOperatorSinglePhase {
    typedef int tt_does_not_need_aborts;
    typedef double tt_does_not_need_push;

    TreeExecutorTwoFunc& executor;

    template <typename C>
    void operator () (Task* t, C& ctx) {

      if (t->hasMode (Task::DIVIDE)) {
        // CtxWrapper<C> uctx {executor, ctx, t};
        CtxWrapper uctx {executor, t};
        executor.divFunc (t->getElem (), uctx);

        // if (uctx.getNumChild () == 0) {
          // t->setMode (Task::CONQUER);
//
        // } else {
          // t->setNumChildren (uctx.getNumChild ());
        // }


        if (t->getNumChild () == 0) {
          t->setMode (Task::CONQUER);
        }
      } // end outer if

      if (t->hasMode (Task::CONQUER)) {
        executor.conqFunc (t->getElem());

        Task* parent = t->getParent ();
        if (parent != nullptr && parent->processedLastChild()) {
          parent->setMode (Task::CONQUER);
          // ctx.push (parent);
          executor.push (parent);
        }

        // task can be deallocated now
        executor.taskAlloc.destroy (t);
        executor.taskAlloc.deallocate (t, 1);
      }

    }
  };

  void push (Task* t) {
    workList.push (t);
  }

  Task* spawn (const T& elem, Task* parent) {
    parent->incNumChild ();

    Task* child = taskAlloc.allocate (1);
    assert (child != nullptr);
    taskAlloc.construct (child, elem, parent, Task::DIVIDE);

    workList.push (child);
    return child;
  }

  DivFunc divFunc;
  ConqFunc conqFunc;
  std::string loopname;
  TaskAlloc taskAlloc;
  WL_ty workList;

public:
  TreeExecutorTwoFunc (const DivFunc& divFunc, const ConqFunc& conqFunc, const char* loopname)
    :
      divFunc (divFunc),
      conqFunc (conqFunc),
      loopname (loopname)
  {}

  template <typename R>
  void execute (const R& initRange) {

    galois::do_all_choice(initRange,
        [this] (const T& item) {
          Task* initTask = taskAlloc.allocate (1);
          taskAlloc.construct (initTask, item, nullptr, Task::DIVIDE);
          push (initTask);
        },
        std::make_tuple(
          galois::loopname("create_initial_tasks"),
          galois::chunk_size<4>()));

    typedef worklists::ExternalReference<WL_ty> WL;
    typename WL::value_type* it = nullptr;

    galois::for_each (it, it,
        ApplyOperatorSinglePhase {*this},
        galois::loopname(loopname.c_str()),
                      galois::wl<WL>(std::ref(workList)));

    // initialTasks deleted in ApplyOperatorSinglePhase,
  }
};


template <typename I, typename DivFunc, typename ConqFunc>
void for_each_ordered_tree (I beg, I end, const DivFunc& divFunc, const ConqFunc& conqFunc, const char* loopname) {

  typedef typename std::iterator_traits<I>::value_type T;

  TreeExecutorTwoFunc<T, DivFunc, ConqFunc, false> executor {divFunc, conqFunc, loopname};
  executor.execute (galois::runtime::makeStandardRange (beg, end));
}

template <typename T, typename DivFunc, typename ConqFunc>
void for_each_ordered_tree (const T& rootItem, const DivFunc& divFunc, const ConqFunc& conqFunc, const char* loopname=nullptr) {

  T tmp[] = { rootItem };
  for_each_ordered_tree (&tmp[0], &tmp[1], divFunc, conqFunc, loopname);
}

template <typename F>
class TreeExecStack {

protected:

  struct Task {
    GALOIS_ATTRIBUTE_ALIGN_CACHE_LINE std::atomic<unsigned> numChild;
    Task* parent;

    explicit Task (Task* parent): numChild (0), parent (parent)
    {}
  };

  typedef std::pair<Task*, F*> WorkItem;
  static const unsigned CHUNK_SIZE = 2;
  typedef worklists::AltChunkedLIFO<CHUNK_SIZE, WorkItem> WL_ty;
  // typedef worklists::AltChunkedFIFO<CHUNK_SIZE, WorkItem> WL_ty;

public:

  class CtxWrapper : private boost::noncopyable {
    TreeExecStack* executor;
    Task* parent;

  public:
    CtxWrapper (TreeExecStack* executor, Task* parent)
      : boost::noncopyable (), executor (executor), parent (parent)
    {}

    void spawn (F& f) {
      executor->spawn (f, parent);
    }

    void sync () {
      executor->syncLoop (*this);
    }

    unsigned getNumChild (void) const {
      return parent->numChild;
    }
  };

protected:
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
      reportStat_Tsum(loopname, "Pushes", stat_pushes);
      reportStat_Tsum(loopname, "Iterations", stat_iterations);
    }
  };

  void spawn (F& f, Task* parent) {
    ++(parent->numChild);
    push (WorkItem (parent, &f));
  }

  void push (const WorkItem& p) {
    workList.push (p);
    PerThreadData& ptd = *(perThreadData.getLocal ());
    ++(ptd.stat_pushes);
  }

  void syncLoop (CtxWrapper& ctx) {
    while (ctx.getNumChild () != 0) {
      applyOperatorRecursive ();
    }
  }

  void applyOperatorRecursive () {
    galois::optional<WorkItem> funcNparent = workList.pop ();

    if (funcNparent) {
      PerThreadData& ptd = *(perThreadData.getLocal ());
      ++(ptd.stat_iterations);

      if (!ptd.didWork) {
        ptd.didWork = true;
      }

      Task task {funcNparent->first};

      CtxWrapper ctx {this, &task};

      funcNparent->second->operator () (ctx);

      Task* parent = funcNparent->first;

      if (parent != nullptr) {
        --(parent->numChild);
      }
    }
  }

  const char* loopname;
  substrate::PerThreadStorage<PerThreadData> perThreadData;
  substrate::TerminationDetection& term;
  WL_ty workList;

public:
  TreeExecStack (const char* loopname):
    loopname (loopname),
    perThreadData (loopname),
    term (substrate::getSystemTermination (activeThreads))
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
      substrate::asmPause (); // Take a breath, let the token propagate
    } while (!term.globalTermination ());

    ptd.reportStats ();
  }

};

template <typename F>
void for_each_ordered_tree_impl (F& initTask, const char* loopname=nullptr) {
  //  assert (initTask != nullptr);

  TreeExecStack<F> e (loopname);

  e.initWork (initTask);

  substrate::getThreadPool().run (galois::getActiveThreads(),
      [&e] () { e.initThread(); },
      std::ref (e));
}
class TreeTaskBase;

typedef TreeExecStack<TreeTaskBase>::CtxWrapper TreeTaskContext;

class TreeTaskBase {
public:
  virtual void operator () (TreeTaskContext& ctx) = 0;
};

template <typename F>
void for_each_ordered_tree (F& initTask, const char* loopname=nullptr) {
  for_each_ordered_tree_impl<F> (initTask, loopname);
}

void for_each_ordered_tree_generic (TreeTaskBase& initTask, const char* loopname=nullptr);
} // end namespace runtime
} // end namespace galois


#endif  // GALOIS_RUNTIME_TREEEXEC_H
