/** ParaMeter runtime -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2012, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 *
 * @section Description
 *
 * Implementation of ParaMeter runtime
 * Ordered with speculation not supported yet
 *
 * @author Amber Hassaan <ahassaan@ices.utexas.edu>
 */
#ifndef GALOIS_RUNTIME_PARAMETER_H_
#define GALOIS_RUNTIME_PARAMETER_H_

#include "Galois/UserContext.h"
#include "Galois/TypeTraits.h"
#include "Galois/Mem.h"
#include "Galois/Runtime/Context.h"
#include "Galois/Runtime/ForeachTraits.h"
#include "Galois/Runtime/Config.h"
#include "Galois/Runtime/Support.h"
#include "Galois/Runtime/Threads.h"
#include "Galois/Runtime/PerCPU.h"
#include "Galois/Runtime/WorkList.h"
#include "Galois/Runtime/Termination.h"
#include "Galois/Runtime/LoopHooks.h"

#include "llvm/Support/CommandLine.h"

#include <deque>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <cstdio>


namespace GaloisRuntime {


// __attribute__((weak)) llvm::cl::opt<bool> useParaMeter("p", llvm::cl::desc("Use ParaMeter to execute for_each loops"), llvm::cl::init(false));


template <typename WorkListTy, typename FunctionTy>
class ParaMeterExecutor;

// Single ParaMeter stats file per run of an app
// which includes all instances of for_each loops
// run with ParaMeter Executor
//
// basically, from commandline parser calls enableParaMeter
// and we
// - set a flag
// - open a stats file in overwrite mode
// - print stats header
// - close file

// for each for_each loop, we create an instace of ParaMeterExecutor
// which
// - opens stats file in append mode
// - prints stats
// - closes file when loop finishes
class ParaMeter: private boost::noncopyable {
private:
  template <typename WorkListTy, typename FunctionTy>
  friend class ParaMeterExecutor;

  struct StepStats {
    size_t step;
    size_t availParallelism;
    size_t workListSize;

    void dump(FILE* out, const char* loopname) const {
      if (out) {
        fprintf(out, "%s, %zu, %zu, %zu\n", loopname, step, availParallelism, workListSize);
      }
    }
  };
    
  static void printHeader(FILE* out) {
    fprintf(out, "LOOPNAME, STEP, PARALLELISM, WORKLIST_SIZE\n");
  }

  static const char* getStatsFileName();

  static void init ();

public:

  template <typename WLTy, typename IterTy, typename Func>
  static void for_each_impl(IterTy b, IterTy e, Func func, const char* loopname) {

    typedef typename WLTy::template retype<typename std::iterator_traits<IterTy>::value_type>::WL ActualWLTy;

    ParaMeter::init ();

    FILE* statsFH = fopen (ParaMeter::getStatsFileName (), "a"); // open in append mode
    
    ParaMeterExecutor<ActualWLTy, Func> executor(func, statsFH, loopname);
    executor.addInitialWork(b, e);
    executor.run();

    fclose (statsFH);

  }

  // TODO: need to empty the workList passed (i.e. the parameter wl)
  template <typename WLTy, typename Func> 
  static void for_each_impl (WLTy& wl, Func func, const char* loopname) {

    ParaMeter::init ();

    FILE* statsFH = fopen (ParaMeter::getStatsFileName (), "a"); // open in append mode
    ParaMeterExecutor<WLTy, Func> executor (wl, func, statsFH, loopname);

    executor.run ();

    fclose (statsFH);
  }

};


template <typename WorkListTy, typename FunctionTy>
class ParaMeterExecutor: private boost::noncopyable {

private:
  typedef typename WorkListTy::value_type value_type;

  typedef Galois::UserContext<value_type> UserContextTy;

  struct IterationContext {
    UserContextTy facing;
    SimpleRuntimeContext cnx;

    void resetUserCtx () {
      
      if (ForeachTraits<FunctionTy>::NeedsPIA) {
        facing.__resetAlloc ();
      }

      if (ForeachTraits<FunctionTy>::NeedsPush) {
        facing.__getPushBuffer ().clear ();
      }

      if (ForeachTraits<FunctionTy>::NeedsBreak) {
        facing.__resetBreak ();
      }

    }
  };

  typedef std::deque<IterationContext*> IterQueue;

  class ParaMeterWorkList: private boost::noncopyable {
    WorkListTy* curr;
    WorkListTy* next;

  public:

    explicit ParaMeterWorkList (WorkListTy& wl) {
      curr = new WorkListTy ();
      next = new WorkListTy ();

      // XXX: workList must be empty after for_each finishes
      copyWL (wl);
    }

    ParaMeterWorkList () {
      curr = new WorkListTy ();
      next = new WorkListTy ();
    }


    ~ParaMeterWorkList() {
      delete curr;
      curr = NULL;
      delete next;
      next = NULL;
    }

    WorkListTy& getCurr() {
      return *curr;
    }

    WorkListTy& getNext() {
      return *next;
    }

    void switchWorkLists() {
      delete curr;
      curr = next;
      next = new WorkListTy();
    }

  private:
    void copyWL (WorkListTy& wl) {

      for (boost::optional<value_type> item = wl.pop (); item; item = wl.pop ()) {
        curr->push (*item);
      }
    }
  };

  ParaMeterWorkList workList;
  FunctionTy body;
  FILE* pstatsFile;
  const char* loopname;


  IterQueue commitQueue;
  // XXX: may turn out to be unnecessary
  std::vector<ParaMeter::StepStats> allSteps;

public:

  ParaMeterExecutor (FunctionTy _body, FILE* statsFH, const char* _loopname)
  : workList (), body (_body), pstatsFile (statsFH), loopname (_loopname) {

    if (this->loopname == NULL) {
      this->loopname = "foreach";
    }
    
  }

  // TODO: combine with the constructor above, if this constructor persists
  ParaMeterExecutor (WorkListTy& _wl, FunctionTy _body, FILE* statsFH, const char* _loopname)
  : workList (_wl), body (_body), pstatsFile (statsFH), loopname (_loopname) {

    if (this->loopname == NULL) {
      this->loopname = "foreach";
    }
    
  }
  ~ParaMeterExecutor() {
  }

  template <typename Iter>
  bool addInitialWork(Iter b, Iter e) {
    workList.getCurr().push_initial(b,e);
    return true;
  }

  void run() {

    beginLoop();

    size_t currStep = 0;
    bool done = false;
    while (!done) {
      // do initialization for a new step
      //
      // while currWorkList is not empty {
      //  remove an item from current workList
      //  create a new iteration context
      //  run function with iteration and item
      //  if aborted {
      //    add item to nextWorkList
      //  } else {
      //    add iteration to commit queue
      //  }
      // }
      //
      // measure commit queue's size
      // if (size == 0) {
      //  ERROR, no progress?
      // }
      // for each iter in commit queue {
      //  commit iteration
      //    release all locks
      //    add new items to nextWorkList
      //    collect locks/neighborhood stats
      // }
      //
      // log current step
      // move to next step
      //
      size_t numIter = 0;

      for (boost::optional<value_type> item = workList.getCurr().pop();
          item; item = workList.getCurr().pop()) {

        IterationContext& it = newIteration();

        bool doabort = false;
        try {
          body(*item, it.facing);

        } catch (int a) {
          doabort = true;
        }

        if (doabort) {
          abortIteration (it, *item);

        } else {

          if (ForeachTraits<FunctionTy>::NeedsBreak) {
            if (it.facing.__breakHappened()) {
              assert(0 && "ParaMeterExecutor: can't handle breaks yet");
              abort();
            }
          }

          commitQueue.push_back(&it);
        }

        ++numIter;
      }

      if (numIter == 0) {
        done = true;
        continue;
      }

      size_t numActivities = commitQueue.size();

      if (numActivities == 0) {
        std::cerr << "ParaMeterExecutor: no progress made in step=" << currStep << std::endl;
        abort();
      }

      double avgLocks = 0;
      for (typename IterQueue::iterator i = commitQueue.begin(), ei = commitQueue.end();
          i != ei; ++i) {
        unsigned numLocks = commitIteration(*(*i));
        avgLocks += double(numLocks);
      }

      avgLocks /= numActivities;

      commitQueue.clear();

      // switch worklists
      // dump stats
      ParaMeter::StepStats stat;
      stat.step = currStep;
      stat.availParallelism = numActivities;
      stat.workListSize = numIter;


      finishStep(stat);
      ++currStep;

    } // end while

    finishLoop();
  }
    
private:

  void beginLoop() {
  }

  void finishStep (const ParaMeter::StepStats& stat) {
    allSteps.push_back (stat);
    stat.dump(pstatsFile, loopname);
    workList.switchWorkLists ();
    setThreadContext(NULL);
  }

  void finishLoop() {
    setThreadContext(NULL);
  }

  IterationContext& newIteration () const {
    IterationContext* it = new IterationContext();
    
    it->resetUserCtx ();
    setThreadContext(&(it->cnx));

    return *it;
  }

  unsigned retireIteration (IterationContext& it, const bool abort) const {
    if (ForeachTraits<FunctionTy>::NeedsPush) {
      it.facing.__getPushBuffer().clear();
    }

    if (ForeachTraits<FunctionTy>::NeedsPIA) {
      it.facing.__resetAlloc();
    }

    if (ForeachTraits<FunctionTy>::NeedsBreak) {
      it.facing.__resetBreak();
    }

    unsigned numLocks = 0;
    if (abort) {
      numLocks = it.cnx.cancel_iteration();
    } else {
      numLocks = it.cnx.commit_iteration();
    }

    delete &it;

    return numLocks;
  }

  unsigned abortIteration (IterationContext& it, value_type& item) {
    clearConflictLock();
    workList.getNext().push(item);
    return retireIteration(it, true);
  }

  unsigned commitIteration(IterationContext& it) {
    if (ForeachTraits<FunctionTy>::NeedsPush) {

      for (typename UserContextTy::pushBufferTy::iterator a = it.facing.__getPushBuffer().begin (),
          ea = it.facing.__getPushBuffer().end(); a != ea; ++a) {
        workList.getNext().push(*a);
      }
    }

    return retireIteration(it, false);
  }
};


} // end namespace

#endif // GALOIS_RUNTIME_PARAMETER_H_

