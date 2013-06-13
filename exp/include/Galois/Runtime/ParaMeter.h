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
#ifndef GALOIS_RUNTIME_PARAMETER_H
#define GALOIS_RUNTIME_PARAMETER_H

#include "Galois/TypeTraits.h"
#include "Galois/Mem.h"

#include "Galois/Runtime/Context.h"
#include "Galois/Runtime/ForEachTraits.h"
#include "Galois/Runtime/ParallelWork.h"
#include "Galois/Runtime/Support.h"
#include "Galois/Runtime/Termination.h"
#include "Galois/Runtime/ThreadPool.h"
#include "Galois/WorkList/Fifo.h"
#include "Galois/Runtime/ll/gio.h"

#include "llvm/Support/CommandLine.h"

#include <deque>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <cstdio>

namespace Galois {
namespace WorkList {
template<class ContainerTy = FIFO<>,class T=int>
class ParaMeter: private boost::noncopyable {
};
}

namespace Runtime {

namespace ParaMeterInit {
  void init();
  const char* getStatsFileName();
}

namespace {

// Single ParaMeter stats file per run of an app
// which includes all instances of for_each loops
// run with ParaMeter Executor
template<class ContainerTy,class T, class FunctionTy>
class ForEachWork<Galois::WorkList::ParaMeter<ContainerTy>,T,FunctionTy> {
  typedef T value_type;
  typedef Galois::Runtime::UserContextAccess<value_type> UserContextTy;
  typedef typename ContainerTy::template retype<value_type>::WL WorkListTy;

  struct StepStats {
    size_t step;
    size_t parallelism;
    size_t workListSize;

    void dump(FILE* out, const char* loopname) const {
      if (out) {
        fprintf(out, "%s, %zu, %zu, %zu\n", loopname, step, parallelism, workListSize);
      }
    }
  };

  struct IterationContext {
    UserContextTy facing;
    SimpleRuntimeContext cnx;

    void resetUserCtx() {
      if (ForEachTraits<FunctionTy>::NeedsPIA) {
        facing.resetAlloc();
      }

      if (ForEachTraits<FunctionTy>::NeedsPush) {
        facing.getPushBuffer().clear();
      }

      if (ForEachTraits<FunctionTy>::NeedsBreak) {
        // TODO: no support for breaks yet
        // facing.resetBreak();
      }
    }
  };

  typedef std::deque<IterationContext*> IterQueue;

  class ParaMeterWorkList: private boost::noncopyable {
    WorkListTy* curr;
    WorkListTy* next;

    void copyWL(WorkListTy& wl) {
      for (Galois::optional<value_type> item = wl.pop(); item; item = wl.pop()) {
        curr->push(*item);
      }
    }
  public:
    explicit ParaMeterWorkList(WorkListTy& wl) {
      curr = new WorkListTy();
      next = new WorkListTy();

      // XXX: workList must be empty after for_each finishes
      copyWL(wl);
    }

    ParaMeterWorkList() {
      curr = new WorkListTy ();
      next = new WorkListTy ();
    }

    ~ParaMeterWorkList() {
      delete curr;
      delete next;
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
  };


  void go() {
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

      Galois::optional<value_type> item;
      while ((item = workList.getCurr().pop())) {
        IterationContext& it = newIteration();

        bool doabort = false;
        try {
          function(*item, it.facing.data ());

        } catch (ConflictFlag flag) {
          clearConflictLock();
          switch (flag) {
            case Galois::Runtime::CONFLICT:
              doabort = true;
              break;

            case Galois::Runtime::BREAK:
              GALOIS_DIE("can't handle breaks yet");
              break;

            default:
              abort ();
          }
        }

        if (doabort) {
          abortIteration(it, *item);

        } else {
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
        GALOIS_DIE("no progress made in step ", currStep);
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
      StepStats stat;
      stat.step = currStep;
      stat.parallelism = numActivities;
      stat.workListSize = numIter;

      finishStep(stat);
      ++currStep;
    }

    finishLoop();
  }
    
  void beginLoop() { }

  void finishStep(const StepStats& stat) {
    allSteps.push_back(stat);
    stat.dump(pstatsFile, loopname);
    workList.switchWorkLists();
    setThreadContext(NULL);
  }

  void finishLoop() {
    setThreadContext(NULL);
  }

  IterationContext& newIteration() const {
    IterationContext* it = new IterationContext();
    
    it->resetUserCtx();
    setThreadContext(&(it->cnx));

    return *it;
  }

  unsigned retireIteration(IterationContext& it, const bool abort) const {

    it.resetUserCtx ();

    unsigned numLocks = 0;
    if (abort) {
      numLocks = it.cnx.cancel_iteration();
    } else {
      numLocks = it.cnx.commit_iteration();
    }

    delete &it;

    return numLocks;
  }

  unsigned abortIteration(IterationContext& it, value_type& item) {
    clearConflictLock();
    workList.getNext().push(item);
    return retireIteration(it, true);
  }

  unsigned commitIteration(IterationContext& it) {
    if (ForEachTraits<FunctionTy>::NeedsPush) {
      for (typename UserContextTy::pushBufferTy::iterator a = it.facing.getPushBuffer().begin(),
          ea = it.facing.getPushBuffer().end(); a != ea; ++a) {
        workList.getNext().push(*a);
      }
    }

    return retireIteration(it, false);
  }

public:
  ForEachWork(FunctionTy& f, const char* ln): function(f), loopname(ln) { }

  ForEachWork (WorkListTy& wl, FunctionTy& f, const char* ln) 
    : workList (wl), function (f), loopname (ln) 
  {}

  template<typename IterTy>
  bool AddInitialWork(IterTy b, IterTy e) {
    workList.getCurr().push_initial(b, e);
    return true;
  }

  void initThread() {}

  void operator()() {
    ParaMeterInit::init();
    pstatsFile = fopen(ParaMeterInit::getStatsFileName(), "a"); // open in append mode
    go();
    fclose(pstatsFile);
    pstatsFile = NULL;
  }

private:
  ParaMeterWorkList workList;
  FunctionTy function;
  const char* loopname;
  FILE* pstatsFile;

  IterQueue commitQueue;
  // XXX: may turn out to be unnecessary
  std::vector<StepStats> allSteps;
};


} // end namespace
}
}

#else
#warning Reincluding ParaMeter
#endif // GALOIS_RUNTIME_PARAMETER_H


