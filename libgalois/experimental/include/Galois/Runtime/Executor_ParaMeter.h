/** ParaMeter runtime -*- C++ -*-
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
 * @section Description
 *
 * Implementation of ParaMeter runtime.  Ordered with speculation not
 * supported yet
 *
 * @author Amber Hassaan <ahassaan@ices.utexas.edu>
 */
#ifndef GALOIS_RUNTIME_EXECUTOR_PARAMETER_H
#define GALOIS_RUNTIME_EXECUTOR_PARAMETER_H

#include "Galois/gtuple.h"
#include "Galois/Accumulator.h"
#include "Galois/Traits.h"
#include "Galois/Mem.h"
#include "Galois/Runtime/Context.h"
#include "Galois/Runtime/Executor_ForEach.h"
#include "Galois/Runtime/Support.h"
#include "Galois/gIO.h"
#include "Galois/WorkList/Simple.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <deque>
#include <vector>

#include "llvm/Support/CommandLine.h"

namespace Galois {
namespace Runtime {

  extern llvm::cl::opt<bool> useParaMeterOpt;

namespace ParaMeter {

struct StepStats {

  const size_t step;
  GAccumulator<size_t> parallelism;
  const size_t workListSize;

  explicit StepStats (size_t _step, size_t _wlsz): step (_step), parallelism (),  workListSize (_wlsz) {}

  explicit StepStats (size_t _step, size_t par, size_t _wlsz): step (_step), parallelism (), workListSize (_wlsz) {
    parallelism += par;
  }

  static void printHeader(FILE* out) {
    fprintf(out, "LOOPNAME, STEP, PARALLELISM, WORKLIST_SIZE\n");
  }

  void dump(FILE* out, const char* loopname) const {
    if (out) {
      fprintf(out, "%s, %zu, %zu, %zu\n", loopname, step, parallelism.reduceRO (), workListSize);
    }
  }
};


// TODO(ddn): Take stats file as a trait or from loopname
FILE* getStatsFile (void);
void closeStatsFile (void);

// Single ParaMeter stats file per run of an app
// which includes all instances of for_each loops
// run with ParaMeter Executor
template<class T, class FunctionTy, class ArgsTy>
class ParaMeterExecutor {
  typedef T value_type;
  typedef typename WorkList::GFIFO<int>::template retype<value_type>::type WorkListTy;

  static const bool needsStats = !exists_by_supertype<does_not_need_stats_tag, ArgsTy>::value;
  static const bool needsPush = !exists_by_supertype<does_not_need_push_tag, ArgsTy>::value;
  static const bool needsAborts = !exists_by_supertype<does_not_need_aborts_tag, ArgsTy>::value;
  static const bool needsPia = exists_by_supertype<needs_per_iter_alloc_tag, ArgsTy>::value;
  static const bool needsBreak = exists_by_supertype<needs_parallel_break_tag, ArgsTy>::value;

  struct IterationContext {
    Galois::Runtime::UserContextAccess<value_type> facing;
    SimpleRuntimeContext ctx;

    void resetUserCtx() {
      if (needsPia)
        facing.resetAlloc();

      if (needsPush)
        facing.getPushBuffer().clear();
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
      StepStats stat {currStep, numIter};
      stat.parallelism += numActivities;

      finishStep(stat);
      ++currStep;
    }

    finishLoop();
  }
    
  void beginLoop() { }

  void finishStep(const StepStats& stat) {
    allSteps.push_back(stat);
    stat.dump(getStatsFile (), loopname);
    workList.switchWorkLists();
    setThreadContext(NULL);
  }

  void finishLoop() {
    setThreadContext(NULL);
  }

  IterationContext& newIteration() const {
    IterationContext* it = new IterationContext();
    
    it->resetUserCtx();
    setThreadContext(&(it->ctx));

    return *it;
  }

  unsigned retireIteration(IterationContext& it, const bool abort) const {

    it.resetUserCtx ();

    unsigned numLocks = 0;
    if (abort) {
      numLocks = it.ctx.cancelIteration();
    } else {
      numLocks = it.ctx.commitIteration();
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
    if (needsPush) {
      for (auto a = it.facing.getPushBuffer().begin(),
          ea = it.facing.getPushBuffer().end(); a != ea; ++a) {
        workList.getNext().push(*a);
      }
    }

    return retireIteration(it, false);
  }

public:
  static_assert(!needsBreak, "not supported by this executor");

  ParaMeterExecutor(FunctionTy& f, const ArgsTy& args): function(f), 
    loopname(get_by_supertype<loopname_tag>(args).value) { }

  template<typename RangeTy>
  void init(const RangeTy& range) {
    workList.getCurr().push_initial(range.begin(), range.end());

    go();

    closeStatsFile ();
  }

  template<typename RangeTy>
  void initThread(const RangeTy& range) { }

  void operator()() { }

private:
  ParaMeterWorkList workList;
  FunctionTy function;
  const char* loopname;
  FILE* pstatsFile;

  IterQueue commitQueue;
  // XXX: may turn out to be unnecessary
  std::vector<StepStats> allSteps;
};


template<typename R, typename F, typename ArgsTuple>
void for_each_param (const R& range, const F& func, const ArgsTuple& argsTuple) {

  using T = typename R::values_type;

  auto tpl = Galois::get_default_trait_values(argsTuple,
      std::make_tuple (loopname_tag {}),
      std::make_tuple (default_loopname {}));

  using Tpl_ty = decltype (tpl);

  using Exec = ParaMeterExecutor<T, F, Tpl_ty>;
  Exec exec (func, tpl);

  exec.init (range);

}


} // end namespace ParaMeter
} // end namespace Runtime

namespace WorkList {

template<class T=int>
class ParaMeter {
public:
  template<bool _concurrent>
  struct rethread { typedef ParaMeter<T> type; };

  template<typename _T>
  struct retype { typedef ParaMeter<_T> type; };

  typedef T value_type;
};

} // end WorkList

namespace Runtime {

template<class T, class FunctionTy, class ArgsTy>
class ForEachExecutor<Galois::WorkList::ParaMeter<T>, FunctionTy, ArgsTy>:
  public ParaMeter::ParaMeterExecutor<T, FunctionTy, ArgsTy> 
{
  typedef ParaMeter::ParaMeterExecutor<T, FunctionTy, ArgsTy> SuperTy;
  ForEachExecutor(const FunctionTy& f, const ArgsTy& args): SuperTy(f, args) { }
};





} // end Runtime;

template <typename I, typename F, typename... Args> 
void for_each_exp (const I& beg, const I& end, const F& func, const Args&... args) {

  auto range = Runtime::makeStandardRange (beg, end);
  auto tpl = std::make_tuple (args...);

  if (Runtime::useParaMeterOpt) {
    Runtime::ParaMeter::for_each_param (range, func, tpl);
  } else {
    Runtime::for_each_gen (range, func, tpl);
  }
}


template <typename C, typename F, typename... Args> 
void for_each_local_exp (C& cont, const F& func, const Args&... args) {

  auto range = Runtime::makeLocalRange(cont);
  auto tpl = std::make_tuple (args...);

  if (Runtime::useParaMeterOpt) {
    Runtime::ParaMeter::for_each_param (range, func, tpl);
  } else {
    Runtime::for_each_gen (range, func, tpl);
  }
}

} // end Galois
#endif
