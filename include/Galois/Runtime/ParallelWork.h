/** Galois scheduler and runtime -*- C++ -*-
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
 * Implementation of the Galois foreach iterator. Includes various 
 * specializations to operators to reduce runtime overhead.
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#ifndef GALOIS_RUNTIME_PARALLELWORK_H
#define GALOIS_RUNTIME_PARALLELWORK_H

#include "Galois/Mem.h"
#include "Galois/Runtime/ForeachTraits.h"
#include "Galois/Runtime/Config.h"
#include "Galois/Runtime/Support.h"
#include "Galois/Runtime/Context.h"
#include "Galois/Runtime/Threads.h"
#include "Galois/Runtime/PerCPU.h"
#include "Galois/Runtime/WorkList.h"
#include "Galois/Runtime/Termination.h"
#include "Galois/Runtime/LoopHooks.h"

#ifdef GALOIS_EXP
#include "Galois/Runtime/SimpleTaskPool.h"
#endif

#include <algorithm>

namespace GaloisRuntime {

class LoopStatistics {
  unsigned long conflicts;
  unsigned long iterations;

public:
  LoopStatistics() :conflicts(0), iterations(0) { }
  void inc_iterations(int amount = 1) {
    iterations += amount;
  }
  void inc_conflicts() {
    ++conflicts;
  }
  void report_stat(unsigned int tid, const char* loopname) const {
    reportStatSum("Conflicts", conflicts, loopname);
    reportStatSum("Iterations", iterations, loopname);
    reportStatAvg("ConflictsThreadDistribution", conflicts, loopname);
    reportStatAvg("IterationsThreadDistribution", iterations, loopname);
  }
};

template<class WorkListTy, class FunctionTy>
class DoAllWork {
  typedef typename WorkListTy::value_type value_type;

  WorkListTy global_wl;
  FunctionTy& fn;

public:
  DoAllWork(FunctionTy& f): fn(f) { }

  template<typename Iter>
  bool AddInitialWork(Iter b, Iter e) {
    global_wl.push_initial(b,e);
    return true;
  }

  void operator()() {
    boost::optional<value_type> p = global_wl.pop();
    while (p) {
      fn(*p);
      p = global_wl.pop();
    }
#ifdef GALOIS_EXP
    SimpleTaskPool& pool = getSystemTaskPool();
    pool.work();
#endif
  }
};

template<class T, class FunctionTy>
class ForEachWorkBase {
protected:
  typedef T value_type;

  struct ThreadLocalData {
    Galois::UserContext<value_type> facing;
    SimpleRuntimeContext cnx;
    LoopStatistics stat;
    TerminationDetection::TokenHolder* lterm;
  };

  FunctionTy& function;
  const char* loopname;

  PerCPU<ThreadLocalData> tdata;
  TerminationDetection term;

  void incrementConflicts(ThreadLocalData& tld) {
    if (ForeachTraits<FunctionTy>::NeedsStats)
      tld.stat.inc_conflicts();
  }

  void incrementIterations(ThreadLocalData& tld) {
    if (ForeachTraits<FunctionTy>::NeedsStats)
      tld.stat.inc_iterations();
  }

  ThreadLocalData& initWorker() {
    ThreadLocalData& tld = tdata.get();
    setThreadContext(&tld.cnx);
    tld.lterm = term.getLocalTokenHolder();
    return tld;
  }

  void deinitWorker() {
    setThreadContext(0);
  }

  void localTermination() {
    term.localTermination();
  }

  bool globalTermination() {
    return term.globalTermination();
  }

  ForEachWorkBase(FunctionTy& f, const char* n):function(f), loopname(n) { }

  virtual ~ForEachWorkBase() {
    if (ForeachTraits<FunctionTy>::NeedsStats) {
      for (unsigned int i = 0; i < GaloisRuntime::ThreadPool::getActiveThreads(); ++i)
        tdata.get(i).stat.report_stat(i, loopname);
      GaloisRuntime::statDone();
    }
  }
};

template<class WorkListTy, class T, class FunctionTy, bool isSimple>
class ForEachWork: public ForEachWorkBase<T, FunctionTy> {
  typedef ForEachWorkBase<T, FunctionTy> Super;
  typedef typename Super::value_type value_type;
  typedef typename Super::ThreadLocalData ThreadLocalData;

  typedef typename WorkListTy::template retype<value_type>::WL WLTy;
  typedef WorkList::LevelStealing<WorkList::FIFO<value_type>, value_type> AbortedList;

  WLTy global_wl;
  AbortedList aborted;
  LL::CacheLineStorage<volatile long> break_happened; //hit flag
  LL::CacheLineStorage<volatile long> abort_happened; //hit flag

  void commitIteration(ThreadLocalData& tld) {
    if (ForeachTraits<FunctionTy>::NeedsPush) {
      global_wl.push(tld.facing.__getPushBuffer().begin(),
		     tld.facing.__getPushBuffer().end());
      tld.facing.__resetPushBuffer();
    }

    if (ForeachTraits<FunctionTy>::NeedsPIA)
      tld.facing.__resetAlloc();
    if (ForeachTraits<FunctionTy>::NeedsBreak && tld.facing.__breakHappened())
      break_happened.data = 1;
    if (ForeachTraits<FunctionTy>::NeedsAborts)
      tld.cnx.commit_iteration();
  }

  void abortIteration(value_type val, ThreadLocalData& tld) {
    assert(ForeachTraits<FunctionTy>::NeedsAborts);

    clearConflictLock();
    if (ForeachTraits<FunctionTy>::NeedsAborts)
      tld.cnx.cancel_iteration();
    Super::incrementConflicts(tld);
    __sync_synchronize();
    aborted.push(val);
    abort_happened.data = 1;
    //don't listen to breaks from aborted iterations
    tld.facing.__resetBreak();
    //clear push buffer
    tld.facing.__resetPushBuffer();

    if (ForeachTraits<FunctionTy>::NeedsPIA)
      tld.facing.__resetAlloc();
  }

  // NB(ddn): Be careful about modifying this function. Small functions can be
  // sensitive to instruction cache behavior which is based on compiler inlining
  // heuristics and therefore fragile
  template<bool doOne>
  void doProcess(boost::optional<value_type> p, ThreadLocalData& tld) {
    try {
      do {
        Super::incrementIterations(tld);
        if (ForeachTraits<FunctionTy>::NeedsAborts)
          tld.cnx.start_iteration();
        Super::function(*p, tld.facing);
        commitIteration(tld);
        if (ForeachTraits<FunctionTy>::NeedsBreak && break_happened.data)
	  break;
        if (doOne)
          break;
        p = global_wl.pop();
      } while (p);
    } catch (ConflictFlag a) {
      abortIteration(*p, tld);
    }
  }

  template<bool isLeader>
  void drainAborted(ThreadLocalData& tld) {
    if (!ForeachTraits<FunctionTy>::NeedsAborts) return;
    if (!isLeader) return;
    if (!abort_happened.data) return;
    tld.lterm->workHappened();
    abort_happened.data = 0;
    boost::optional<value_type> p = aborted.pop();
    while (p) {
      if (ForeachTraits<FunctionTy>::NeedsBreak && break_happened.data) 
	return;
      doProcess<true>(*p, tld);
      p = aborted.pop();
    }
  }

  template<bool isLeader>
  void go() {
    ThreadLocalData& tld = Super::initWorker();
#ifdef GALOIS_EXP
    SimpleTaskPool& pool = getSystemTaskPool();
#endif

    do {
      boost::optional<value_type> p = global_wl.pop();
      if (p)
        tld.lterm->workHappened();
      while (p) {
        if (ForeachTraits<FunctionTy>::NeedsBreak && break_happened.data)
	  goto leaveLoop;
        doProcess<false>(p, tld);
	drainAborted<isLeader>(tld);
	p = global_wl.pop();
      }

      drainAborted<isLeader>(tld);
      if (ForeachTraits<FunctionTy>::NeedsBreak && break_happened.data)
	goto leaveLoop;

#ifdef GALOIS_EXP
      pool.work();
#endif

      Super::localTermination();
    } while (!Super::globalTermination());
  leaveLoop:
    Super::deinitWorker();
  }

public:
  ForEachWork(FunctionTy& _f, const char* _loopname) :Super(_f, _loopname) {
    abort_happened.data = 0;
    break_happened.data = 0;
  }

  template<typename Iter>
  bool AddInitialWork(Iter b, Iter e) {
    global_wl.push_initial(b,e);
    return true;
  }

  void operator()() {
    if (LL::getTID() == 0)
      go<true>();
    else
      go<false>();
  }
};

template<typename T1, typename T2>
struct FillWork {
  T1 b;
  T1 e;
  T2& g;
  unsigned int num;
  unsigned int dist;
  
  FillWork(T1& _b, T1& _e, T2& _g) :b(_b), e(_e), g(_g) {
    unsigned int a = ThreadPool::getActiveThreads();
    dist = std::distance(b, e);
    num = (dist + a - 1) / a; //round up
  }

  void operator()(void) {
    unsigned int id = LL::getTID();
    T1 b2 = b;
    T1 e2 = b;
    //stay in bounds
    unsigned int A = std::min(num * id, dist);
    unsigned int B = std::min(num * (id + 1), dist);
    std::advance(b2, A);
    std::advance(e2, B);
    g.AddInitialWork(b2,e2);
  }
};

template<typename WLTy, typename IterTy, typename FunctionTy>
void for_each_impl(IterTy b, IterTy e, FunctionTy f, const char* loopname) {
  assert(!inGaloisForEach);

  inGaloisForEach = true;

  typedef typename std::iterator_traits<IterTy>::value_type T;
  const bool simple = 
    !ForeachTraits<FunctionTy>::NeedsAborts &&
    !ForeachTraits<FunctionTy>::NeedsBreak;
  typedef ForEachWork<WLTy,T,FunctionTy,simple> WorkTy;

  WorkTy W(f, loopname);
  FillWork<IterTy, WorkTy> fw2(b, e, W);

  RunCommand w[3];
  w[0].work = Config::ref(fw2);
  w[0].isParallel = true;
  w[0].barrierAfter = true;
  w[0].profile = true;
  w[1].work = Config::ref(W);
  w[1].isParallel = true;
  w[1].barrierAfter = true;
  w[1].profile = true;
  w[2].work = &runAllLoopExitHandlers;
  w[2].isParallel = false;
  w[2].barrierAfter = true;
  w[2].profile = true;
  getSystemThreadPool().run(&w[0], &w[3]);

  inGaloisForEach = false;
}

template<typename WLTy, typename IterTy, typename FunctionTy>
void do_all_impl(IterTy b, IterTy e, FunctionTy f, const char* loopname) {
  assert(!inGaloisForEach);

  inGaloisForEach = true;

  DoAllWork<WLTy, FunctionTy> W(f);

  W.AddInitialWork(b,e);

  RunCommand w[1];
  w[0].work = Config::ref(W);
  w[0].isParallel = true;
  w[0].barrierAfter = true;
  w[0].profile = true;
  getSystemThreadPool().run(&w[0], &w[1]);

  inGaloisForEach = false;
}

}

#endif
