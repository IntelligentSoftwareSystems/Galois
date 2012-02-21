/** Galois scheduler and runtime -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
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

#include <algorithm>


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

namespace GaloisRuntime {

class LoopStatistics {
  unsigned long conflicts;
  unsigned long iterations;
  OnlineStatistics iterationTimes;

public:
  LoopStatistics() :conflicts(0), iterations(0) { }
  void inc_iterations() {
    ++iterations;
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
class ForEachWork {
  typedef typename WorkListTy::value_type value_type;
  typedef WorkList::LevelStealing<WorkList::FIFO<value_type>, value_type> AbortedList;
  
  struct ThreadLocalData {
    Galois::UserContext<value_type> facing;
    SimpleRuntimeContext cnx;
    LoopStatistics stat;
    TerminationDetection::TokenHolder* lterm;
  };

  WorkListTy global_wl;
  FunctionTy& f;
  const char* loopname;

  PerCPU<ThreadLocalData> tdata;
  TerminationDetection term;
  AbortedList aborted;
  LL::CacheLineStorage<volatile long> break_happened; //hit flag
  LL::CacheLineStorage<volatile long> abort_happened; //hit flag

  void finishIteration(bool aborting, value_type val, ThreadLocalData& tld) {
    if (aborting) {
      clearConflictLock();
      tld.cnx.cancel_iteration();
      tld.stat.inc_conflicts();
      __sync_synchronize();
      aborted.push(val);
      abort_happened.data = 1;
      //don't listen to breaks from aborted iterations
      tld.facing.__resetBreak();
      //clear push buffer
      tld.facing.__resetPushBuffer();
    }

    if (ForeachTraits<FunctionTy>::NeedsPush) {
      global_wl.push(tld.facing.__getPushBuffer().begin(),
		     tld.facing.__getPushBuffer().end());
      tld.facing.__resetPushBuffer();
    }

    assert(tld.facing.__getPushBuffer().capacity() == 0);

    // NB: since push buffer uses PIA, reset after getting push buffer
    tld.facing.__resetAlloc();
    if (ForeachTraits<FunctionTy>::NeedsBreak)
      if (tld.facing.__breakHappened())
        break_happened.data = 1;
    if (!aborting)
      tld.cnx.commit_iteration();
  }

  void doProcess(value_type val, ThreadLocalData& tld) {
    tld.stat.inc_iterations();
    tld.cnx.start_iteration();
    bool aborted = false;
    try {
      f(val, tld.facing);
    } catch (int a) {
      aborted = true;
    }
    finishIteration(aborted, val, tld);
  }

  template<bool isLeader>
  inline void drainAborted(ThreadLocalData& tld) {
    if (!isLeader) return;
    if (!abort_happened.data) return;
    tld.lterm->workHappened();
    abort_happened.data = 0;
    boost::optional<value_type> p = aborted.pop();
    while (p) {
      if (ForeachTraits<FunctionTy>::NeedsBreak && break_happened.data) 
	return;
      doProcess(*p, tld);
      p = aborted.pop();
    }
  }

public:
  ForEachWork(FunctionTy& _f, const char* _loopname)
    :f(_f), loopname(_loopname) {
    abort_happened.data = 0;
    break_happened.data = 0;
  }

  ~ForEachWork() {
    for (unsigned int i = 0; i < GaloisRuntime::ThreadPool::getActiveThreads(); ++i)
      tdata.get(i).stat.report_stat(i, loopname);
    GaloisRuntime::statDone();
  }

  template<typename Iter>
  bool AddInitialWork(Iter b, Iter e) {
    global_wl.push_initial(b,e);
    return true;
  }

  template<bool isLeader>
  void go() {
    ThreadLocalData& tld = tdata.get();
    setThreadContext(&tld.cnx);
    tld.lterm = term.getLocalTokenHolder();

    do {
      boost::optional<value_type> p = global_wl.pop();
      if (p)
        tld.lterm->workHappened();
      while (p) {
        if (ForeachTraits<FunctionTy>::NeedsBreak && break_happened.data)
	  goto leaveLoop;
        doProcess(*p, tld);
	drainAborted<isLeader>(tld);
	p = global_wl.pop();
      }

      drainAborted<isLeader>(tld);
      if (ForeachTraits<FunctionTy>::NeedsBreak && break_happened.data)
	goto leaveLoop;

      term.localTermination();
    } while (!term.globalTermination());
  leaveLoop:
    setThreadContext(0);
  }

  void operator()() {
    if (tdata.myEffectiveID() == 0)
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

template<typename WLTy, typename IterTy, typename Function>
void for_each_impl(IterTy b, IterTy e, Function f, const char* loopname) {

  typedef typename WLTy::template retype<typename std::iterator_traits<IterTy>::value_type>::WL aWLTy;

  ForEachWork<aWLTy, Function> GW(f, loopname);
  FillWork<IterTy, ForEachWork<aWLTy, Function> > fw2(b,e,GW);

  RunCommand w[3];
  w[0].work = config::ref(fw2);
  w[0].isParallel = true;
  w[0].barrierAfter = true;
  w[0].profile = true;
  w[1].work = config::ref(GW);
  w[1].isParallel = true;
  w[1].barrierAfter = true;
  w[1].profile = true;
  w[2].work = &runAllLoopExitHandlers;
  w[2].isParallel = false;
  w[2].barrierAfter = true;
  w[2].profile = true;
  getSystemThreadPool().run(&w[0], &w[3]);
}

}

#endif
