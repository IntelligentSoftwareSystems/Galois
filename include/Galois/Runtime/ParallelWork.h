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
#include "Galois/Runtime/Termination.h"
#include "Galois/Runtime/LoopHooks.h"
#include "Galois/Runtime/WorkList.h"

#ifdef GALOIS_EXP
#include "Galois/Runtime/SimpleTaskPool.h"
#endif

#include <algorithm>

namespace GaloisRuntime {

template<typename T1, typename T2>
struct Initializer {
  T1 b;
  T1 e;
  T2& g;
  
  Initializer(const T1& _b, const T1& _e, T2& _g) :b(_b), e(_e), g(_g) {}

  void operator()(void) {
    g.AddInitialWork(b, e);
  }
};

template <bool Enabled> 
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


template <>
class LoopStatistics<false> {
public:
  inline void inc_iterations () const {}

  inline void inc_conflicts () const {}

  inline void report_stat (unsigned int tid, const char* loopname) const {}
};

template<class T, class FunctionTy>
class ForEachWorkBase {
protected:
  typedef T value_type;

  struct ThreadLocalData {
    Galois::UserContext<value_type> facing;
    SimpleRuntimeContext cnx;
    LoopStatistics<ForeachTraits<FunctionTy>::NeedsStats> stat;
    TerminationDetection::TokenHolder* lterm;

    void incrementConflicts() {
      if (ForeachTraits<FunctionTy>::NeedsStats)
	stat.inc_conflicts();
    }

    void incrementIterations() {
      if (ForeachTraits<FunctionTy>::NeedsStats)
	stat.inc_iterations();
    }
   
  };

  FunctionTy& function;
  const char* loopname;
  TerminationDetection term;

  PerCPU<ThreadLocalData> tdata;

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

template<class WorkListTy, class T, class FunctionTy>
class ForEachWork {
protected:
  typedef T value_type;
  typedef typename WorkListTy::template retype<value_type>::WL WLTy;
  typedef WorkList::FIFO<value_type, true> AbortedList;

  struct ThreadLocalData {
    Galois::UserContext<value_type> facing;
    SimpleRuntimeContext cnx;
    LoopStatistics<ForeachTraits<FunctionTy>::NeedsStats> stat;
    TerminationDetection::TokenHolder* lterm;
  };

  FunctionTy& function;
  const char* loopname;

  TerminationDetection term;
  WLTy wl;
  AbortedList aborted;

  LL::CacheLineStorage<volatile long> break_happened; //hit flag
  LL::CacheLineStorage<volatile long> abort_happened; //hit flag

  inline void commitIteration(ThreadLocalData& tld) {
    if (ForeachTraits<FunctionTy>::NeedsPush) {
      wl.push(tld.facing.__getPushBuffer().begin(),
		     tld.facing.__getPushBuffer().end());
      tld.facing.__resetPushBuffer();
    }
    if (ForeachTraits<FunctionTy>::NeedsPIA)
      tld.facing.__resetAlloc();
    if (ForeachTraits<FunctionTy>::NeedsAborts)
      tld.cnx.commit_iteration();
  }

  GALOIS_ATTRIBUTE_NOINLINE
  void abortIteration(value_type val, ThreadLocalData& tld) {
    assert(ForeachTraits<FunctionTy>::NeedsAborts);

    clearConflictLock();
    tld.cnx.cancel_iteration();
    if (ForeachTraits<FunctionTy>::NeedsStats)
      tld.stat.inc_conflicts();
    aborted.push(val);
    __sync_synchronize();
    abort_happened.data = 1;
    //clear push buffer
    if (ForeachTraits<FunctionTy>::NeedsPush)
      tld.facing.__resetPushBuffer();
    //reset allocator
    if (ForeachTraits<FunctionTy>::NeedsPIA)
      tld.facing.__resetAlloc();
  }

  inline void doProcess(boost::optional<value_type>& p, ThreadLocalData& tld) {
    if (ForeachTraits<FunctionTy>::NeedsStats)
      tld.stat.inc_iterations();
    if (ForeachTraits<FunctionTy>::NeedsAborts)
      tld.cnx.start_iteration();
    function(*p, tld.facing);
    commitIteration(tld);
  }

  GALOIS_ATTRIBUTE_NOINLINE
  void handleBreak() {
    //set flag and empty worklist
    break_happened.data = 1;
    boost::optional<value_type> p;
    do {
      p = wl.pop();
    } while (p);
    do {
      p = aborted.pop();
    } while (p);
    do {
      p = wl.pop();
    } while (p);
  }

  template<unsigned limit, typename WL>
  bool runQueue(ThreadLocalData& tld, WL& lwl) {
    boost::optional<value_type> p = lwl.pop();
    bool workHappened = false;
    if (p)
      workHappened = true;
    try {
      if (!limit) { //compile time optimization
	while (p) {
	  doProcess(p, tld);
	  p = lwl.pop();
	};
       } else {
       	unsigned num = limit;
       	while (p && num) {
       	  doProcess(p, tld);
       	  if (--num)
       	    p = lwl.pop();
       	}
      }
    } catch (ConflictFlag i) {
      switch(i) {
      case GaloisRuntime::CONFLICT:
	abortIteration(*p, tld);
	break;
      case GaloisRuntime::BREAK:
	handleBreak();
	workHappened = false; // short circuit caller
	break;
      default:
	assert(0 && "Unhandled throw");
	abort();
	break;
      }
    }
    return workHappened;
  }

  GALOIS_ATTRIBUTE_NOINLINE
  void handleAborts(ThreadLocalData& tld) {
    abort_happened.data = 0;
    runQueue<0>(tld, aborted);
  }

  template<bool checkAbort>
  void go() {
    //Thread Local Data goes on the local stack
    //to be NUMA friendly
    ThreadLocalData tld;
    setThreadContext(&tld.cnx);
    tld.lterm = term.getLocalTokenHolder();
#ifdef GALOIS_EXP
    SimpleTaskPool& pool = getSystemTaskPool();
    pool.initializeThread();
#endif

    do {
      bool didWork = false;
      do {
	if (checkAbort)
	  didWork = runQueue<255>(tld, wl);
	else 
	  didWork = runQueue<0>(tld, wl);
	if (didWork)
	  tld.lterm->workHappened();
	if (checkAbort && abort_happened.data && 
	    (!ForeachTraits<FunctionTy>::NeedsBreak || break_happened.data)) {
	  tld.lterm->workHappened();
	  handleAborts(tld);
	}
      } while (didWork);
      if (ForeachTraits<FunctionTy>::NeedsBreak && break_happened.data) {
	handleBreak();
	break;
      }
#ifdef GALOIS_EXP
      pool.work();
#endif
      term.localTermination();
    } while (!term.globalTermination());

    setThreadContext(0);
    if (ForeachTraits<FunctionTy>::NeedsStats)
      tld.stat.report_stat(LL::getTID(), loopname);
  }

public:
  ForEachWork(FunctionTy& _f, const char* _loopname) 
    :loopname(_loopname), function(_f)
  {
    abort_happened.data = 0;
    break_happened.data = 0;
  }

  ~ForEachWork() {
    if (ForeachTraits<FunctionTy>::NeedsStats)
      GaloisRuntime::statDone();
  }

  template<typename Iter>
  void AddInitialWork(Iter b, Iter e) {
    wl.initializeThread();
    wl.push_initial(b,e);
  }

  void operator()() {
    if (LL::getTID() == 0 &&
	ThreadPool::getActiveThreads() > 1 && 
	ForeachTraits<FunctionTy>::NeedsAborts)
      go<true>();
    else
      go<false>();
  }

};


template<typename WLTy, typename IterTy, typename FunctionTy>
void for_each_impl(IterTy b, IterTy e, FunctionTy f, const char* loopname) {
  assert(!inGaloisForEach);

  inGaloisForEach = true;

  typedef typename std::iterator_traits<IterTy>::value_type T;
  typedef ForEachWork<WLTy,T,FunctionTy> WorkTy;

  WorkTy W(f, loopname);
  RunCommand w[3];

  Initializer<IterTy, WorkTy> fw2(b, e, W);
  w[0].work = Config::ref(fw2);
  w[0].isParallel = true;
  w[0].barrierAfter = true;
  w[0].profile = false;
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

template<class FunPar>
struct doAllWrapper : public FunPar {
  typedef int tt_does_not_need_stats;
  typedef int tt_does_not_need_parallel_push;
  typedef int tt_does_not_need_aborts;
  doAllWrapper(const FunPar& f) :FunPar(f) {}

  template<typename T1, typename T2>
  void operator()(T1& t, T2&) {
    FunPar::operator()(t);
  }
};

template<typename WLTy, typename IterTy, typename FunctionTy>
void do_all_impl(IterTy b, IterTy e, FunctionTy f, const char* loopname) {
  assert(!inGaloisForEach);

  inGaloisForEach = true;

  typedef typename std::iterator_traits<IterTy>::value_type T;
  typedef ForEachWork<WLTy,T,doAllWrapper<FunctionTy>,true> WorkTy;

  doAllWrapper<FunctionTy> F(f);
  WorkTy W(F, loopname);

  Initializer<IterTy, WorkTy > fw2(b, e, W);
  RunCommand w[2];
  w[0].work = Config::ref(fw2);
  w[0].isParallel = true;
  w[0].barrierAfter = true;
  w[0].profile = false;
  w[1].work = Config::ref(W);
  w[1].isParallel = true;
  w[1].barrierAfter = true;
  w[1].profile = false;
  getSystemThreadPool().run(&w[0], &w[2]);

  inGaloisForEach = false;
}


} // end namespace

#endif
