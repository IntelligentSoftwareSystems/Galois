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
#include "Galois/Runtime/ForEachTraits.h"
#include "Galois/Runtime/Config.h"
#include "Galois/Runtime/Support.h"
#include "Galois/Runtime/Context.h"
#include "Galois/Runtime/Threads.h"
#include "Galois/Runtime/PerCPU.h"
#include "Galois/Runtime/Termination.h"
#include "Galois/Runtime/LoopHooks.h"
#include "Galois/Runtime/WorkList.h"

#ifdef GALOIS_EXP
//#include "Galois/Runtime/SimpleTaskPool.h"
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
  const char* loopname;

public:
  explicit LoopStatistics(const char* ln) :conflicts(0), iterations(0), loopname(ln) { }
  ~LoopStatistics() {
    reportStat(loopname, "Conflicts", conflicts);
    reportStat(loopname, "Iterations", iterations);
  }
  inline void inc_iterations(int amount = 1) {
    iterations += amount;
  }
  inline void inc_conflicts() {
    ++conflicts;
  }
};


template <>
class LoopStatistics<false> {
public:
  explicit LoopStatistics(const char* ln) {}
  inline void inc_iterations() const { }
  inline void inc_conflicts() const { }
};

template<class WorkListTy, class T, class FunctionTy>
class ForEachWork {
protected:
  typedef T value_type;
  typedef typename WorkListTy::template retype<value_type>::WL WLTy;
  typedef WorkList::GFIFO<value_type> AbortedList;

  struct ThreadLocalData {
    GaloisRuntime::UserContextAccess<value_type> facing;
    SimpleRuntimeContext cnx;
    LoopStatistics<ForEachTraits<FunctionTy>::NeedsStats> stat;
    TerminationDetection::TokenHolder* lterm;

    ThreadLocalData(const char* ln) :stat(ln) {}
  };

  WLTy default_wl;
  WLTy& wl;
  FunctionTy& function;
  const char* loopname;

  TerminationDetection term;
  PerPackageStorage<AbortedList> aborted;
  PerThreadStorage<bool> broke;

  inline void commitIteration(ThreadLocalData& tld) {
    if (ForEachTraits<FunctionTy>::NeedsPush) {
      wl.push(tld.facing.getPushBuffer().begin(),
              tld.facing.getPushBuffer().end());
      tld.facing.resetPushBuffer();
    }
    if (ForEachTraits<FunctionTy>::NeedsPIA)
      tld.facing.resetAlloc();
    if (ForEachTraits<FunctionTy>::NeedsAborts)
      tld.cnx.commit_iteration();
  }

  GALOIS_ATTRIBUTE_NOINLINE
  void abortIteration(value_type val, ThreadLocalData& tld, bool recursiveAbort) {
    assert(ForEachTraits<FunctionTy>::NeedsAborts);

    clearConflictLock();
    tld.cnx.cancel_iteration();
    tld.stat.inc_conflicts(); //Class specialization handles opt
    if (recursiveAbort)
      aborted.getRemote(LL::getLeaderForPackage(LL::getPackageForThread(LL::getTID()) / 2))->push(val);
    else
      aborted.getLocal()->push(val);
    //clear push buffer
    if (ForEachTraits<FunctionTy>::NeedsPush)
      tld.facing.resetPushBuffer();
    //reset allocator
    if (ForEachTraits<FunctionTy>::NeedsPIA)
      tld.facing.resetAlloc();
  }

  inline void doProcess(boost::optional<value_type>& p, ThreadLocalData& tld) {
    tld.stat.inc_iterations(); //Class specialization handles opt
    if (ForEachTraits<FunctionTy>::NeedsAborts)
      tld.cnx.start_iteration();
    function(*p, tld.facing.data());
    commitIteration(tld);
  }

  GALOIS_ATTRIBUTE_NOINLINE
  void handleBreak(ThreadLocalData& tld) {
    commitIteration(tld);
    for (unsigned x = 0; x < broke.size(); ++x)
      *broke.getRemote(x) = true;
  }

  bool runQueueSimple(ThreadLocalData& tld) {
    bool workHappened = false;
    boost::optional<value_type> p = wl.pop();
    if (p)
      workHappened = true;
    while (p) {
      doProcess(p, tld);
      p = wl.pop();
    }
    return workHappened;
  }

  template<bool limit, typename WL>
  bool runQueue(ThreadLocalData& tld, WL& lwl, bool recursiveAbort) {
    bool workHappened = false;
    boost::optional<value_type> p = lwl.pop();
    unsigned num = 0;
    if (p)
      workHappened = true;
#if G_USE_EH
    try {
      while (p) {
	doProcess(p, tld);
	if (limit) {
	  ++num;
	  if (num == 32)
	    break;
	}
	p = lwl.pop();
      }
    } catch (ConflictFlag const& i) {
      switch(i) {
      case GaloisRuntime::CONFLICT:
	abortIteration(*p, tld, recursiveAbort);
	break;
      case GaloisRuntime::BREAK:
	handleBreak(tld);
	return false;
      default:
	abort();
      }
    }
#else
    if (setjmp(hackjmp)) {
      abortIteration(*p, tld, recursiveAbort);
      return workHappened;
    }
    while (p) {
      doProcess(p, tld);
      if (limit) {
	++num;
	if (num == 32)
	  break;
      }
      p = lwl.pop();
    }
#endif
    return workHappened;
  }

  GALOIS_ATTRIBUTE_NOINLINE
  bool handleAborts(ThreadLocalData& tld) {
    return runQueue<false>(tld, *aborted.getLocal(), true);
  }

  template<bool checkAbort>
  void go() {
    //Thread Local Data goes on the local stack
    //to be NUMA friendly
    ThreadLocalData tld(loopname);
    if (ForEachTraits<FunctionTy>::NeedsAborts)
      setThreadContext(&tld.cnx);
    tld.lterm = term.getLocalTokenHolder();
#ifdef GALOIS_EXP
//    SimpleTaskPool& pool = getSystemTaskPool();
#endif
    do {
      bool didWork;
      do {
	didWork = false;
	//Run some iterations
	if (ForEachTraits<FunctionTy>::NeedsBreak || ForEachTraits<FunctionTy>::NeedsAborts)
	  didWork = runQueue<checkAbort || ForEachTraits<FunctionTy>::NeedsBreak>(tld, wl, false);
	else //No try/catch
	  didWork = runQueueSimple(tld);
	//Check for break
	if (ForEachTraits<FunctionTy>::NeedsBreak && *broke.getLocal())
	  break;
	//Check for abort
	if (checkAbort)
	  didWork |= handleAborts(tld);
	//Update node color
	if (didWork)
	  tld.lterm->workHappened();
      } while (didWork);
      if (ForEachTraits<FunctionTy>::NeedsBreak && *broke.getLocal())
	break;
#ifdef GALOIS_EXP
      //pool.work();
#endif

      term.localTermination();
    } while ((ForEachTraits<FunctionTy>::NeedsPush 
	     ||ForEachTraits<FunctionTy>::NeedsBreak
	     ||ForEachTraits<FunctionTy>::NeedsAborts)
	     && !term.globalTermination());

    setThreadContext(0);
  }

public:
  ForEachWork(FunctionTy& f, const char* l): wl(default_wl), function(f), loopname(l) { }

  template<typename W>
  ForEachWork(W& w, FunctionTy& f, const char* l): wl(w), function(f), loopname(l) { }

  template<typename Iter>
  void AddInitialWork(Iter b, Iter e) {
    // term.initializeThread();
    wl.push_initial(b,e);
  }

  void operator()() {
    if (LL::isLeaderForPackage(LL::getTID()) &&
	Galois::getActiveThreads() > 1 && 
	ForEachTraits<FunctionTy>::NeedsAborts)
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
  Initializer<IterTy, WorkTy> init(b, e, W);
  RunCommand w[4] = {Config::ref(init), 
		     Config::ref(getSystemBarrier()),
		     Config::ref(W),
		     Config::ref(getSystemBarrier())};
  getSystemThreadPool().run(&w[0], &w[4]);
  runAllLoopExitHandlers();
  inGaloisForEach = false;
}

template<class FunctionTy>
struct DoAllWrapper: public FunctionTy {
  typedef int tt_does_not_need_stats;
  typedef int tt_does_not_need_parallel_push;
  typedef int tt_does_not_need_aborts;
  DoAllWrapper(const FunctionTy& f) :FunctionTy(f) {}

  template<typename T1, typename T2>
  void operator()(T1& t, T2&) {
    FunctionTy::operator()(t);
  }
};

template<typename WLTy, typename IterTy, typename FunctionTy>
void do_all_impl(IterTy b, IterTy e, FunctionTy f, const char* loopname) {
  assert(!inGaloisForEach);

  inGaloisForEach = true;

  typedef typename std::iterator_traits<IterTy>::value_type T;
  typedef ForEachWork<WLTy,T,DoAllWrapper<FunctionTy> > WorkTy;

  DoAllWrapper<FunctionTy> F(f);
  WorkTy W(F, loopname);

  Initializer<IterTy, WorkTy> init(b, e, W);
  RunCommand w[4] = {Config::ref(init),
		     Config::ref(getSystemBarrier()),
		     Config::ref(W),
		     Config::ref(getSystemBarrier())};
  getSystemThreadPool().run(&w[0], &w[4]);
  runAllLoopExitHandlers();

  inGaloisForEach = false;
}

template<typename FunctionTy>
struct WOnEach {
  FunctionTy& fn;
  WOnEach(FunctionTy& f) :fn(f) {}
  void operator()(void) const {
    fn(GaloisRuntime::LL::getTID(), 
       Galois::getActiveThreads());   
  }
};

template<typename FunctionTy>
void on_each_impl(FunctionTy fn, const char* loopname = 0) {
  WOnEach<FunctionTy> fw(fn);
  GaloisRuntime::RunCommand w[2] = {Config::ref(fw),
				    Config::ref(getSystemBarrier())};
  GaloisRuntime::getSystemThreadPool().run(&w[0], &w[2]);
}


struct WPreAlloc {
  int n;
  WPreAlloc(int m) :n(m) {}
  void operator()() {
    GaloisRuntime::MM::pagePreAlloc(n);
  }
};

static inline void preAlloc_impl(int num) {
  int a = Galois::getActiveThreads();
  a = (num + a - 1) / a;
  WPreAlloc P(a);
  GaloisRuntime::RunCommand w[2] = {Config::ref(P),
				    Config::ref(getSystemBarrier())};
  GaloisRuntime::getSystemThreadPool().run(&w[0], &w[2]);
}




} // end namespace

#endif

