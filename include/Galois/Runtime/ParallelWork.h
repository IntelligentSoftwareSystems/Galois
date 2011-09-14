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
#include <numeric>
#include <sstream>
#include <math.h>

#include "Galois/TypeTraits.h"
#include "Galois/Mem.h"
#include "Galois/Runtime/Config.h"
#include "Galois/Runtime/Support.h"
#include "Galois/Runtime/Context.h"
#include "Galois/Runtime/Threads.h"
#include "Galois/Runtime/PerCPU.h"
#include "Galois/Runtime/WorkList.h"
#include "Galois/Runtime/DebugWorkList.h"
#include "Galois/Runtime/Termination.h"
#include "Galois/Runtime/LoopHooks.h"

namespace GaloisRuntime {

class LoopStatistics {
  unsigned long conflicts;
  unsigned long iterations;
public:
  LoopStatistics() :conflicts(0), iterations(0) {}
  void inc_iterations() {
    ++iterations;
  }
  void inc_conflicts() {
    ++conflicts;
  }
  void report_stat(const char* loopname) const {
    reportStatSum("Conflicts", conflicts, loopname);
    reportStatSum("Iterations", iterations, loopname);
    reportStatAvg("ConflictsDistribution", conflicts, loopname);
    reportStatAvg("IterationsDistribution", iterations, loopname);
  }
};

template<typename Function>
struct Configurator {
  enum {
    CollectStats = !Galois::does_not_need_stats<Function>::value,
    NeedsBreak = Galois::needs_parallel_break<Function>::value,
    NeedsPush = !Galois::does_not_need_parallel_push<Function>::value,
    NeedsContext = !Galois::does_not_need_context<Function>::value,
    NeedsPIA = Galois::needs_per_iter_alloc<Function>::value
  };
};

template<class WorkListTy, class Function>
class ForEachWork {
  typedef typename WorkListTy::value_type value_type;
  typedef GaloisRuntime::WorkList::FIFO<value_type, true> AbortedListTy;
  
  struct tldTy {
    Galois::UserContext<value_type> facing;
    SimpleRuntimeContext cnx;
    LoopStatistics stat;
    TerminationDetection::tokenHolder* lterm;
  };

  WorkListTy global_wl;
  Function& f;
  const char* loopname;

  PerCPU<tldTy> tdata;
  TerminationDetection term;
  AbortedListTy aborted;
  volatile long abort_happened; //hit flag

  void doAborted(value_type val) {
    aborted.push(val);
    abort_happened = 1;
  }

  void doProcess(value_type val, tldTy& tld) {
    tld.stat.inc_iterations();
    tld.cnx.start_iteration();
    try {
      f(val, tld.facing);
    } catch (int a) {
      tld.cnx.cancel_iteration();
      tld.stat.inc_conflicts();
      doAborted(val);
      return;
    }
    tld.cnx.commit_iteration();
    global_wl.push(tld.facing.__getPushBuffer().begin(), tld.facing.__getPushBuffer().end());
    tld.facing.__getPushBuffer().clear();
    tld.facing.__resetAlloc();
  }

  void drainAborted(tldTy& tld) {
    tld.lterm->workHappened();
    abort_happened = 0;
    std::pair<bool, value_type> p = aborted.pop();
    while (p.first) {
      doProcess(p.second, tld);
      p = aborted.pop();
    }
  }

  void drainWL() {
    std::pair<bool, value_type> p = global_wl.pop();
    while (p.first)
      p = global_wl.pop();
  }

public:
  template<typename IterTy>
  ForEachWork(IterTy b, IterTy e, Function& _f, const char* _loopname)
    :f(_f), loopname(_loopname), abort_happened(0) {
    global_wl.fill_initial(b, e);
  }
  ForEachWork(Function& _f, const char* _loopname)
    :f(_f), loopname(_loopname), abort_happened(0) {
  }

  ~ForEachWork() {
    for (unsigned int i = 0; i < GaloisRuntime::getSystemThreadPool().getActiveThreads(); ++i)
      tdata.get(i).stat.report_stat(loopname);
    GaloisRuntime::statDone();
  }

  template<typename IT>
  void AddInitialWork(IT& b, IT& e) {
    global_wl.push(b,e);
  }

  //FIXME: Add back in function based typetrait specialization
  template<bool isLeader>
  void go() {
    tldTy& tld = tdata.get();
    setThreadContext(&tld.cnx);
    tld.lterm = term.getLocalTokenHolder();

    do {
      std::pair<bool, value_type> p = global_wl.pop();
      if (p.first)
        tld.lterm->workHappened();
      while (p.first) {
        doProcess(p.second, tld);
	if (isLeader && abort_happened)
	  drainAborted(tld);
	if (tld.facing.__breakHappened())
	  drainWL();
	p = global_wl.pop();
      }

      if (isLeader && abort_happened) {
	drainAborted(tld);
        continue;
      }

      term.localTermination();
    } while (!term.globalTermination());
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
class FillWork {
  public:
  T1 b;
  T1 e;
  T2& g;
  int num;
  int a;
  FillWork(T1& _b, T1& _e, T2& _g) :b(_b), e(_e), g(_g) {
    a = getSystemThreadPool().getActiveThreads();
    num = std::distance(b,e) / a;
  }
  void operator()(void) {
    int id = ThreadPool::getMyID();
    T1 b2 = b;
    std::advance(b2, num * id);
    T1 e2 = b;
    if (id == (a - 1))
      e2 = e;
    else
      std::advance(e2, num * (id + 1));
    g.AddInitialWork(b2,e2);
  }
};

template<typename WLTy, typename IterTy, typename Function>
void for_each_impl(IterTy b, IterTy e, Function f, const char* loopname) {

  typedef typename WLTy::template retype<typename std::iterator_traits<IterTy>::value_type>::WL aWLTy;

  ForEachWork<aWLTy, Function> GW(f, loopname);
  //ForEachWork<aWLTy, Function> GW(f, loopname);

  FillWork<IterTy, ForEachWork<aWLTy, Function> > fw2(b,e,GW);

  runCMD w[3];
  w[0].work = config::ref(fw2);
  w[0].isParallel = true;
  w[0].barrierAfter = false;
  w[1].work = config::ref(GW);
  w[1].isParallel = true;
  w[1].barrierAfter = true;
  w[2].work = &runAllLoopExitHandlers;
  w[2].isParallel = false;
  w[2].barrierAfter = true;

  getSystemThreadPool().run(&w[0], &w[3]);
}

}

#endif
